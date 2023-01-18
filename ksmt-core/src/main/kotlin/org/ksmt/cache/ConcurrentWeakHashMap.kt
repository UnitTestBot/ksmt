package org.ksmt.cache

import java.lang.ref.ReferenceQueue
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.locks.ReentrantLock

abstract class Node<K : Any, V : Any> {
    abstract fun getKeyReference(): Any
    abstract fun getKey(): K?
    abstract fun getValue(): V
    abstract fun setValue(value: V)
    abstract fun isAlive(): Boolean
    abstract fun isDead(): Boolean
    abstract fun retire()
    abstract fun die()
}

interface RemoveHandler<K, V> {
    fun onRemove(key: K?, value: V)
}

/** A cleanup is not taking place.  */
private const val IDLE = 0

/** A cleanup is required due to write modification.  */
private const val REQUIRED = 1

/** A cleanup is in progress and will transition to idle.  */
private const val PROCESSING_TO_IDLE = 2

/** A cleanup is in progress and will transition to required.  */
private const val PROCESSING_TO_REQUIRED = 3

abstract class ConcurrentWeakHashMapCache<K : Any, V : Any> {
    private val data = ConcurrentHashMap<Any, Node<K, V>>()
    private val keyReferenceQueue = ReferenceQueue<K>()
    private val cleanupLock = ReentrantLock()
    private val cleanupStatus = AtomicInteger(IDLE)
    private var removeHandler: RemoveHandler<K, V>? = null

    abstract fun lookupKey(key: K): Any

    abstract fun newNode(key: K, referenceQueue: ReferenceQueue<K>, value: V): Node<K, V>

    fun addRemoveHandler(handler: RemoveHandler<K, V>) {
        removeHandler = handler
    }

    fun get(key: K): V? = getNode(key)?.getValue()

    fun put(key: K, value: V, onlyIfAbsent: Boolean): V? {
        val lookupKey = lookupKey(key)
        return putUtil(key, value, onlyIfAbsent, lookupKey)
    }

    fun internKey(key: K, valueStub: V): K {
        val lookupKey = lookupKey(key)
        return internUtil(key, valueStub, lookupKey)
    }

    private fun getNode(key: K): Node<K, V>? {
        val lookupKey = lookupKey(key)
        val node = data.get(lookupKey)
        afterRead()
        return node
    }

    private fun putUtil(
        key: K,
        value: V,
        onlyIfAbsent: Boolean,
        lookupKey: Any,
    ): V? {
        var node: Node<K, V>? = null

        while (true) {
            var current = data.get(lookupKey)

            if (current == null) {

                if (node == null) {
                    node = newNode(key, keyReferenceQueue, value)
                }

                current = data.putIfAbsent(node.getKeyReference(), node)

                if (current == null) {
                    // Data was successfully modified
                    afterWrite()
                    return null
                } else if (onlyIfAbsent) {
                    /**
                     *  Data was modified after previous get and we have a node
                     *  but we don't need to modify it
                     *  */
                    return current.getValue().also {
                        afterRead()
                    }
                }

            } else if (onlyIfAbsent) {
                // We have a node and we don't need to modify it
                return current.getValue().also {
                    afterRead()
                }
            }

            // We have a node and we need to modify it

            if (!current.isAlive()) {
                // Current node is scheduled for removal. retry put
                continue
            }

            val oldValue = updateValueIfAlive(current, value)

            // Current node is scheduled for removal, retry put
            if (oldValue == null) {
                continue
            }

            notifyRemove(key, oldValue)
            afterRead()
            return oldValue
        }
    }

    private fun internUtil(
        key: K,
        value: V,
        lookupKey: Any
    ): K {
        var node: Node<K, V>? = null

        while (true) {
            var current = data.get(lookupKey)

            if (current == null) {

                if (node == null) {
                    node = newNode(key, keyReferenceQueue, value)
                }

                current = data.putIfAbsent(node.getKeyReference(), node)

                if (current == null) {
                    //  No previously associated value -> [key] is a new unique object.
                    afterWrite()
                    return key
                } else {
                    // Acquire previously interned object
                    val currentKey = current.getKey()
                    if (currentKey != null) return currentKey

                    // Previously interned object was removed. Try cleanup and retry interning
                    afterWrite()
                }

            } else {
                // Acquire previously interned object
                val currentKey = current.getKey()
                if (currentKey != null) return currentKey

                // Previously interned object was removed. Try cleanup and retry interning
                afterWrite()
            }
        }
    }

    private fun updateValueIfAlive(node: Node<K, V>, value: V): V? = synchronized(node) {
        if (!node.isAlive()) return null
        val oldValue = node.getValue()
        node.setValue(value)
        oldValue
    }

    private fun notifyRemove(key: K?, value: V) {
        removeHandler?.onRemove(key, value)
    }

    private fun afterRead() {
        if (cleanupStatus.get() == REQUIRED) {
            runCleanup()
        }
    }

    private fun afterWrite() {
        var status = cleanupStatus.get()
        while (true) {
            when (status) {
                IDLE -> {
                    cleanupStatus.compareAndSet(IDLE, REQUIRED)
                    runCleanup()
                    return
                }

                REQUIRED -> {
                    runCleanup()
                    return
                }

                PROCESSING_TO_IDLE -> {
                    if (cleanupStatus.compareAndSet(PROCESSING_TO_IDLE, PROCESSING_TO_REQUIRED)) {
                        return
                    }
                    status = cleanupStatus.get()
                    continue
                }

                PROCESSING_TO_REQUIRED -> return
            }
        }
    }

    private inline fun ifNotProcessing(body: () -> Unit) {
        val status = cleanupStatus.get()
        if (status == PROCESSING_TO_IDLE || status == PROCESSING_TO_REQUIRED) {
            return
        }
        body()
    }

    private fun runCleanup() = ifNotProcessing {
        if (cleanupLock.tryLock()) {
            try {
                ifNotProcessing {
                    cleanup()
                }
            } finally {
                cleanupLock.unlock()
            }
        }
    }


    private fun cleanup() {
        cleanupStatus.set(PROCESSING_TO_IDLE)
        try {
            drainKeyReferences()
        } finally {
            if (!cleanupStatus.compareAndSet(PROCESSING_TO_IDLE, IDLE)) {
                cleanupStatus.set(REQUIRED)
            }
        }
    }

    private fun drainKeyReferences() {
        while (true) {
            val keyRef = keyReferenceQueue.poll() ?: break
            val node = data.get(keyRef)
            if (node != null) {
                cleanupEntry(node)
            }
        }
    }

    private fun cleanupEntry(node: Node<K, V>) {
        val keyRef = node.getKeyReference()
        val key = node.getKey()
        var nodeResurrected = false
        var removed = false

        data.computeIfPresent(keyRef) { _, newNode ->
            if (newNode != node) return@computeIfPresent newNode

            synchronized(newNode) {
                if (key != null) {
                    nodeResurrected = true
                    return@computeIfPresent newNode
                }

                removed = true
                node.retire()
            }

            null
        }

        if (nodeResurrected) {
            return
        }

        makeDead(node)

        if (removed) {
            notifyRemove(key, node.getValue())
        }
    }

    private fun makeDead(node: Node<K, V>) = synchronized(node) {
        if (node.isDead()) return
        node.die()
    }
}
