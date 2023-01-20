package org.ksmt.cache.weak

import org.ksmt.cache.CacheRemoveHandler
import java.lang.ref.Reference
import java.lang.ref.ReferenceQueue
import java.lang.ref.WeakReference
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.locks.ReentrantLock

abstract class ConcurrentWeakHashMapCache<K : Any, V : Any> {
    private val data = ConcurrentHashMap<Any, KeyRefNode<K, V>>()
    private val keyReferenceQueue = ReferenceQueue<K>()
    private val cleanupLock = ReentrantLock()
    private val cleanupStatus = AtomicInteger(IDLE)
    private val modificationsSinceLastCleanup = AtomicInteger(0)
    private var removeHandler: CacheRemoveHandler<K, V>? = null

    abstract fun lookupKey(key: K): Any

    abstract fun newNode(key: K, referenceQueue: ReferenceQueue<K>, value: V, hash: Int): KeyRefNode<K, V>

    abstract class KeyRefNode<K : Any, V : Any>(keyRef: Reference<K>) {
        @Volatile
        private var keyReference: Reference<out K> = keyRef

        fun getKeyReference(): Any = keyReference

        fun getKey(): K? = keyReference.get()

        abstract fun getValue(): V
        abstract fun setValue(value: V)

        fun isAlive(): Boolean = keyReference !== deadRef

        fun isDead(): Boolean = keyReference === deadRef

        fun die() {
            val keyRef = keyReference
            keyReference = deadRef
            keyRef.clear()
        }

        companion object {
            @JvmStatic
            private val deadRef: Reference<Nothing> = WeakReference(null)
        }
    }

    fun addRemoveHandler(handler: CacheRemoveHandler<K, V>) {
        removeHandler = handler
    }

    fun get(key: K): V? = getNode(key)?.getValue()

    fun put(key: K, value: V, onlyIfAbsent: Boolean): V? = putUtil(key, value, onlyIfAbsent)

    fun internKey(key: K, valueStub: V): K = internUtil(key, valueStub)

    private fun getNode(key: K): KeyRefNode<K, V>? {
        val lookupKey = lookupKey(key)
        val node = data.get(lookupKey)
        afterRead()
        return node
    }

    @Suppress("LoopWithTooManyJumpStatements")
    private fun putUtil(key: K, value: V, onlyIfAbsent: Boolean): V? {
        var node: KeyRefNode<K, V>? = null
        val lookupKey = lookupKey(key)

        while (true) {
            var current = data.get(lookupKey)

            if (current == null) {

                if (node == null) {
                    node = newNode(key, keyReferenceQueue, value, lookupKey.hashCode())
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

    @Suppress("NestedBlockDepth")
    private fun internUtil(key: K, value: V): K {
        var node: KeyRefNode<K, V>? = null
        val lookupKey = lookupKey(key)

        while (true) {
            var current = data.get(lookupKey)

            if (current == null) {

                if (node == null) {
                    node = newNode(key, keyReferenceQueue, value, lookupKey.hashCode())
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
                    scheduleCleanup()
                }

            } else {
                // Acquire previously interned object
                val currentKey = current.getKey()
                if (currentKey != null) return currentKey

                // Previously interned object was removed. Try cleanup and retry interning
                scheduleCleanup()
            }
        }
    }

    private fun updateValueIfAlive(node: KeyRefNode<K, V>, value: V): V? = synchronized(node) {
        if (!node.isAlive()) return null
        val oldValue = node.getValue()
        node.setValue(value)
        oldValue
    }

    private fun notifyRemove(key: K?, value: V) {
        removeHandler?.onRemove(key, value)
    }

    private fun afterRead() {
        runCleanupIfRequired()
    }

    private fun afterWrite() {
        if (modificationsSinceLastCleanup.incrementAndGet() >= MODIFICATIONS_TO_CLEANUP) {
            modificationsSinceLastCleanup.set(0)
            scheduleCleanup()
        } else {
            runCleanupIfRequired()
        }
    }

    private fun runCleanupIfRequired() {
        if (cleanupStatus.get() == REQUIRED) {
            runCleanup()
        }
    }

    private fun scheduleCleanup() {
        while (true) {
            when (cleanupStatus.get()) {
                // No ongoing cleanup -> schedule and run
                IDLE -> {
                    cleanupStatus.compareAndSet(IDLE, REQUIRED)
                    runCleanup()
                    return
                }
                // No ongoing cleanup and cleanup is already scheduled -> run
                REQUIRED -> {
                    runCleanup()
                    return
                }
                // Cleanup is running. Try to reschedule cleanup after completion.
                PROCESSING_TO_IDLE -> {
                    if (cleanupStatus.compareAndSet(PROCESSING_TO_IDLE, PROCESSING_TO_REQUIRED)) {
                        return
                    }
                    // Cleanup status changed. Retry
                }
                // Cleanup is running and will be rescheduled right after completion.
                PROCESSING_TO_REQUIRED -> {
                    return
                }
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

    private fun cleanupEntry(node: KeyRefNode<K, V>) {
        val keyRef = node.getKeyReference()
        val key = node.getKey()
        var nodeResurrected = false
        var removed = false

        data.computeIfPresent(keyRef) { _, newNode ->
            // We have a new node associated with the key. Our node was already removed.
            if (newNode != node) return@computeIfPresent newNode

            synchronized(newNode) {
                // Key is reachable for some reason. Don't remove node
                if (key != null) {
                    nodeResurrected = true
                    return@computeIfPresent newNode
                }

                // Mark node as removed
                node.die()
                removed = true
            }

            // Remove node from data
            null
        }

        if (nodeResurrected) {
            return
        }

        synchronized(node) {
            // Mark node as removed
            if (!node.isDead()) {
                node.die()
            }
        }

        if (removed) {
            notifyRemove(key, node.getValue())
        }
    }

    companion object {
        private const val MODIFICATIONS_TO_CLEANUP = 16

        //A cleanup is not taking place.
        private const val IDLE = 0

        // A cleanup is required due to write modification.
        private const val REQUIRED = 1

        // A cleanup is in progress and will transition to idle.
        private const val PROCESSING_TO_IDLE = 2

        // A cleanup is in progress and will transition to required.
        private const val PROCESSING_TO_REQUIRED = 3
    }
}
