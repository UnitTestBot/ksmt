package io.ksmt.cache.weak

import io.ksmt.cache.CacheRemoveHandler
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
        val node = data[lookupKey]
        afterRead()
        return node
    }

    @Suppress("LoopWithTooManyJumpStatements")
    private fun putUtil(key: K, value: V, onlyIfAbsent: Boolean): V? {
        var node: KeyRefNode<K, V>? = null
        val lookupKey = lookupKey(key)

        while (true) {
            var current = data[lookupKey]

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
            var current = data[lookupKey]

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
        // We don't care about possible status changes (see this function usages).
        if (status == PROCESSING_TO_IDLE || status == PROCESSING_TO_REQUIRED) {
            return
        }
        body()
    }

    /**
     * Run cleanup if it's not already running (see [ifNotProcessing]).
     *
     * Note: cleanup status may change during status check.
     * We don't care about such changes because according to
     * the [cleanupStatus] automaton we will run cleanup after next read/write operation.
     * */
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

    /**
     * Cleanup staled references.
     * */
    private fun cleanup() {
        /**
         * Forcibly update cleanup status, because:
         * 1. The [cleanupLock] ensures that we don't run [cleanup] in parallel.
         * 2. Cleanup will be performed regardless of the status.
         * 3. If someone requested a cleanup again before this status update we don't care (because of 2).
         * 4. A situation when someone requested a cleanup AFTER status update is handled in finally block.
         * */
        cleanupStatus.set(PROCESSING_TO_IDLE)
        try {
            drainKeyReferences()
        } finally {
            if (!cleanupStatus.compareAndSet(PROCESSING_TO_IDLE, IDLE)) {
                cleanupStatus.set(REQUIRED)
            }
        }
    }

    /**
     * Retrieve and cleanup staled references from [keyReferenceQueue].
     *
     * Several modifications are possible during a cleanup session:
     * 1. put/intern. See [cleanupEntry] for an explanation of why this is safe.
     * 2. GC detected a new unreachable reference.
     * 2.1 Staled reference was added to [keyReferenceQueue] during cleanup session.
     * It is safe, because [ReferenceQueue] is synchronized.
     * 2.2 Staled reference was NOT added to [keyReferenceQueue] during cleanup session.
     * It is also safe, because it will be added to [keyReferenceQueue] after some time and
     * will be cleaned during the next cleanup session.
     * */
    private fun drainKeyReferences() {
        while (true) {
            val keyRef = keyReferenceQueue.poll() ?: break
            val node = data[keyRef]
            if (node != null) {
                cleanupEntry(node)
            }
        }
    }

    /**
     * Cleanup a single staled reference.
     *
     * See the comments in the body of the method for the
     * safety guarantees of possible modifications.
     * */
    private fun cleanupEntry(node: KeyRefNode<K, V>) {
        /**
         * Hold a strong reference to the key object.
         * If for some reasons key is not null (reachable) it won't be
         * collected by GC until the method completes.
         * */
        val key = node.getKey()

        val keyRef = node.getKeyReference()
        var nodeResurrected = false
        var removed = false

        /**
         * Possible situations:
         *
         * 1. [key] is reachable. For some reasons (should be impossible) we try
         * to cleanup an entry, with a reachable key.
         * We detect such situations with a [nodeResurrected] and no cleanup is performed.
         *
         * 2. [key] is null (unreachable). The associated node can't be accessed anywhere else,
         * because it's impossible to construct a lookupKey that is equal to the current key
         * (we have no null keys and therefore `non null` == `null` is always false).
         * */
        data.computeIfPresent(keyRef) { _, newNode ->
            /**
             * We have a new node associated with the key.
             * Our node was already removed.
             * Should be impossible because of the situations described above.
             * */
            if (newNode !== node) return@computeIfPresent newNode

            synchronized(newNode) {
                // Key is reachable for some reason. Don't remove node. (See situation 1)
                if (key != null) {
                    nodeResurrected = true
                    return@computeIfPresent newNode
                }

                // Mark node as removed
                newNode.die()
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
            if (node.isAlive()) {
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
