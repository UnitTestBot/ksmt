package org.ksmt.cache.weak

import org.ksmt.cache.CacheRemoveHandler
import java.lang.ref.Reference
import java.lang.ref.ReferenceQueue

abstract class WeakHashMapCache<K : Any, V : Any> {
    private val data = HashMap<Any, KeyRefNode<K, V>>()
    private val keyReferenceQueue = ReferenceQueue<K>()
    private var modificationsSinceLastCleanup = 0
    private var removeHandler: CacheRemoveHandler<K, V>? = null

    abstract fun lookupKey(key: K): Any

    abstract fun newNode(key: K, referenceQueue: ReferenceQueue<K>, value: V, hash: Int): KeyRefNode<K, V>

    abstract class KeyRefNode<K : Any, V : Any>(private val keyReference: Reference<K>) {
        fun getKeyReference(): Any = keyReference
        fun getKey(): K? = keyReference.get()

        abstract fun getValue(): V
        abstract fun setValue(value: V)
    }

    fun addRemoveHandler(handler: CacheRemoveHandler<K, V>) {
        removeHandler = handler
    }

    fun get(key: K): V? = getNode(key)?.getValue()

    fun put(key: K, value: V, onlyIfAbsent: Boolean): V? {
        val lookupKey = lookupKey(key)
        val current = data.get(lookupKey)

        return if (current == null) {
            val node = newNode(key, keyReferenceQueue, value, lookupKey.hashCode())
            data.put(node.getKeyReference(), node)
            afterWrite()
            null
        } else if (onlyIfAbsent) {
            current.getValue()
        } else {
            val oldValue = current.getValue()
            current.setValue(value)
            oldValue
        }
    }

    fun internKey(key: K, valueStub: V): K {
        val lookupKey = lookupKey(key)
        val current = data.get(lookupKey)
        val currentKey = current?.getKey()

        if (current == null || currentKey == null) {
            val node = newNode(key, keyReferenceQueue, valueStub, lookupKey.hashCode())
            data.put(node.getKeyReference(), node)

            if (current == null) {
                afterWrite()
            } else {
                cleanupReferences()
            }

            return key
        }
        return currentKey
    }

    private fun getNode(key: K): KeyRefNode<K, V>? {
        val lookupKey = lookupKey(key)
        return data.get(lookupKey)
    }

    private fun notifyRemove(key: K?, value: V) {
        removeHandler?.onRemove(key, value)
    }

    private fun afterWrite() {
        if (modificationsSinceLastCleanup++ >= MODIFICATIONS_TO_CLEANUP) {
            modificationsSinceLastCleanup = 0
            cleanupReferences()
        }
    }

    private fun cleanupReferences() {
        while (true) {
            val keyRef = keyReferenceQueue.poll() ?: break
            val node = data.remove(keyRef)
            if (node != null) {
                notifyRemove(node.getKey(), node.getValue())
            }
        }
    }

    companion object {
        private const val MODIFICATIONS_TO_CLEANUP = 16
    }
}
