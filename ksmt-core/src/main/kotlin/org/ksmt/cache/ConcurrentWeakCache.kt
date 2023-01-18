package org.ksmt.cache

import java.lang.ref.Reference
import java.lang.ref.ReferenceQueue
import java.lang.ref.WeakReference

class ConcurrentWeakCache<K : KInternedObject, V : Any> : ConcurrentWeakHashMapCache<K, V>() {
    override fun lookupKey(key: K): Any = LookupCacheKey(key)

    override fun newNode(key: K, referenceQueue: ReferenceQueue<K>, value: V): KeyRefNode<K, V> {
        val keyRef = WeakCacheKey(key, referenceQueue)
        return CacheNode(keyRef, value)
    }

    private class CacheNode<T : KInternedObject, V : Any>(
        keyRef: Reference<T>,
        private var value: V
    ) : KeyRefNode<T, V>(keyRef) {

        override fun getValue(): V = value

        override fun setValue(value: V) {
            this.value = value
        }
    }

    private interface CacheKey<K : KInternedObject> {
        fun get(): K?
    }

    private class WeakCacheKey<K : KInternedObject>(
        key: K?, queue: ReferenceQueue<K>?
    ) : WeakReference<K>(key, queue), CacheKey<K> {
        private val hashCode: Int = System.identityHashCode(key)

        override fun equals(other: Any?): Boolean = when {
            other === this -> true
            other is CacheKey<*> -> get() === other.get()
            else -> false
        }

        override fun hashCode(): Int = hashCode

        override fun toString(): String =
            "{key=${get()} hash=$hashCode}"
    }

    private class LookupCacheKey<K : KInternedObject>(
        private val key: K
    ) : CacheKey<K> {
        private val hashCode: Int = System.identityHashCode(key)

        override fun get(): K = key

        override fun equals(other: Any?): Boolean = when {
            other === this -> true
            other is CacheKey<*> -> get() === other.get()
            else -> false
        }

        override fun hashCode(): Int = hashCode

        override fun toString(): String =
            "{key=${get()} hash=$hashCode}"
    }
}
