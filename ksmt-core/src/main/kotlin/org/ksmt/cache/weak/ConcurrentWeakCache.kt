package org.ksmt.cache.weak

import org.ksmt.cache.KInternedObject
import java.lang.ref.Reference
import java.lang.ref.ReferenceQueue

class ConcurrentWeakCache<K : KInternedObject, V : Any> : ConcurrentWeakHashMapCache<K, V>() {
    override fun lookupKey(key: K): Any = WeakCacheKeyLookup(key)

    override fun newNode(key: K, referenceQueue: ReferenceQueue<K>, value: V): KeyRefNode<K, V> {
        val keyRef = WeakCacheKeyRef(key, referenceQueue)
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
}
