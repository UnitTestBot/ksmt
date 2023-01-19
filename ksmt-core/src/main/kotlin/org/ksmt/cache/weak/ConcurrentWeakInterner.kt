package org.ksmt.cache.weak

import org.ksmt.cache.KInternedObject
import java.lang.ref.Reference
import java.lang.ref.ReferenceQueue

class ConcurrentWeakInterner<T : KInternedObject> : ConcurrentWeakHashMapCache<T, Any>() {

    fun intern(obj: T): T = internKey(obj, valueStub)

    override fun lookupKey(key: T): Any = WeakInternerKeyLookup(key)

    override fun newNode(key: T, referenceQueue: ReferenceQueue<T>, value: Any): KeyRefNode<T, Any> {
        val keyRef = WeakInternerKeyRef(key, referenceQueue)
        return InternerNode(keyRef)
    }

    private class InternerNode<T : KInternedObject>(keyRef: Reference<T>) : KeyRefNode<T, Any>(keyRef) {

        override fun getValue(): Any = valueStub

        override fun setValue(value: Any) {
        }
    }

    companion object {
        private val valueStub = Any()
    }
}
