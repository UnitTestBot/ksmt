package io.ksmt.cache.weak

import io.ksmt.cache.KInternedObject
import java.lang.ref.Reference
import java.lang.ref.ReferenceQueue

class WeakInterner<T : KInternedObject> : WeakHashMapCache<T, Any>() {

    fun intern(obj: T): T = internKey(obj, valueStub)

    override fun lookupKey(key: T): Any = WeakInternerKeyLookup(key)

    override fun newNode(key: T, referenceQueue: ReferenceQueue<T>, value: Any, hash: Int): KeyRefNode<T, Any> {
        val keyRef = WeakInternerKeyRef(key, referenceQueue, hash)
        return InternerNode(keyRef)
    }

    private class InternerNode<T : KInternedObject>(keyRef: Reference<T>) : KeyRefNode<T, Any>(keyRef) {

        override fun getValue(): Any = valueStub

        override fun setValue(value: Any) {
            // Values are not used for interning.
        }
    }

    companion object {
        private val valueStub = Any()
    }
}
