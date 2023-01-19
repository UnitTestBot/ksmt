package org.ksmt.cache

import java.lang.ref.Reference
import java.lang.ref.ReferenceQueue
import java.lang.ref.WeakReference

class ConcurrentWeakInterner<T : KInternedObject> : ConcurrentWeakHashMapCache<T, Any>() {

    fun intern(obj: T): T = internKey(obj, valueStub)

    override fun lookupKey(key: T): Any = LookupInternerKey(key)

    override fun newNode(key: T, referenceQueue: ReferenceQueue<T>, value: Any): KeyRefNode<T, Any> {
        val keyRef = WeakInternerKey(key, referenceQueue)
        return InternerNode(keyRef)
    }

    private class InternerNode<T : KInternedObject>(keyRef: Reference<T>) : KeyRefNode<T, Any>(keyRef) {

        override fun getValue(): Any = valueStub

        override fun setValue(value: Any) {
        }
    }

    private interface InternerKey<T : KInternedObject> {
        fun get(): T?
    }

    private class WeakInternerKey<T : KInternedObject>(
        key: T?, queue: ReferenceQueue<T>?
    ) : WeakReference<T>(key, queue), InternerKey<T> {
        private val hashCode: Int = key?.internHashCode() ?: 0

        override fun equals(other: Any?): Boolean = when {
            other === this -> true
            other is InternerKey<*> -> objectEquality(get(), other.get())
            else -> false
        }

        override fun hashCode(): Int = hashCode

        override fun toString(): String =
            "{key=${get()} hash=$hashCode}"
    }

    private class LookupInternerKey<T : KInternedObject>(
        private val key: T
    ) : InternerKey<T> {
        private val hashCode: Int = key.internHashCode()

        override fun get(): T = key

        override fun equals(other: Any?): Boolean = when {
            other === this -> true
            other is InternerKey<*> -> objectEquality(get(), other.get())
            else -> false
        }

        override fun hashCode(): Int = hashCode

        override fun toString(): String =
            "{key=${get()} hash=$hashCode}"
    }

    companion object {
        private val valueStub = Any()

        @JvmStatic
        fun objectEquality(lhs: KInternedObject?, rhs: Any?): Boolean =
            (lhs === rhs) || (lhs !== null && rhs !== null && lhs.internEquals(rhs))
    }
}
