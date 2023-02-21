package org.ksmt.cache.weak

import org.ksmt.cache.KInternedObject
import java.lang.ref.ReferenceQueue
import java.lang.ref.WeakReference


internal interface WeakInternerKey<T : KInternedObject> {
    fun get(): T?
}

internal class WeakInternerKeyRef<T : KInternedObject>(
    key: T?, queue: ReferenceQueue<T>?,
    /**
     * All [WeakInternerKeyRef] usages must ensure that [hashCode]
     * is computed in the same way as for [WeakInternerKeyLookup].
     * */
    private val hashCode: Int
) : WeakReference<T>(key, queue), WeakInternerKey<T> {
    override fun equals(other: Any?): Boolean = when {
        other === this -> true
        other is WeakInternerKey<*> -> {
            val lhs = get()
            val rhs = other.get()
            (lhs === rhs) || (lhs !== null && rhs !== null && lhs.internEquals(rhs))
        }
        else -> false
    }

    override fun hashCode(): Int = hashCode

    override fun toString(): String =
        "{key=${get()} hash=$hashCode}"
}

internal class WeakInternerKeyLookup<T : KInternedObject>(
    private val key: T?
) : WeakInternerKey<T> {
    private val hashCode: Int = key?.internHashCode() ?: 0

    override fun get(): T? = key

    override fun equals(other: Any?): Boolean = when {
        other === this -> true
        other is WeakInternerKey<*> -> {
            val rhs = other.get()
            (key === rhs) || (key !== null && rhs !== null && key.internEquals(rhs))
        }
        else -> false
    }

    override fun hashCode(): Int = hashCode

    override fun toString(): String =
        "{key=${get()} hash=$hashCode}"
}
