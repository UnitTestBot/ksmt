package org.ksmt.cache.weak

import org.ksmt.cache.KInternedObject
import org.ksmt.cache.weak.WeakInternerKey.Companion.objectEquality
import java.lang.ref.ReferenceQueue
import java.lang.ref.WeakReference


internal interface WeakInternerKey<T : KInternedObject> {
    fun get(): T?

    companion object {
        @JvmStatic
        fun objectEquality(lhs: KInternedObject?, rhs: Any?): Boolean =
            (lhs === rhs) || (lhs !== null && rhs !== null && lhs.internEquals(rhs))
    }
}

internal class WeakInternerKeyRef<T : KInternedObject>(
    key: T?, queue: ReferenceQueue<T>?,
    private val hashCode: Int
) : WeakReference<T>(key, queue), WeakInternerKey<T> {
    override fun equals(other: Any?): Boolean = when {
        other === this -> true
        other is WeakInternerKey<*> -> objectEquality(get(), other.get())
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
        other is WeakInternerKey<*> -> objectEquality(key, other.get())
        else -> false
    }

    override fun hashCode(): Int = hashCode

    override fun toString(): String =
        "{key=${get()} hash=$hashCode}"
}
