package org.ksmt.cache.weak

import org.ksmt.cache.KInternedObject
import java.lang.ref.ReferenceQueue
import java.lang.ref.WeakReference

internal interface WeakCacheKey<K : KInternedObject> {
    fun get(): K?
}

internal class WeakCacheKeyRef<K : KInternedObject>(
    key: K?, queue: ReferenceQueue<K>?,
    /**
     * All [WeakCacheKeyRef] usages must ensure that [hashCode]
     * is computed in the same way as for [WeakCacheKeyLookup].
     * */
    private val hashCode: Int
) : WeakReference<K>(key, queue), WeakCacheKey<K> {

    override fun equals(other: Any?): Boolean = when {
        other === this -> true
        other is WeakCacheKey<*> -> get() === other.get()
        else -> false
    }

    override fun hashCode(): Int = hashCode

    override fun toString(): String =
        "{key=${get()} hash=$hashCode}"
}

internal class WeakCacheKeyLookup<K : KInternedObject>(
    private val key: K?
) : WeakCacheKey<K> {
    private val hashCode: Int = System.identityHashCode(key)

    override fun get(): K? = key

    override fun equals(other: Any?): Boolean = when {
        other === this -> true
        other is WeakCacheKey<*> -> key === other.get()
        else -> false
    }

    override fun hashCode(): Int = hashCode

    override fun toString(): String =
        "{key=${get()} hash=$hashCode}"
}
