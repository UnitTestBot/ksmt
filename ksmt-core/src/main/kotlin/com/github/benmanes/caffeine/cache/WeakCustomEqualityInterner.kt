package com.github.benmanes.caffeine.cache

import com.github.benmanes.caffeine.cache.References.InternalReference
import org.ksmt.cache.CustomObjectEquality
import java.lang.ref.Reference
import java.lang.ref.ReferenceQueue
import java.lang.ref.WeakReference

private val cacheNodeFactory by lazy {
    BoundedLocalCache::class.java.declaredFields
        .first { it.name == "nodeFactory" }
        .also { it.isAccessible = true }
}

internal class WeakKeyCustomEqualityReference<K : CustomObjectEquality>(
    key: K?, queue: ReferenceQueue<K>?
) : WeakReference<K>(key, queue), InternalReference<K> {
    private val hashCode: Int = key?.customHashCode() ?: 0

    override fun getKeyReference(): Any = this

    override fun equals(other: Any?): Boolean = when {
        other === this -> true
        other is InternalReference<*> -> CustomObjectEquality.objectEquality(get(), other.get())
        else -> false
    }

    override fun hashCode(): Int = hashCode

    override fun toString(): String =
        "{key=${get()} hash=$hashCode}"
}

internal class LookupKeyCustomEqualityReference<K : CustomObjectEquality>(
    private val key: K
) : InternalReference<K> {
    private val hashCode: Int = key.customHashCode()

    override fun get(): K = key

    override fun getKeyReference(): Any = this

    override fun equals(other: Any?): Boolean = when {
        other === this -> true
        other is InternalReference<*> -> CustomObjectEquality.objectEquality(get(), other.get())
        else -> false
    }

    override fun hashCode(): Int = hashCode

    override fun toString(): String =
        "{key=${get()} hash=$hashCode}"
}

internal class CustomEqualityObjectInterned<K : CustomObjectEquality> : Node<K, Any?>, NodeFactory<K, Any?> {
    @Volatile
    private var keyReference: Reference<*>? = null

    constructor()

    constructor(keyReference: Reference<K>?) {
        this.keyReference = keyReference
    }

    @Suppress("UNCHECKED_CAST")
    override fun getKey(): K? = keyReference?.get() as K?

    override fun getKeyReference(): Any? = keyReference

    override fun getValue(): Any = valueStub

    override fun getValueReference(): Any = valueStub

    override fun setValue(value: Any?, referenceQueue: ReferenceQueue<Any?>?) {}

    override fun containsValue(value: Any?): Boolean = value == getValue()

    override fun newNode(
        key: K?, keyReferenceQueue: ReferenceQueue<K>?,
        value: Any?, valueReferenceQueue: ReferenceQueue<Any?>?,
        weight: Int, now: Long
    ): Node<K, Any?> = CustomEqualityObjectInterned<K>(WeakKeyCustomEqualityReference(key, keyReferenceQueue))

    @Suppress("UNCHECKED_CAST")
    override fun newNode(
        keyReference: Any?, value: Any?,
        valueReferenceQueue: ReferenceQueue<Any?>?,
        weight: Int, now: Long
    ): Node<K, Any?> = CustomEqualityObjectInterned(keyReference as Reference<K>?)

    @Suppress("UNCHECKED_CAST")
    override fun newLookupKey(key: Any): Any =
        LookupKeyCustomEqualityReference(key as K)

    override fun newReferenceKey(key: K?, referenceQueue: ReferenceQueue<K>?): Any =
        WeakKeyCustomEqualityReference(key, referenceQueue)

    override fun isAlive(): Boolean {
        val keyRef = keyReference
        return keyRef !== NodeFactory.RETIRED_WEAK_KEY && keyRef !== NodeFactory.DEAD_WEAK_KEY
    }

    override fun isRetired(): Boolean = keyReference === NodeFactory.RETIRED_WEAK_KEY

    override fun retire() {
        val keyRef = keyReference
        keyReference = NodeFactory.RETIRED_WEAK_KEY
        keyRef?.clear()
    }

    override fun isDead(): Boolean = keyReference === NodeFactory.DEAD_WEAK_KEY

    override fun die() {
        val keyRef = keyReference
        keyReference = NodeFactory.DEAD_WEAK_KEY
        keyRef?.clear()
    }

    companion object {
        @JvmStatic
        internal val valueStub = Any()
    }
}

@Suppress("UNCHECKED_CAST")
private fun <K : CustomObjectEquality> mkWeakCustomEqualityCache(): BoundedLocalCache<K, Any?> {
    val internCache = Caffeine.newWeakInterner<K>()
    val customEqualityNodeFactory = CustomEqualityObjectInterned<K>()
    cacheNodeFactory.set(internCache, customEqualityNodeFactory)
    return internCache as BoundedLocalCache<K, Any?>
}


internal class WeakCustomEqualityInterner<E: CustomObjectEquality> : Interner<E> {
    private val cache = mkWeakCustomEqualityCache<E>()

    override fun intern(sample: E): E {
        while (true) {
            val canonical = cache.getKey(sample)
            if (canonical != null) {
                return canonical
            }
            val value = cache.putIfAbsent(sample, CustomEqualityObjectInterned.valueStub)

            if (value == null) {
                return sample
            }
        }
    }
}

fun <T : CustomObjectEquality> mkWeakCustomEqualityInterner(): Interner<T> =
    WeakCustomEqualityInterner()
