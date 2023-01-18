package com.github.benmanes.caffeine.cache

import com.github.benmanes.caffeine.cache.References.InternalReference
import org.ksmt.cache.KInternedObject
import java.lang.ref.Reference
import java.lang.ref.ReferenceQueue
import java.lang.ref.WeakReference

private val cacheNodeFactory by lazy {
    BoundedLocalCache::class.java.declaredFields
        .first { it.name == "nodeFactory" }
        .also { it.isAccessible = true }
}

internal class WeakKeyCustomEqualityReference<K : KInternedObject>(
    key: K?, queue: ReferenceQueue<K>?
) : WeakReference<K>(key, queue), InternalReference<K> {
    private val hashCode: Int = key?.internHashCode() ?: 0

    override fun getKeyReference(): Any = this

    override fun equals(other: Any?): Boolean = when {
        other === this -> true
        other is InternalReference<*> -> KInternedObject.objectEquality(get(), other.get())
        else -> false
    }

    override fun hashCode(): Int = hashCode

    override fun toString(): String =
        "{key=${get()} hash=$hashCode}"
}

internal class LookupKeyCustomEqualityReference<K : KInternedObject>(
    private val key: K
) : InternalReference<K> {
    private val hashCode: Int = key.internHashCode()

    override fun get(): K = key

    override fun getKeyReference(): Any = this

    override fun equals(other: Any?): Boolean = when {
        other === this -> true
        other is InternalReference<*> -> KInternedObject.objectEquality(get(), other.get())
        else -> false
    }

    override fun hashCode(): Int = hashCode

    override fun toString(): String =
        "{key=${get()} hash=$hashCode}"
}

internal class CustomEqualityObjectInterned<K : KInternedObject> : Node<K, Any?>, NodeFactory<K, Any?> {
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

private const val INTERNER_INITIAL_CAPACITY = 64

@Suppress("UNCHECKED_CAST")
private fun <K : KInternedObject> mkWeakCustomEqualityCache(): BoundedLocalCache<K, Any?> {
    val builder = Caffeine.newBuilder()
        .executor(Runnable::run)
        .initialCapacity(INTERNER_INITIAL_CAPACITY)
        .weakKeys()
        .also { it.interner = true }
    val internCache = LocalCacheFactory.newBoundedLocalCache(builder,  null,  false)

    val customEqualityNodeFactory = CustomEqualityObjectInterned<K>()
    cacheNodeFactory.set(internCache, customEqualityNodeFactory)

    return internCache as BoundedLocalCache<K, Any?>
}

internal class WeakCustomEqualityInterner<E: KInternedObject> : Interner<E> {
    private val cache by lazy { mkWeakCustomEqualityCache<E>() }

    override fun intern(sample: E): E {
        while (true) {
            val value = cache.putIfAbsent(sample, CustomEqualityObjectInterned.valueStub)

            //  No previously associated value -> [sample] is a new unique object.
            if (value == null) {
                return sample
            }

            // Search for previously interned object
            val canonical = cache.getKey(sample)
            if (canonical != null) {
                return canonical
            }
            // Previously interned object was removed. Retry interning
        }
    }
}

fun <T : KInternedObject> mkWeakCustomEqualityInterner(): Interner<T> =
    WeakCustomEqualityInterner()
