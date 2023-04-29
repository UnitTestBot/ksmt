package io.ksmt.cache

import io.ksmt.KAst
import io.ksmt.KContext
import io.ksmt.cache.weak.ConcurrentWeakCache
import io.ksmt.cache.weak.WeakCache
import java.util.concurrent.ConcurrentHashMap

/**
 * A cache specialized to use [KAst] as a key.
 *
 * See [mkAstCache] for cache creation.
 * */
interface AstCache<K, V : Any> where K : KAst, K : KInternedObject {

    /**
     * Find a value associated with a [ast].
     * */
    fun get(ast: K): V?

    /**
     * Store a value for the [ast].
     * Return a previously associated value or null, if nothing was associated.
     * */
    fun put(ast: K, value: V): V?

    /**
     * Store a value for the [ast] only if nothing was previously associated.
     * Return a previously associated value or null, if nothing was associated.
     * */
    fun putIfAbsent(ast: K, value: V): V?

    /**
     * Register a handler which is triggered when an entry is removed from the cache.
     * */
    fun registerOnDeleteHandler(handler: CacheRemoveHandler<K, V>)
}

class ConcurrentGcAstCache<K, V : Any> : AstCache<K, V> where K : KAst, K : KInternedObject {
    private val cache = ConcurrentWeakCache<K, V>()

    override fun get(ast: K): V? = cache.get(ast)
    override fun put(ast: K, value: V): V? = cache.put(ast, value, onlyIfAbsent = false)
    override fun putIfAbsent(ast: K, value: V): V? = cache.put(ast, value, onlyIfAbsent = true)
    override fun registerOnDeleteHandler(handler: CacheRemoveHandler<K, V>) {
        cache.addRemoveHandler(handler)
    }
}

class SingleThreadGcAstCache<K, V : Any> : AstCache<K, V> where K : KAst, K : KInternedObject {
    private val cache = WeakCache<K, V>()

    override fun get(ast: K): V? = cache.get(ast)
    override fun put(ast: K, value: V): V? = cache.put(ast, value, onlyIfAbsent = false)
    override fun putIfAbsent(ast: K, value: V): V? = cache.put(ast, value, onlyIfAbsent = true)
    override fun registerOnDeleteHandler(handler: CacheRemoveHandler<K, V>) {
        cache.addRemoveHandler(handler)
    }
}

class ConcurrentNoGcAstCache<K, V : Any> : AstCache<K, V> where K : KAst, K : KInternedObject {
    private val cache = ConcurrentHashMap<K, V>()

    override fun get(ast: K): V? = cache[ast]
    override fun put(ast: K, value: V): V? = cache.put(ast, value)
    override fun putIfAbsent(ast: K, value: V): V? = cache.putIfAbsent(ast, value)
    override fun registerOnDeleteHandler(handler: CacheRemoveHandler<K, V>) {
        // Entries are never deleted
    }
}

class SingleThreadNoGcAstCache<K, V : Any> : AstCache<K, V> where K : KAst, K : KInternedObject {
    private val cache = HashMap<K, V>()

    override fun get(ast: K): V? = cache[ast]
    override fun put(ast: K, value: V): V? = cache.put(ast, value)
    override fun putIfAbsent(ast: K, value: V): V? = cache.putIfAbsent(ast, value)
    override fun registerOnDeleteHandler(handler: CacheRemoveHandler<K, V>) {
        // Entries are never deleted
    }
}

/**
 * Select the proper [AstCache] implementation according to
 * required [operationMode] and [astManagementMode].
 * */
fun <K, V : Any> mkAstCache(
    operationMode: KContext.OperationMode,
    astManagementMode: KContext.AstManagementMode
): AstCache<K, V> where K : KAst, K : KInternedObject = when (operationMode) {
    KContext.OperationMode.SINGLE_THREAD -> when (astManagementMode) {
        KContext.AstManagementMode.GC -> SingleThreadGcAstCache()
        KContext.AstManagementMode.NO_GC -> SingleThreadNoGcAstCache()
    }
    KContext.OperationMode.CONCURRENT -> when (astManagementMode) {
        KContext.AstManagementMode.GC -> ConcurrentGcAstCache()
        KContext.AstManagementMode.NO_GC -> ConcurrentNoGcAstCache()
    }
}
