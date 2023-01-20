package org.ksmt.cache

import org.ksmt.KAst
import org.ksmt.KContext
import org.ksmt.cache.weak.ConcurrentWeakCache
import org.ksmt.cache.weak.WeakCache
import java.util.concurrent.ConcurrentHashMap

interface AstCache<K, V : Any> where K : KAst, K : KInternedObject {
    fun get(ast: K): V?
    fun put(ast: K, value: V): V?
    fun putIfAbsent(ast: K, value: V): V?
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

    override fun get(ast: K): V? = cache.get(ast)
    override fun put(ast: K, value: V): V? = cache.put(ast, value)
    override fun putIfAbsent(ast: K, value: V): V? = cache.putIfAbsent(ast, value)
    override fun registerOnDeleteHandler(handler: CacheRemoveHandler<K, V>) {
        // Entries are never deleted
    }
}

class SingleThreadNoGcAstCache<K, V : Any> : AstCache<K, V> where K : KAst, K : KInternedObject {
    private val cache = HashMap<K, V>()

    override fun get(ast: K): V? = cache.get(ast)
    override fun put(ast: K, value: V): V? = cache.put(ast, value)
    override fun putIfAbsent(ast: K, value: V): V? = cache.putIfAbsent(ast, value)
    override fun registerOnDeleteHandler(handler: CacheRemoveHandler<K, V>) {
        // Entries are never deleted
    }
}

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
