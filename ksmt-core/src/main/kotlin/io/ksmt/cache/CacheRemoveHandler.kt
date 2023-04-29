package io.ksmt.cache

/**
 * The onRemove event handler in [AstCache].
 *
 * See [AstCache.registerOnDeleteHandler].
 * */
interface CacheRemoveHandler<K, V> {
    fun onRemove(key: K?, value: V)
}
