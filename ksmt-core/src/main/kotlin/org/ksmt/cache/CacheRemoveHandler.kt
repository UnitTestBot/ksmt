package org.ksmt.cache

interface CacheRemoveHandler<K, V> {
    fun onRemove(key: K?, value: V)
}
