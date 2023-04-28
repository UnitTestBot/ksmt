package io.ksmt.cache

import io.ksmt.KContext
import java.util.concurrent.ConcurrentHashMap


fun <K, V> mkCache(operationMode: KContext.OperationMode): MutableMap<K, V> =
    when (operationMode) {
        KContext.OperationMode.SINGLE_THREAD -> HashMap()
        KContext.OperationMode.CONCURRENT -> ConcurrentHashMap<K, V>()
    }
