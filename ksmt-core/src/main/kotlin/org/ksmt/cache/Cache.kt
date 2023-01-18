package org.ksmt.cache

import com.github.benmanes.caffeine.cache.Cache
import com.github.benmanes.caffeine.cache.Caffeine
import com.github.benmanes.caffeine.cache.Interner
import com.github.benmanes.caffeine.cache.mkWeakCustomEqualityInterner
import org.ksmt.KAst

fun <T> mkAstInterner(): Interner<T> where T : KAst, T : CustomObjectEquality =
    mkWeakCustomEqualityInterner()

fun <K : KAst, V> mkAstCache(): Cache<K, V> =
    Caffeine
        .newBuilder()
        .weakKeys()
        .build()
