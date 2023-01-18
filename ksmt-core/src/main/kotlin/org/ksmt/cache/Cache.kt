package org.ksmt.cache

import com.github.benmanes.caffeine.cache.Cache
import com.github.benmanes.caffeine.cache.Caffeine
import com.github.benmanes.caffeine.cache.Interner
import com.github.benmanes.caffeine.cache.mkWeakCustomEqualityInterner
import org.ksmt.KAst

fun <T> mkAstInterner(): Interner<T> where T : KAst, T : KInternedObject =
    mkWeakCustomEqualityInterner()

fun <K, V> mkAstCache(): Cache<K, V> where K : KAst, K : KInternedObject =
    Caffeine
        .newBuilder()
        .executor(Runnable::run)
        .weakKeys()
        .build()

fun <K, V> mkCache(): Cache<K, V> =
    Caffeine
        .newBuilder()
        .build()
