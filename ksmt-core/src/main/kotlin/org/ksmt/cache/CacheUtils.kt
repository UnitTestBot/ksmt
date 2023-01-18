package org.ksmt.cache

import org.ksmt.KAst
import java.util.concurrent.ConcurrentHashMap

fun <K, V : Any> mkAstCache() where K : KAst, K : KInternedObject = ConcurrentWeakCache<K, V>()

fun <K, V> mkCache() = ConcurrentHashMap<K, V>()
