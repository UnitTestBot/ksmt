package org.ksmt.cache

import org.ksmt.KAst
import org.ksmt.KContext
import java.util.concurrent.ConcurrentHashMap

interface AstInterner<T> where T : KAst, T : KInternedObject {
    fun intern(ast: T): T
}

class ConcurrentGcAstInterner<T> : AstInterner<T> where T : KAst, T : KInternedObject {
    private val interner by lazy { ConcurrentWeakInterner<T>() }
    override fun intern(ast: T): T = interner.intern(ast)
}

private class NoGcInternKey<T : KInternedObject>(private val key: T) {
    override fun hashCode(): Int = key.internHashCode()
    override fun equals(other: Any?): Boolean =
        other != null && key.internEquals((other as NoGcInternKey<*>).key)
}

class SingleThreadNoGcAstInterner<T> : AstInterner<T> where T : KAst, T : KInternedObject {
    private val interner by lazy { HashMap<NoGcInternKey<T>, T>() }
    override fun intern(ast: T): T = interner.putIfAbsent(NoGcInternKey(ast), ast) ?: ast
}

class ConcurrentNoGcAstInterner<T> : AstInterner<T> where T : KAst, T : KInternedObject {
    private val interner by lazy { ConcurrentHashMap<NoGcInternKey<T>, T>() }
    override fun intern(ast: T): T = interner.putIfAbsent(NoGcInternKey(ast), ast) ?: ast
}

fun <T> mkAstInterner(
    operationMode: KContext.OperationMode,
    astManagementMode: KContext.AstManagementMode
): AstInterner<T> where T : KAst, T : KInternedObject = when (operationMode) {
    KContext.OperationMode.SINGLE_THREAD -> when (astManagementMode) {
        KContext.AstManagementMode.GC -> TODO()
        KContext.AstManagementMode.NO_GC -> SingleThreadNoGcAstInterner()
    }
    KContext.OperationMode.CONCURRENT -> when (astManagementMode) {
        KContext.AstManagementMode.GC -> ConcurrentGcAstInterner()
        KContext.AstManagementMode.NO_GC -> ConcurrentNoGcAstInterner()
    }
}
