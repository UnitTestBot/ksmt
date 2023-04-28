package io.ksmt.cache

import io.ksmt.KAst
import io.ksmt.KContext
import io.ksmt.cache.weak.ConcurrentWeakInterner
import io.ksmt.cache.weak.WeakInterner
import java.util.concurrent.ConcurrentHashMap

/**
 * Interner for [KAst].
 * Ensures that if any two objects are equals according to [KInternedObject.internEquals],
 * they are equal by reference (actually the same object).
 *
 * See [mkAstInterner] for interner creation.
 * */
interface AstInterner<T> where T : KAst, T : KInternedObject {

    /**
     * Intern provided [ast].
     *
     * If there are no objects, which are equal to [ast]
     * according to [KInternedObject.internEquals] save and return the provided [ast].
     * Otherwise, return the previously exising object.
     * */
    fun intern(ast: T): T
}

class ConcurrentGcAstInterner<T> : AstInterner<T> where T : KAst, T : KInternedObject {
    private val interner by lazy { ConcurrentWeakInterner<T>() }
    override fun intern(ast: T): T = interner.intern(ast)
}

class SingleThreadGcAstInterner<T> : AstInterner<T> where T : KAst, T : KInternedObject {
    private val interner by lazy { WeakInterner<T>() }
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

/**
 * Select the proper [AstInterner] implementation according to
 * required [operationMode] and [astManagementMode].
 * */
fun <T> mkAstInterner(
    operationMode: KContext.OperationMode,
    astManagementMode: KContext.AstManagementMode
): AstInterner<T> where T : KAst, T : KInternedObject = when (operationMode) {
    KContext.OperationMode.SINGLE_THREAD -> when (astManagementMode) {
        KContext.AstManagementMode.GC -> SingleThreadGcAstInterner()
        KContext.AstManagementMode.NO_GC -> SingleThreadNoGcAstInterner()
    }
    KContext.OperationMode.CONCURRENT -> when (astManagementMode) {
        KContext.AstManagementMode.GC -> ConcurrentGcAstInterner()
        KContext.AstManagementMode.NO_GC -> ConcurrentNoGcAstInterner()
    }
}
