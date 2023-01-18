package org.ksmt.cache

import org.ksmt.KAst

class AstInterner<T> where T : KAst, T : KInternedObject {
    private val interner by lazy { ConcurrentWeakInterner<T>() }
    fun intern(ast: T): T = interner.intern(ast)
}

fun <T> mkAstInterner(): AstInterner<T> where T : KAst, T : KInternedObject = AstInterner()
