package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.sort.KSort

abstract class KConstDecl<T : KSort>(
    ctx: KContext,
    name: String,
    sort: T
) : KFuncDecl<T>(ctx, name, sort, emptyList()) {
    fun apply() = apply(emptyList())

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
