package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

open class KConstDecl<T : KSort>(
    ctx: KContext,
    name: String,
    sort: T
) : KFuncDecl<T>(ctx, name, sort, emptyList()) {
    fun apply() = apply(emptyList())

    override fun apply(args: List<KExpr<*>>): KApp<T, *> {
        require(args.isEmpty()) { "Constant must have no arguments" }

        return ctx.mkConstApp(this@KConstDecl)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
