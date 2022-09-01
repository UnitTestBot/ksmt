package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.sort.KFpRoundingModeSort

class KFpRoundingModeDecl(ctx: KContext, val value: KFpRoundingMode) :
    KConstDecl<KFpRoundingModeSort>(ctx, value.modeName, ctx.mkFpRoundingModeSort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KFpRoundingModeSort, *> = ctx.mkFpRoundingModeExpr(value)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
