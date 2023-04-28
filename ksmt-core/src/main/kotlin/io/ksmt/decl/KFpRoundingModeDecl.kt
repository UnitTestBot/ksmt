package io.ksmt.decl

import io.ksmt.KContext
import io.ksmt.expr.KApp
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFpRoundingMode
import io.ksmt.sort.KFpRoundingModeSort

class KFpRoundingModeDecl(
    ctx: KContext,
    val value: KFpRoundingMode
) : KConstDecl<KFpRoundingModeSort>(ctx, value.modeName, ctx.mkFpRoundingModeSort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KFpRoundingModeSort, *> = ctx.mkFpRoundingModeExpr(value)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
