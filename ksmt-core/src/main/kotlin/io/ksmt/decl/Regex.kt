package io.ksmt.decl

import io.ksmt.KContext
import io.ksmt.expr.KApp
import io.ksmt.expr.KExpr
import io.ksmt.sort.KRegexSort

class KRegexLiteralDecl internal constructor(
    ctx: KContext,
    val value: String
) : KConstDecl<KRegexSort>(ctx, value, ctx.mkRegexSort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KRegexSort, *> = ctx.mkRegexLiteral(value)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
