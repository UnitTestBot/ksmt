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

class KRegexConcatDecl internal constructor(
    ctx: KContext,
) : KFuncDecl2<KRegexSort, KRegexSort, KRegexSort>(
    ctx,
    name = "concat",
    resultSort = ctx.mkRegexSort(),
    ctx.mkRegexSort(),
    ctx.mkRegexSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KRegexSort>,
        arg1: KExpr<KRegexSort>
    ): KApp<KRegexSort, *> = mkRegexConcatNoSimplify(arg0, arg1)
}

class KRegexUnionDecl internal constructor(
    ctx: KContext,
) : KFuncDecl2<KRegexSort, KRegexSort, KRegexSort>(
    ctx,
    name = "union",
    resultSort = ctx.mkRegexSort(),
    ctx.mkRegexSort(),
    ctx.mkRegexSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KRegexSort>,
        arg1: KExpr<KRegexSort>
    ): KApp<KRegexSort, *> = mkRegexUnionNoSimplify(arg0, arg1)
}

class KRegexIntersectionDecl internal constructor(
    ctx: KContext,
) : KFuncDecl2<KRegexSort, KRegexSort, KRegexSort>(
    ctx,
    name = "intersect",
    resultSort = ctx.mkRegexSort(),
    ctx.mkRegexSort(),
    ctx.mkRegexSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KRegexSort>,
        arg1: KExpr<KRegexSort>
    ): KApp<KRegexSort, *> = mkRegexIntersectionNoSimplify(arg0, arg1)
}
