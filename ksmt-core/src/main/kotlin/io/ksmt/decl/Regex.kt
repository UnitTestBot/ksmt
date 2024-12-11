package io.ksmt.decl

import io.ksmt.KContext
import io.ksmt.expr.KApp
import io.ksmt.expr.KExpr
import io.ksmt.sort.KRegexSort
import io.ksmt.sort.KStringSort

class KRegexConcatDecl internal constructor(
    ctx: KContext,
) : KFuncDecl2<KRegexSort, KRegexSort, KRegexSort>(
    ctx,
    name = "regex_concat",
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

class KRegexKleeneClosureDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KRegexSort, KRegexSort>(ctx, "closure", ctx.mkRegexSort(), ctx.mkRegexSort()) {
    override fun KContext.apply(arg: KExpr<KRegexSort>): KApp<KRegexSort, KRegexSort> = mkRegexKleeneClosureNoSimplify(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KRegexKleeneCrossDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KRegexSort, KRegexSort>(ctx, "kleene_cross", ctx.mkRegexSort(), ctx.mkRegexSort()) {
    override fun KContext.apply(arg: KExpr<KRegexSort>): KApp<KRegexSort, KRegexSort> = mkRegexKleeneCrossNoSimplify(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KRegexDifferenceDecl internal constructor(
    ctx: KContext,
) : KFuncDecl2<KRegexSort, KRegexSort, KRegexSort>(
    ctx,
    name = "diff",
    resultSort = ctx.mkRegexSort(),
    ctx.mkRegexSort(),
    ctx.mkRegexSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KRegexSort>,
        arg1: KExpr<KRegexSort>
    ): KApp<KRegexSort, *> = mkRegexDifferenceNoSimplify(arg0, arg1)
}

class KRegexComplementDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KRegexSort, KRegexSort>(ctx, "comp", ctx.mkRegexSort(), ctx.mkRegexSort()) {
    override fun KContext.apply(arg: KExpr<KRegexSort>): KApp<KRegexSort, KRegexSort> = mkRegexComplementNoSimplify(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KRegexOptionDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KRegexSort, KRegexSort>(ctx, "opt", ctx.mkRegexSort(), ctx.mkRegexSort()) {
    override fun KContext.apply(arg: KExpr<KRegexSort>): KApp<KRegexSort, KRegexSort> = mkRegexOptionNoSimplify(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KRangeDecl internal constructor(
    ctx: KContext,
) : KFuncDecl2<KRegexSort, KStringSort, KStringSort>(
    ctx,
    name = "range",
    resultSort = ctx.mkRegexSort(),
    ctx.mkStringSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KStringSort>
    ): KApp<KRegexSort, *> = mkRangeNoSimplify(arg0, arg1)
}

class KEpsilonDecl internal constructor(
    ctx: KContext
) : KConstDecl<KRegexSort>(ctx, "eps", ctx.mkRegexSort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KRegexSort, *> = ctx.mkEpsilon()
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KAllDecl internal constructor(
    ctx: KContext
) : KConstDecl<KRegexSort>(ctx, "all", ctx.mkRegexSort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KRegexSort, *> = ctx.mkAll()
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KAllCharDecl internal constructor(
    ctx: KContext
) : KConstDecl<KRegexSort>(ctx, "all_char", ctx.mkRegexSort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KRegexSort, *> = ctx.mkAllChar()
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
