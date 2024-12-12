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
    "regex_concat",
    ctx.mkRegexSort(),
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
    "regex_union",
    ctx.mkRegexSort(),
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
    "regex_intersect",
    ctx.mkRegexSort(),
    ctx.mkRegexSort(),
    ctx.mkRegexSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KRegexSort>,
        arg1: KExpr<KRegexSort>
    ): KApp<KRegexSort, *> = mkRegexIntersectionNoSimplify(arg0, arg1)
}

class KRegexStarDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KRegexSort, KRegexSort>(
    ctx,
    "regex_star",
    ctx.mkRegexSort(),
    ctx.mkRegexSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg: KExpr<KRegexSort>
    ): KApp<KRegexSort, KRegexSort> = mkRegexStarNoSimplify(arg)
}

class KRegexCrossDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KRegexSort, KRegexSort>(
    ctx,
    "regex_cross",
    ctx.mkRegexSort(),
    ctx.mkRegexSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg: KExpr<KRegexSort>
    ): KApp<KRegexSort, KRegexSort> = mkRegexCrossNoSimplify(arg)
}

class KRegexDifferenceDecl internal constructor(
    ctx: KContext,
) : KFuncDecl2<KRegexSort, KRegexSort, KRegexSort>(
    ctx,
    "regex_diff",
    ctx.mkRegexSort(),
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
) : KFuncDecl1<KRegexSort, KRegexSort>(
    ctx,
    "regex_comp",
    ctx.mkRegexSort(),
    ctx.mkRegexSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg: KExpr<KRegexSort>
    ): KApp<KRegexSort, KRegexSort> = mkRegexComplementNoSimplify(arg)
}

class KRegexOptionDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KRegexSort, KRegexSort>(
    ctx,
    "regex_opt",
    ctx.mkRegexSort(),
    ctx.mkRegexSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg: KExpr<KRegexSort>
    ): KApp<KRegexSort, KRegexSort> = mkRegexOptionNoSimplify(arg)
}

class KRegexRangeDecl internal constructor(
    ctx: KContext,
) : KFuncDecl2<KRegexSort, KStringSort, KStringSort>(
    ctx,
    "regex_range",
    ctx.mkRegexSort(),
    ctx.mkStringSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KStringSort>
    ): KApp<KRegexSort, *> = mkRegexRangeNoSimplify(arg0, arg1)
}

class KRegexEpsilonDecl internal constructor(
    ctx: KContext
) : KConstDecl<KRegexSort>(
    ctx,
    "regex_eps",
    ctx.mkRegexSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun apply(
        args: List<KExpr<*>>
    ): KApp<KRegexSort, *> = ctx.mkRegexEpsilon()
}

class KRegexAllDecl internal constructor(
    ctx: KContext
) : KConstDecl<KRegexSort>(
    ctx,
    "regex_all",
    ctx.mkRegexSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun apply(
        args: List<KExpr<*>>
    ): KApp<KRegexSort, *> = ctx.mkRegexAll()
}

class KRegexAllCharDecl internal constructor(
    ctx: KContext
) : KConstDecl<KRegexSort>(
    ctx,
    "regex_all_char",
    ctx.mkRegexSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun apply(
        args: List<KExpr<*>>
    ): KApp<KRegexSort, *> = ctx.mkRegexAllChar()
}
