package io.ksmt.decl

import io.ksmt.KContext
import io.ksmt.expr.KApp
import io.ksmt.expr.KExpr
import io.ksmt.sort.KSort
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRegexSort
import io.ksmt.sort.KStringSort

class KStringLiteralDecl internal constructor(
    ctx: KContext,
    val value: String
) : KConstDecl<KStringSort>(ctx, value, ctx.mkStringSort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KStringSort, *> = ctx.mkStringLiteral(value)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KStringConcatDecl internal constructor(
    ctx: KContext,
) : KFuncDecl2<KStringSort, KStringSort, KStringSort>(
    ctx,
    name = "concat",
    resultSort = ctx.mkStringSort(),
    ctx.mkStringSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KStringSort>
    ): KApp<KStringSort, *> = mkStringConcatExprNoSimplify(arg0, arg1)
}

class KStringLenDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KIntSort, KStringSort>(ctx, "len", ctx.mkIntSort(), ctx.mkStringSort()) {
    override fun KContext.apply(arg: KExpr<KStringSort>): KApp<KIntSort, KStringSort> = mkStringLenNoSimplify(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KStringToRegexDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KRegexSort, KStringSort>(ctx, "to_regex", ctx.mkRegexSort(), ctx.mkStringSort()) {
    override fun KContext.apply(arg: KExpr<KStringSort>): KApp<KRegexSort, KStringSort> = mkStringToRegexNoSimplify(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KStringInRegexDecl internal constructor(
    ctx: KContext,
) : KFuncDecl2<KBoolSort, KStringSort, KRegexSort>(
    ctx,
    name = "in_regex",
    resultSort = ctx.mkBoolSort(),
    ctx.mkStringSort(),
    ctx.mkRegexSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KRegexSort>
    ): KApp<KBoolSort, *> = mkStringInRegexNoSimplify(arg0, arg1)
}

class KSuffixOfDecl internal constructor(
    ctx: KContext,
) : KFuncDecl2<KBoolSort, KStringSort, KStringSort>(
    ctx,
    name = "suffix_of",
    resultSort = ctx.mkBoolSort(),
    ctx.mkStringSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KStringSort>
    ): KApp<KBoolSort, *> = mkSuffixOfNoSimplify(arg0, arg1)
}

class KPrefixOfDecl internal constructor(
    ctx: KContext,
) : KFuncDecl2<KBoolSort, KStringSort, KStringSort>(
    ctx,
    name = "prefix_of",
    resultSort = ctx.mkBoolSort(),
    ctx.mkStringSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KStringSort>
    ): KApp<KBoolSort, *> = mkPrefixOfNoSimplify(arg0, arg1)
}
