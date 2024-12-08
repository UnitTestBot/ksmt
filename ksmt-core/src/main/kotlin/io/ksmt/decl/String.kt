package io.ksmt.decl

import io.ksmt.KContext
import io.ksmt.expr.KApp
import io.ksmt.expr.KExpr
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
    name = "str_concat",
    resultSort = ctx.mkStringSort(),
    ctx.mkStringSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KStringSort>
    ): KApp<KStringSort, *> = mkStringConcatNoSimplify(arg0, arg1)
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

class KStringLtDecl internal constructor(ctx: KContext) :
    KFuncDecl2<KBoolSort, KStringSort, KStringSort>(ctx, "stringLt", ctx.mkBoolSort(), ctx.mkStringSort(), ctx.mkStringSort()) {
    override fun KContext.apply(arg0: KExpr<KStringSort>, arg1: KExpr<KStringSort>): KApp<KBoolSort, *> = mkStringLtNoSimplify(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KStringLeDecl internal constructor(ctx: KContext) :
    KFuncDecl2<KBoolSort, KStringSort, KStringSort>(ctx, "stringLe", ctx.mkBoolSort(), ctx.mkStringSort(), ctx.mkStringSort()) {
    override fun KContext.apply(arg0: KExpr<KStringSort>, arg1: KExpr<KStringSort>): KApp<KBoolSort, *> = mkStringLeNoSimplify(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KStringGtDecl internal constructor(ctx: KContext) :
    KFuncDecl2<KBoolSort, KStringSort, KStringSort>(ctx, "stringGt", ctx.mkBoolSort(), ctx.mkStringSort(), ctx.mkStringSort()) {
    override fun KContext.apply(arg0: KExpr<KStringSort>, arg1: KExpr<KStringSort>): KApp<KBoolSort, *> = mkStringGtNoSimplify(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KStringGeDecl internal constructor(ctx: KContext) :
    KFuncDecl2<KBoolSort, KStringSort, KStringSort>(ctx, "stringGe", ctx.mkBoolSort(), ctx.mkStringSort(), ctx.mkStringSort()) {
    override fun KContext.apply(arg0: KExpr<KStringSort>, arg1: KExpr<KStringSort>): KApp<KBoolSort, *> = mkStringGeNoSimplify(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KStringContainsDecl internal constructor(ctx: KContext) :
    KFuncDecl2<KBoolSort, KStringSort, KStringSort>(ctx, "contains", ctx.mkBoolSort(), ctx.mkStringSort(), ctx.mkStringSort()) {
    override fun KContext.apply(arg0: KExpr<KStringSort>, arg1: KExpr<KStringSort>): KApp<KBoolSort, *> = mkStringContainsNoSimplify(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KSingletonSubstringDecl : RuntimeException("Not yet implemented")

class KSubstringDecl : RuntimeException("Not yet implemented")

class KIndexOfDecl : RuntimeException("Not yet implemented")

class KStringReplaceDecl internal constructor(
    ctx: KContext,
) : KFuncDecl3<KStringSort, KStringSort, KStringSort, KStringSort>(
    ctx,
    name = "str_replace",
    resultSort = ctx.mkStringSort(),
    ctx.mkStringSort(),
    ctx.mkStringSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KStringSort>,
        arg2: KExpr<KStringSort>
    ): KApp<KStringSort, *> = mkStringReplaceNoSimplify(arg0, arg1, arg2)
}

class KStringReplaceAllDecl internal constructor(
    ctx: KContext,
) : KFuncDecl3<KStringSort, KStringSort, KStringSort, KStringSort>(
    ctx,
    name = "str_replace_all",
    resultSort = ctx.mkStringSort(),
    ctx.mkStringSort(),
    ctx.mkStringSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KStringSort>,
        arg2: KExpr<KStringSort>
    ): KApp<KStringSort, *> = mkStringReplaceAllNoSimplify(arg0, arg1, arg2)
}

/*
    Maps to and from integers.
 */

class KStringIsDigitDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KBoolSort, KStringSort>(ctx, "is_digit", ctx.mkBoolSort(), ctx.mkStringSort()) {
    override fun KContext.apply(arg: KExpr<KStringSort>): KApp<KBoolSort, KStringSort> = mkStringIsDigitNoSimplify(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KStringToCodeDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KIntSort, KStringSort>(ctx, "to_code", ctx.mkIntSort(), ctx.mkStringSort()) {
    override fun KContext.apply(arg: KExpr<KStringSort>): KApp<KIntSort, KStringSort> = mkStringToCodeNoSimplify(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KStringFromCodeDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KStringSort, KIntSort>(ctx, "from_code", ctx.mkStringSort(), ctx.mkIntSort()) {
    override fun KContext.apply(arg: KExpr<KIntSort>): KApp<KStringSort, KIntSort> = mkStringFromCodeNoSimplify(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KStringToIntDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KIntSort, KStringSort>(ctx, "to_int", ctx.mkIntSort(), ctx.mkStringSort()) {
    override fun KContext.apply(arg: KExpr<KStringSort>): KApp<KIntSort, KStringSort> = mkStringToIntNoSimplify(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KStringFromIntDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KStringSort, KIntSort>(ctx, "from_int", ctx.mkStringSort(), ctx.mkIntSort()) {
    override fun KContext.apply(arg: KExpr<KIntSort>): KApp<KStringSort, KIntSort> = mkStringFromIntNoSimplify(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
