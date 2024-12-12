package io.ksmt.decl

import io.ksmt.KContext
import io.ksmt.expr.KApp
import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRegexSort
import io.ksmt.sort.KStringSort

class KStringConcatDecl internal constructor(
    ctx: KContext,
) : KFuncDecl2<KStringSort, KStringSort, KStringSort>(
    ctx,
    "str_concat",
    ctx.mkStringSort(),
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
) : KFuncDecl1<KIntSort, KStringSort>(
    ctx,
    "str_len",
    ctx.mkIntSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg: KExpr<KStringSort>
    ): KApp<KIntSort, KStringSort> = mkStringLenNoSimplify(arg)
}

class KStringToRegexDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KRegexSort, KStringSort>(
    ctx,
    "str_to_regex",
    ctx.mkRegexSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg: KExpr<KStringSort>
    ): KApp<KRegexSort, KStringSort> = mkStringToRegexNoSimplify(arg)
}

class KStringInRegexDecl internal constructor(
    ctx: KContext,
) : KFuncDecl2<KBoolSort, KStringSort, KRegexSort>(
    ctx,
    "str_in_regex",
    ctx.mkBoolSort(),
    ctx.mkStringSort(),
    ctx.mkRegexSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KRegexSort>
    ): KApp<KBoolSort, *> = mkStringInRegexNoSimplify(arg0, arg1)
}

class KStringSuffixOfDecl internal constructor(
    ctx: KContext,
) : KFuncDecl2<KBoolSort, KStringSort, KStringSort>(
    ctx,
    "str_suffix_of",
    ctx.mkBoolSort(),
    ctx.mkStringSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KStringSort>
    ): KApp<KBoolSort, *> = mkStringSuffixOfNoSimplify(arg0, arg1)
}

class KStringPrefixOfDecl internal constructor(
    ctx: KContext,
) : KFuncDecl2<KBoolSort, KStringSort, KStringSort>(
    ctx,
    "str_prefix_of",
    ctx.mkBoolSort(),
    ctx.mkStringSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KStringSort>
    ): KApp<KBoolSort, *> = mkStringPrefixOfNoSimplify(arg0, arg1)
}

class KStringLtDecl internal constructor(
    ctx: KContext
) : KFuncDecl2<KBoolSort, KStringSort, KStringSort>(
    ctx,
    "str_lt",
    ctx.mkBoolSort(),
    ctx.mkStringSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KStringSort>
    ): KApp<KBoolSort, *> = mkStringLtNoSimplify(arg0, arg1)
}

class KStringLeDecl internal constructor(
    ctx: KContext
) : KFuncDecl2<KBoolSort, KStringSort, KStringSort>(
    ctx,
    "str_le",
    ctx.mkBoolSort(),
    ctx.mkStringSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KStringSort>
    ): KApp<KBoolSort, *> = mkStringLeNoSimplify(arg0, arg1)
}

class KStringGtDecl internal constructor(
    ctx: KContext
) : KFuncDecl2<KBoolSort, KStringSort, KStringSort>(
    ctx,
    "str_gt",
    ctx.mkBoolSort(),
    ctx.mkStringSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KStringSort>
    ): KApp<KBoolSort, *> = mkStringGtNoSimplify(arg0, arg1)
}

class KStringGeDecl internal constructor(
    ctx: KContext
) : KFuncDecl2<KBoolSort, KStringSort, KStringSort>(
    ctx,
    "str_ge",
    ctx.mkBoolSort(),
    ctx.mkStringSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KStringSort>
    ): KApp<KBoolSort, *> = mkStringGeNoSimplify(arg0, arg1)
}

class KStringContainsDecl internal constructor(
    ctx: KContext
) : KFuncDecl2<KBoolSort, KStringSort, KStringSort>(
    ctx,
    "str_contains",
    ctx.mkBoolSort(),
    ctx.mkStringSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KStringSort>
    ): KApp<KBoolSort, *> = mkStringContainsNoSimplify(arg0, arg1)
}

class KStringSingletonSubDecl internal constructor(
    ctx: KContext
) : KFuncDecl2<KStringSort, KStringSort, KIntSort>(
    ctx,
    "str_singleton_sub",
    ctx.mkStringSort(),
    ctx.mkStringSort(),
    ctx.mkIntSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KIntSort>
    ): KApp<KStringSort, *> = mkStringSingletonSubNoSimplify(arg0, arg1)
}

class KStringSubDecl internal constructor(
    ctx: KContext,
) : KFuncDecl3<KStringSort, KStringSort, KIntSort, KIntSort>(
    ctx,
    "str_sub",
    ctx.mkStringSort(),
    ctx.mkStringSort(),
    ctx.mkIntSort(),
    ctx.mkIntSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KIntSort>,
        arg2: KExpr<KIntSort>
    ): KApp<KStringSort, *> = mkStringSubNoSimplify(arg0, arg1, arg2)
}

class KStringIndexOfDecl internal constructor(
    ctx: KContext,
) : KFuncDecl3<KIntSort, KStringSort, KStringSort, KIntSort>(
    ctx,
    "str_index_of",
    ctx.mkIntSort(),
    ctx.mkStringSort(),
    ctx.mkStringSort(),
    ctx.mkIntSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KStringSort>,
        arg2: KExpr<KIntSort>
    ): KApp<KIntSort, *> = mkStringIndexOfNoSimplify(arg0, arg1, arg2)
}

class KStringReplaceDecl internal constructor(
    ctx: KContext,
) : KFuncDecl3<KStringSort, KStringSort, KStringSort, KStringSort>(
    ctx,
    "str_replace",
    ctx.mkStringSort(),
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
    "str_replace_all",
    ctx.mkStringSort(),
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

class KStringReplaceWithRegexDecl internal constructor(
    ctx: KContext,
) : KFuncDecl3<KStringSort, KStringSort, KRegexSort, KStringSort>(
    ctx,
    "str_replace_with_regex",
    ctx.mkStringSort(),
    ctx.mkStringSort(),
    ctx.mkRegexSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KRegexSort>,
        arg2: KExpr<KStringSort>
    ): KApp<KStringSort, *> = mkStringReplaceWithRegexNoSimplify(arg0, arg1, arg2)
}

class KStringReplaceAllWithRegexDecl internal constructor(
    ctx: KContext,
) : KFuncDecl3<KStringSort, KStringSort, KRegexSort, KStringSort>(
    ctx,
    "str_replace_all_with_regex",
    ctx.mkStringSort(),
    ctx.mkStringSort(),
    ctx.mkRegexSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KStringSort>,
        arg1: KExpr<KRegexSort>,
        arg2: KExpr<KStringSort>
    ): KApp<KStringSort, *> = mkStringReplaceAllWithRegexNoSimplify(arg0, arg1, arg2)
}

/*
    Maps to and from integers.
 */

class KStringIsDigitDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KBoolSort, KStringSort>(
    ctx,
    "str_is_digit",
    ctx.mkBoolSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KStringSort>): KApp<KBoolSort, KStringSort> = mkStringIsDigitNoSimplify(arg)
}

class KStringToCodeDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KIntSort, KStringSort>(
    ctx,
    "str_to_code",
    ctx.mkIntSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KStringSort>): KApp<KIntSort, KStringSort> = mkStringToCodeNoSimplify(arg)
}

class KStringFromCodeDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KStringSort, KIntSort>(
    ctx,
    "str_from_code",
    ctx.mkStringSort(),
    ctx.mkIntSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KIntSort>): KApp<KStringSort, KIntSort> = mkStringFromCodeNoSimplify(arg)
}

class KStringToIntDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KIntSort, KStringSort>(
    ctx,
    "str_to_int",
    ctx.mkIntSort(),
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KStringSort>): KApp<KIntSort, KStringSort> = mkStringToIntNoSimplify(arg)
}

class KStringFromIntDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KStringSort, KIntSort>(
    ctx,
    "str_from_int",
    ctx.mkStringSort(),
    ctx.mkIntSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KIntSort>): KApp<KStringSort, KIntSort> = mkStringFromIntNoSimplify(arg)
}

class KStringLiteralDecl internal constructor(
    ctx: KContext,
    val value: String
) : KConstDecl<KStringSort>(
    ctx,
    value,
    ctx.mkStringSort()
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun apply(args: List<KExpr<*>>): KApp<KStringSort, *> = ctx.mkStringLiteral(value)
}
