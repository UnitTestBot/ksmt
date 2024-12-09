package io.ksmt.expr

import io.ksmt.KContext
import io.ksmt.cache.hash
import io.ksmt.cache.structurallyEqual
import io.ksmt.decl.KDecl
import io.ksmt.decl.KRegexKleeneClosureDecl
import io.ksmt.decl.KRegexKleeneCrossDecl
import io.ksmt.decl.KRegexLiteralDecl
import io.ksmt.decl.KRegexComplementDecl
import io.ksmt.decl.KRegexOptionDecl
import io.ksmt.decl.KEpsilonDecl
import io.ksmt.decl.KAllDecl
import io.ksmt.decl.KAllCharDecl
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KRegexSort
import io.ksmt.sort.KStringSort

class KRegexLiteralExpr internal constructor(
    ctx: KContext,
    val value: String
) : KInterpretedValue<KRegexSort>(ctx) {
    override val sort: KRegexSort
        get() = ctx.regexSort

    override val decl: KRegexLiteralDecl
        get() = ctx.mkRegexLiteralDecl(value)

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { value }
}

class KRegexConcatExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KRegexSort>,
    val arg1: KExpr<KRegexSort>
) : KApp<KRegexSort, KRegexSort>(ctx) {
    override val args: List<KExpr<KRegexSort>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KRegexSort>
        get() = ctx.mkRegexConcatDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> = transformer.transform(this)

    override val sort: KRegexSort = ctx.mkRegexSort()

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KRegexUnionExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KRegexSort>,
    val arg1: KExpr<KRegexSort>
) : KApp<KRegexSort, KRegexSort>(ctx) {
    override val args: List<KExpr<KRegexSort>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KRegexSort>
        get() = ctx.mkRegexUnionDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> = transformer.transform(this)

    override val sort: KRegexSort = ctx.mkRegexSort()

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KRegexIntersectionExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KRegexSort>,
    val arg1: KExpr<KRegexSort>
) : KApp<KRegexSort, KRegexSort>(ctx) {
    override val args: List<KExpr<KRegexSort>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KRegexSort>
        get() = ctx.mkRegexIntersectionDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> = transformer.transform(this)

    override val sort: KRegexSort = ctx.mkRegexSort()

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KRegexKleeneClosureExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KRegexSort>
) : KApp<KRegexSort, KRegexSort>(ctx) {
    override val sort: KRegexSort = ctx.regexSort

    override val decl: KRegexKleeneClosureDecl
        get() = ctx.mkRegexKleeneClosureDecl()

    override val args: List<KExpr<KRegexSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KRegexKleeneCrossExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KRegexSort>
) : KApp<KRegexSort, KRegexSort>(ctx) {
    override val sort: KRegexSort = ctx.regexSort

    override val decl: KRegexKleeneCrossDecl
        get() = ctx.mkRegexKleeneCrossDecl()

    override val args: List<KExpr<KRegexSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KRegexDifferenceExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KRegexSort>,
    val arg1: KExpr<KRegexSort>
) : KApp<KRegexSort, KRegexSort>(ctx) {
    override val args: List<KExpr<KRegexSort>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KRegexSort>
        get() = ctx.mkRegexDifferenceDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> = transformer.transform(this)

    override val sort: KRegexSort = ctx.mkRegexSort()

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KRegexComplementExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KRegexSort>
) : KApp<KRegexSort, KRegexSort>(ctx) {
    override val sort: KRegexSort = ctx.regexSort

    override val decl: KRegexComplementDecl
        get() = ctx.mkRegexComplementDecl()

    override val args: List<KExpr<KRegexSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KRegexOptionExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KRegexSort>
) : KApp<KRegexSort, KRegexSort>(ctx) {
    override val sort: KRegexSort = ctx.regexSort

    override val decl: KRegexOptionDecl
        get() = ctx.mkRegexOptionDecl()

    override val args: List<KExpr<KRegexSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}


class KEpsilon(ctx: KContext) : KInterpretedValue<KRegexSort>(ctx) {
    override val sort: KRegexSort = ctx.regexSort

    override val decl: KEpsilonDecl
        get() = ctx.mkEpsilonDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash()
    override fun internEquals(other: Any): Boolean = structurallyEqual(other)
}

class KAll(ctx: KContext) : KInterpretedValue<KRegexSort>(ctx) {
    override val sort: KRegexSort = ctx.regexSort

    override val decl: KAllDecl
        get() = ctx.mkAllDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash()
    override fun internEquals(other: Any): Boolean = structurallyEqual(other)
}

class KAllChar(ctx: KContext) : KInterpretedValue<KRegexSort>(ctx) {
    override val sort: KRegexSort = ctx.regexSort

    override val decl: KAllCharDecl
        get() = ctx.mkAllCharDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash()
    override fun internEquals(other: Any): Boolean = structurallyEqual(other)
}

class KRangeExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KStringSort>
) : KApp<KRegexSort, KStringSort>(ctx) {
    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KRegexSort>
        get() = ctx.mkRangeDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> = transformer.transform(this)

    override val sort: KRegexSort = ctx.mkRegexSort()

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}
