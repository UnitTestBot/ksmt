package io.ksmt.expr

import io.ksmt.KContext
import io.ksmt.cache.hash
import io.ksmt.cache.structurallyEqual
import io.ksmt.decl.KRegexConcatDecl
import io.ksmt.decl.KRegexUnionDecl
import io.ksmt.decl.KRegexIntersectionDecl
import io.ksmt.decl.KRegexStarDecl
import io.ksmt.decl.KRegexCrossDecl
import io.ksmt.decl.KRegexDifferenceDecl
import io.ksmt.decl.KRegexComplementDecl
import io.ksmt.decl.KRegexOptionDecl
import io.ksmt.decl.KRegexRangeDecl
import io.ksmt.decl.KRegexPowerDecl
import io.ksmt.decl.KRegexLoopDecl
import io.ksmt.decl.KRegexEpsilonDecl
import io.ksmt.decl.KRegexAllDecl
import io.ksmt.decl.KRegexAllCharDecl
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KRegexSort
import io.ksmt.sort.KStringSort

class KRegexConcatExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KRegexSort>,
    val arg1: KExpr<KRegexSort>
) : KApp<KRegexSort, KRegexSort>(ctx) {
    override val sort: KRegexSort
        get() = ctx.mkRegexSort()

    override val decl: KRegexConcatDecl
        get() = ctx.mkRegexConcatDecl()

    override val args: List<KExpr<KRegexSort>>
        get() = listOf(arg0, arg1)

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KRegexUnionExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KRegexSort>,
    val arg1: KExpr<KRegexSort>
) : KApp<KRegexSort, KRegexSort>(ctx) {
    override val sort: KRegexSort
        get() = ctx.mkRegexSort()

    override val decl: KRegexUnionDecl
        get() = ctx.mkRegexUnionDecl()

    override val args: List<KExpr<KRegexSort>>
        get() = listOf(arg0, arg1)

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KRegexIntersectionExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KRegexSort>,
    val arg1: KExpr<KRegexSort>
) : KApp<KRegexSort, KRegexSort>(ctx) {
    override val sort: KRegexSort
        get() = ctx.mkRegexSort()

    override val decl: KRegexIntersectionDecl
        get() = ctx.mkRegexIntersectionDecl()

    override val args: List<KExpr<KRegexSort>>
        get() = listOf(arg0, arg1)

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KRegexStarExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KRegexSort>
) : KApp<KRegexSort, KRegexSort>(ctx) {
    override val sort: KRegexSort
        get() = ctx.regexSort

    override val decl: KRegexStarDecl
        get() = ctx.mkRegexStarDecl()

    override val args: List<KExpr<KRegexSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KRegexCrossExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KRegexSort>
) : KApp<KRegexSort, KRegexSort>(ctx) {
    override val sort: KRegexSort
        get() = ctx.regexSort

    override val decl: KRegexCrossDecl
        get() = ctx.mkRegexCrossDecl()

    override val args: List<KExpr<KRegexSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KRegexDifferenceExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KRegexSort>,
    val arg1: KExpr<KRegexSort>
) : KApp<KRegexSort, KRegexSort>(ctx) {
    override val sort: KRegexSort
        get() = ctx.mkRegexSort()

    override val args: List<KExpr<KRegexSort>>
        get() = listOf(arg0, arg1)

    override val decl: KRegexDifferenceDecl
        get() = ctx.mkRegexDifferenceDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KRegexComplementExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KRegexSort>
) : KApp<KRegexSort, KRegexSort>(ctx) {
    override val sort: KRegexSort
        get() = ctx.regexSort

    override val decl: KRegexComplementDecl
        get() = ctx.mkRegexComplementDecl()

    override val args: List<KExpr<KRegexSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KRegexOptionExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KRegexSort>
) : KApp<KRegexSort, KRegexSort>(ctx) {
    override val sort: KRegexSort
        get() = ctx.regexSort

    override val decl: KRegexOptionDecl
        get() = ctx.mkRegexOptionDecl()

    override val args: List<KExpr<KRegexSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KRegexRangeExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KStringSort>
) : KApp<KRegexSort, KStringSort>(ctx) {
    override val sort: KRegexSort
        get() = ctx.mkRegexSort()

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg0, arg1)

    override val decl: KRegexRangeDecl
        get() = ctx.mkRegexRangeDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KRegexPowerExpr internal constructor(
    ctx: KContext,
    val power: Int,
    val arg: KExpr<KRegexSort>,
) : KApp<KRegexSort, KRegexSort>(ctx) {
    override val sort: KRegexSort
        get() = ctx.regexSort

    override val decl: KRegexPowerDecl
        get() = ctx.mkRegexPowerDecl(power)

    override val args: List<KExpr<KRegexSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg, power)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg }, { power })
}

class KRegexLoopExpr internal constructor(
    ctx: KContext,
    val from: Int,
    val to: Int,
    val arg: KExpr<KRegexSort>,
) : KApp<KRegexSort, KRegexSort>(ctx) {
    override val sort: KRegexSort
        get() = ctx.regexSort

    override val decl: KRegexLoopDecl
        get() = ctx.mkRegexLoopDecl(from, to)

    override val args: List<KExpr<KRegexSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg, from, to)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg }, { from }, { to })
}

class KRegexEpsilon(ctx: KContext) : KInterpretedValue<KRegexSort>(ctx) {
    override val sort: KRegexSort
        get() = ctx.regexSort

    override val decl: KRegexEpsilonDecl
        get() = ctx.mkRegexEpsilonDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash()
    override fun internEquals(other: Any): Boolean = structurallyEqual(other)
}

class KRegexAll(ctx: KContext) : KInterpretedValue<KRegexSort>(ctx) {
    override val sort: KRegexSort
        get() = ctx.regexSort

    override val decl: KRegexAllDecl
        get() = ctx.mkRegexAllDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash()
    override fun internEquals(other: Any): Boolean = structurallyEqual(other)
}

class KRegexAllChar(ctx: KContext) : KInterpretedValue<KRegexSort>(ctx) {
    override val sort: KRegexSort
        get() = ctx.regexSort

    override val decl: KRegexAllCharDecl
        get() = ctx.mkRegexAllCharDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash()
    override fun internEquals(other: Any): Boolean = structurallyEqual(other)
}
