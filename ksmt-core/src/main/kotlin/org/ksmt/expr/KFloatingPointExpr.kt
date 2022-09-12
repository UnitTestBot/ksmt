package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.transformer.KTransformer
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFp128Sort
import org.ksmt.sort.KFp16Sort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KRealSort
import org.ksmt.utils.booleanSignBit
import org.ksmt.utils.getExponent
import org.ksmt.utils.getHalfPrecisionExponent
import org.ksmt.utils.halfPrecisionSignificand
import org.ksmt.utils.significand

abstract class KFpValue<T : KFpSort>(
    ctx: KContext,
    val significand: KBitVecValue<out KBvSort>,
    val exponent: KBitVecValue<out KBvSort>,
    val signBit: Boolean
) : KApp<T, KExpr<*>>(ctx) {
    override val args: List<KExpr<*>> = emptyList()

    companion object {
        const val MAX_EXPONENT_SIZE = 63
    }
}

/**
 * Fp16 value. Note that [value] should has biased Fp32 exponent,
 * but a constructed Fp16 will have an unbiased one.
 *
 * Fp32 to Fp16 transformation:
 * sign   exponent         significand
 * 0      00000000   00000000000000000000000    (1 8 23)
 * x      x___xxxx   xxxxxxxxxx_____________    (1 5 10)
 */
class KFp16Value internal constructor(ctx: KContext, val value: Float) :
    KFpValue<KFp16Sort>(
        ctx,
        significand = with(ctx) { value.halfPrecisionSignificand.toBv(KFp16Sort.significandBits - 1u) },
        exponent = with(ctx) { value.getHalfPrecisionExponent(isBiased = false).toBv(KFp16Sort.exponentBits) },
        signBit = value.booleanSignBit
    ) {

    init {
        // TODO add checks for the bounds
    }

    override fun decl(): KDecl<KFp16Sort> = ctx.mkFp16Decl(value)

    override fun sort(): KFp16Sort = ctx.mkFp16Sort()

    override fun accept(transformer: KTransformer): KExpr<KFp16Sort> = transformer.transform(this)
}

class KFp32Value internal constructor(ctx: KContext, val value: Float) :
    KFpValue<KFp32Sort>(
        ctx,
        significand = with(ctx) { value.significand.toBv(KFp32Sort.significandBits - 1u) },
        exponent = with(ctx) { value.getExponent(isBiased = false).toBv(KFp32Sort.exponentBits) },
        signBit = value.booleanSignBit
    ) {
    override fun decl(): KDecl<KFp32Sort> = ctx.mkFp32Decl(value)

    override fun sort(): KFp32Sort = ctx.mkFp32Sort()

    override fun accept(transformer: KTransformer): KExpr<KFp32Sort> = transformer.transform(this)
}

class KFp64Value internal constructor(ctx: KContext, val value: Double) :
    KFpValue<KFp64Sort>(
        ctx,
        significand = with(ctx) { value.significand.toBv(KFp64Sort.significandBits - 1u) },
        exponent = with(ctx) { value.getExponent(isBiased = false).toBv(KFp64Sort.exponentBits) },
        signBit = value.booleanSignBit
    ) {
    override fun decl(): KDecl<KFp64Sort> = ctx.mkFp64Decl(value)

    override fun sort(): KFp64Sort = ctx.mkFp64Sort()

    override fun accept(transformer: KTransformer): KExpr<KFp64Sort> = transformer.transform(this)
}

/**
 * KFp128 value.
 *
 * Note: if [exponentValue] contains more than [KFp128Sort.exponentBits] meaningful bits,
 * only the last [KFp128Sort.exponentBits] of then will be taken.
 */
class KFp128Value internal constructor(
    ctx: KContext,
    val significandValue: Long,
    val exponentValue: Long,
    signBit: Boolean
) : KFpValue<KFp128Sort>(
    ctx,
    significand = with(ctx) { significandValue.toBv(KFp128Sort.significandBits - 1u) },
    exponent = with(ctx) { exponentValue.toBv(KFp128Sort.exponentBits) },
    signBit
) {
    override fun decl(): KDecl<KFp128Sort> = ctx.mkFp128Decl(significandValue, exponentValue, signBit)

    override fun sort(): KFp128Sort = ctx.mkFp128Sort()

    override fun accept(transformer: KTransformer): KExpr<KFp128Sort> = transformer.transform(this)
}

/**
 * KFp value of custom size.
 *
 * Note: if [exponentValue] contains more than [KFp128Sort.exponentBits] meaningful bits,
 * only the last [KFp128Sort.exponentBits] of then will be taken.
 * The same is true for the significand.
 */
class KFpCustomSizeValue internal constructor(
    ctx: KContext,
    val significandSize: UInt,
    val exponentSize: UInt,
    val significandValue: Long,
    val exponentValue: Long,
    signBit: Boolean
) : KFpValue<KFpSort>(
    ctx,
    significand = with(ctx) { significandValue.toBv(significandSize - 1u) },
    exponent = with(ctx) { exponentValue.toBv(exponentSize) },
    signBit
) {
    init {
        require(exponentSize.toInt() <= MAX_EXPONENT_SIZE) {
            "Maximum number of exponent bits is $MAX_EXPONENT_SIZE"
        }
    }

    override fun decl(): KDecl<KFpSort> =
        ctx.mkFpCustomSizeDecl(significandSize, exponentSize, significandValue, exponentValue, signBit)

    override fun sort(): KFpSort = ctx.mkFpSort(exponentSize, significandSize)

    override fun accept(transformer: KTransformer): KExpr<KFpSort> = transformer.transform(this)
}

class KFpAbsExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override fun decl(): KDecl<S> = ctx.mkFpAbsDecl(value.sort())

    override fun sort(): S = value.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Inverts the sign bit.
 */
class KFpNegationExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override fun decl(): KDecl<S> = ctx.mkFpNegationDecl(value.sort())

    override fun sort(): S = value.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

// TODO Can they have different sorts?
class KFpAddExpr<out R: KFpRoundingModeSort, S : KFpSort> internal constructor(
    ctx: KContext,
    val roundingMode: KExpr<out R>,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<*>>(ctx) {
    override val args: List<KExpr<*>>
        get() = listOf(roundingMode, arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkFpAddDecl(roundingMode.sort(), arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

class KFpSubExpr<out R: KFpRoundingModeSort, S : KFpSort> internal constructor(
    ctx: KContext,
    val roundingMode: KExpr<out R>,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<*>>(ctx) {
    override val args: List<KExpr<*>>
        get() = listOf(roundingMode, arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkFpSubDecl(roundingMode.sort(), arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

class KFpMulExpr<out R: KFpRoundingModeSort, S : KFpSort> internal constructor(
    ctx: KContext,
    val roundingMode: KExpr<out R>,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<*>>(ctx) {
    override val args: List<KExpr<*>>
        get() = listOf(roundingMode, arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkFpMulDecl(roundingMode.sort(), arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

class KFpDivExpr<out R: KFpRoundingModeSort, S : KFpSort> internal constructor(
    ctx: KContext,
    val roundingMode: KExpr<out R>,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<*>>(ctx) {
    override val args: List<KExpr<*>>
        get() = listOf(roundingMode, arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkFpDivDecl(roundingMode.sort(), arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

class KFpFusedMulAddExpr<R : KFpRoundingModeSort, S : KFpSort> internal constructor(
    ctx: KContext,
    val roundingMode: KExpr<out R>,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>,
    val arg2: KExpr<S>
) : KApp<S, KExpr<*>>(ctx) {
    override val args: List<KExpr<*>>
        get() = listOf(roundingMode, arg0, arg1, arg2)

    override fun decl(): KDecl<S> = ctx.mkFpFusedMulAddDecl(
        roundingMode.sort(),
        arg0.sort(),
        arg1.sort(),
        arg2.sort()
    )

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)

}

class KFpSqrtExpr<R : KFpRoundingModeSort, S : KFpSort> internal constructor(
    ctx: KContext,
    val roundingMode: KExpr<out R>,
    val value: KExpr<S>
) : KApp<S, KExpr<*>>(ctx) {
    override val args: List<KExpr<*>>
        get() = listOf(roundingMode, value)

    override fun decl(): KDecl<S> = ctx.mkFpSqrtDecl(roundingMode.sort(), value.sort())

    override fun sort(): S = value.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)

}

class KFpRemExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkFpRemDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

class KFpRoundToIntegralExpr<R : KFpRoundingModeSort, S : KFpSort> internal constructor(
    ctx: KContext,
    val roundingMode: KExpr<out R>,
    val value: KExpr<S>
) : KApp<S, KExpr<*>>(ctx) {
    override val args: List<KExpr<*>>
        get() = listOf(roundingMode, value)

    override fun decl(): KDecl<S> = ctx.mkFpRoundToIntegralDecl(roundingMode.sort(), value.sort())

    override fun sort(): S = value.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)

}

class KFpMinExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkFpMinDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

class KFpMaxExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkFpMaxDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

class KFpLessOrEqualExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkFpLessOrEqualDecl(arg0.sort(), arg1.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KFpLessExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkFpLessDecl(arg0.sort(), arg1.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KFpGreaterOrEqualExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkFpGreaterOrEqualDecl(arg0.sort(), arg1.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KFpGreaterExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkFpGreaterDecl(arg0.sort(), arg1.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KFpEqualExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkFpEqualDecl(arg0.sort(), arg1.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KFpIsNormalExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override fun decl(): KDecl<KBoolSort> = ctx.mkFpIsNormalDecl(value.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KFpIsSubnormalExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override fun decl(): KDecl<KBoolSort> = ctx.mkFpIsSubnormalDecl(value.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KFpIsZeroExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override fun decl(): KDecl<KBoolSort> = ctx.mkFpIsZeroDecl(value.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KFpIsInfiniteExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override fun decl(): KDecl<KBoolSort> = ctx.mkFpIsInfiniteDecl(value.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KFpIsNaNExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override fun decl(): KDecl<KBoolSort> = ctx.mkFpIsNaNDecl(value.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KFpIsNegativeExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override fun decl(): KDecl<KBoolSort> = ctx.mkFpIsNegativeDecl(value.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KFpIsPositiveExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override fun decl(): KDecl<KBoolSort> = ctx.mkFpIsPositiveDecl(value.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

// TODO mkFpToFp ???

class KFpToBvExpr<R : KFpRoundingModeSort, S : KFpSort> internal constructor(
    ctx: KContext,
    val roundingMode: KExpr<out R>,
    val value: KExpr<S>,
    val bvSize: Int,
    val isSigned: Boolean
) : KApp<KBvSort, KExpr<*>>(ctx) {
    override val args: List<KExpr<*>>
        get() = listOf(roundingMode, value)

    override fun decl(): KDecl<KBvSort> = ctx.mkFpToBvDecl(roundingMode.sort(), value.sort(), bvSize, isSigned)

    override fun sort(): KBvSort = decl().sort

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

class KFpToRealExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KRealSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override fun decl(): KDecl<KRealSort> = ctx.mkFpToRealDecl(value.sort())

    override fun sort(): KRealSort = ctx.mkRealSort()

    override fun accept(transformer: KTransformer): KExpr<KRealSort> = transformer.transform(this)
}

class KFpToIEEEBvExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBvSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override fun decl(): KDecl<KBvSort> = ctx.mkFpToIEEEBvDecl(value.sort())

    override fun sort(): KBvSort = decl().sort

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}
