package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFp128Sort
import org.ksmt.sort.KFp16Sort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpSort
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
        require(exponentSize.toInt() <= 63) { "Maximum number of exponent bits is 63" }
    }

    override fun decl(): KDecl<KFpSort> =
        ctx.mkFpCustomSizeDecl(significandSize, exponentSize, significandValue, exponentValue, signBit)

    override fun sort(): KFpSort = ctx.mkFpSort(exponentSize, significandSize)

    override fun accept(transformer: KTransformer): KExpr<KFpSort> = transformer.transform(this)
}


