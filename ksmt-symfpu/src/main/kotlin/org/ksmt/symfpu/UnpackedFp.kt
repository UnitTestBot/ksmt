package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.cache.hash
import org.ksmt.cache.structurallyEqual
import org.ksmt.expr.KExpr
import org.ksmt.expr.printer.ExpressionPrinter
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort
import org.ksmt.symfpu.UnpackedFp.PackedFp.Companion.makeBvInf
import org.ksmt.symfpu.UnpackedFp.PackedFp.Companion.makeBvNaN
import org.ksmt.symfpu.UnpackedFp.PackedFp.Companion.makeBvZero
import org.ksmt.utils.FpUtils
import org.ksmt.utils.FpUtils.fpInfExponentBiased
import org.ksmt.utils.FpUtils.fpInfSignificand
import org.ksmt.utils.FpUtils.fpNaNExponentBiased
import org.ksmt.utils.FpUtils.fpNaNSignificand
import org.ksmt.utils.FpUtils.fpZeroExponentBiased
import org.ksmt.utils.FpUtils.fpZeroSignificand
import org.ksmt.utils.cast


fun KExpr<KBvSort>.extendUnsigned(ext: Int): KExpr<KBvSort> {
    return ctx.mkBvZeroExtensionExpr(ext, this)
}

fun KExpr<KBvSort>.extendSigned(ext: Int): KExpr<KBvSort> {
    return ctx.mkBvSignExtensionExpr(ext, this)
}

fun KExpr<KBvSort>.resizeUnsigned(newSize: UInt): KExpr<KBvSort> {
    val width = sort.sizeBits
    return if (newSize > width) {
        ctx.mkBvZeroExtensionExpr((newSize - width).toInt(), this)
    } else {
        this
    }
}

fun KExpr<KBvSort>.resizeSigned(newSize: UInt): KExpr<KBvSort> {
    val width = sort.sizeBits
    return if (newSize > width) {
        ctx.mkBvSignExtensionExpr((newSize - width).toInt(), this)
    } else {
        this
    }
}

fun KExpr<KBvSort>.contract(reduction: Int): KExpr<KBvSort> {
    val width = sort.sizeBits.toInt()
    check(width > reduction)
    return this.ctx.mkBvExtractExpr((width - 1) - reduction, 0, this)
}

@Suppress("LongParameterList")
class UnpackedFp<Fp : KFpSort> private constructor(
    ctx: KContext, override val sort: Fp,
    val sign: KExpr<KBoolSort>, // negative
    val unbiasedExponent: KExpr<KBvSort>,
    val normalizedSignificand: KExpr<KBvSort>,
    val isNaN: KExpr<KBoolSort> = ctx.mkFalse(),
    val isInf: KExpr<KBoolSort> = ctx.mkFalse(),
    val isZero: KExpr<KBoolSort> = ctx.mkFalse(),
    val packedBv: PackedFp = PackedFp.None,
) : KExpr<Fp>(ctx) {

    constructor(
        ctx: KContext, sort: Fp, sign: KExpr<KBoolSort>, exponent: KExpr<KBvSort>, significand: KExpr<KBvSort>,
        packedBv: PackedFp = PackedFp.None,
    ) : this(
        ctx,
        sort,
        sign,
        exponent.matchWidthSigned(ctx, ctx.defaultExponent(sort)),
        significand,
        ctx.mkFalse(),
        ctx.mkFalse(),
        ctx.mkFalse(),
        packedBv
    )

    sealed class PackedFp {
        data class Exists(
            val sign: KExpr<KBoolSort>, val exponent: KExpr<KBvSort>, val significand: KExpr<KBvSort>) : PackedFp() {
            infix fun eq(packedBv: Exists): KExpr<KBoolSort> = with(sign.ctx) {
                mkAnd(sign eq packedBv.sign, exponent eq packedBv.exponent, significand eq packedBv.significand)
            }

            private val ctx = sign.ctx
            val sort = ctx.mkFpSort(exponent.sort.sizeBits, significand.sort.sizeBits + 1u)
        }

        object None : PackedFp()

        fun toIEEE() = if (this is Exists) {
            with(sign.ctx) {
                val bias = mkBv(sort.exponentShiftSize(), exponent.sort.sizeBits)
                mkBvConcatExpr(boolToBv(sign), mkBvAddExpr(exponent, bias), significand)
            }
        } else null

        fun setSign(sign: KExpr<KBoolSort>) = when (this) {
            is Exists -> copy(sign = sign)
            None -> this
        }

        fun exists() = this is Exists

        companion object {
            fun packedIte(
                cond: KExpr<KBoolSort>,
                trueBranch: PackedFp,
                falseBranch: PackedFp,
            ): PackedFp = with(cond.ctx) {
                return when {
                    trueBranch is Exists && falseBranch is Exists -> Exists(
                        sign = mkIte(cond, trueBranch.sign, falseBranch.sign),
                        exponent = mkIte(cond, trueBranch.exponent, falseBranch.exponent),
                        significand = mkIte(cond, trueBranch.significand, falseBranch.significand)
                    )

                    else -> None
                }
            }

            fun KContext.makeBvNaN(sort: KFpSort): Exists {
                return Exists(
                    sign = falseExpr,
                    exponent = FpUtils.unbiasFpExponent(fpNaNExponentBiased(sort), sort.exponentBits).cast(),
                    significand = fpNaNSignificand(sort).cast()
                )
            }

            fun KContext.makeBvInf(sort: KFpSort, sign: KExpr<KBoolSort>): Exists {
                return Exists(
                    sign = sign,
                    exponent = FpUtils.unbiasFpExponent(fpInfExponentBiased(sort), sort.exponentBits).cast(),
                    significand = fpInfSignificand(sort).cast()
                )
            }

            fun KContext.makeBvZero(sort: KFpSort, sign: KExpr<KBoolSort>): Exists {
                return Exists(
                    sign = sign,
                    exponent = FpUtils.unbiasFpExponent(fpZeroExponentBiased(sort), sort.exponentBits).cast(),
                    significand = fpZeroSignificand(sort).cast()
                )
            }
        }
    }

//    private val expWidth = sort.exponentBits.toInt()
//    private val packedSignificand =
//        packedBv?.let { ctx.mkBvExtractExpr(packedBv.sort.sizeBits.toInt() - expWidth - 2, 0, packedBv) }
//    private val packedExponent = packedBv?.let {
//        ctx.mkBvExtractExpr(
//            packedBv.sort.sizeBits.toInt() - 2,
//            packedBv.sort.sizeBits.toInt() - expWidth - 1,
//            packedBv
//        )
//    }

    fun getSignificand(packed: Boolean = false) = if (packed) {
        check(packedBv is PackedFp.Exists)
        packedBv.significand
    } else {
        normalizedSignificand
    }

    fun exponentWidth() = unbiasedExponent.sort.sizeBits
    fun significandWidth() = normalizedSignificand.sort.sizeBits

    //same for exponent
    fun getExponent(packed: Boolean = false) =
        if (packed) {
            check(packedBv is PackedFp.Exists)
            packedBv.exponent
        } else unbiasedExponent


    fun signBv() = ctx.boolToBv(sign)

    override fun accept(transformer: KTransformerBase): KExpr<Fp> {
        check(transformer is FpToBvTransformer.AdapterTermsRewriter) { "Leaked unpackedFp: $this" }
        return transformer.transform(this).cast()
    }

    override fun print(printer: ExpressionPrinter) = with(printer) {
        append("(unpackedFp ")
        append("sign: ${sign}, exponent: $unbiasedExponent, significand: $normalizedSignificand")
        append(")")
    }

    override fun internEquals(other: Any): Boolean =
        structurallyEqual(other) { listOf(sign, unbiasedExponent, normalizedSignificand, isNaN, isInf, isZero) }

    override fun internHashCode(): Int =
        hash(listOf(sign, unbiasedExponent, normalizedSignificand, isNaN, isInf, isZero))


    val isNegative = sign
    val isNegativeInfinity = with(ctx) { isNegative and isInf }
    val isPositiveInfinity = with(ctx) {
        !isNegative and isInf
    }

    // for tests
    internal fun toFp() = ctx.pack(ctx.packToBv(this), sort)

    // Moves the leading 1 up to the correct position, adjusting the
    // exponent as required.
    fun normaliseUp(): UnpackedFp<Fp> = with(ctx) {
        val normal = normaliseShift(normalizedSignificand)

        val exponentWidth = unbiasedExponent.sort.sizeBits
        check(
            normal.shiftAmount.sort.sizeBits < exponentWidth
        ) // May lose data / be incorrect for very small exponents and very large significands

        val signedAlignAmount = normal.shiftAmount.resizeUnsigned(exponentWidth)
        val correctedExponent = mkBvSubExpr(unbiasedExponent, signedAlignAmount)

        // Optimisation : could move the zero detect version in if used in all cases
        //  catch - it zero detection in unpacking is different.
        return UnpackedFp(ctx, sort, sign, correctedExponent, normal.normalised, packedBv)
    }

    fun normaliseUpDetectZero(): UnpackedFp<Fp> = with(ctx) {
        val normal = normaliseShift(normalizedSignificand)

        // May lose data / be incorrect for very small exponents and very large significands
        check(normal.shiftAmount.sort.sizeBits < exponentWidth())

        val signedAlignAmount = normal.shiftAmount.resizeUnsigned(exponentWidth())
        val correctedExponent = ctx.mkBvSubExpr(unbiasedExponent, signedAlignAmount)

        return iteOp(
            isAllZeros(normalizedSignificand),
            makeZero(sort, sign),
            UnpackedFp(ctx, sort, sign, correctedExponent, normal.normalised, packedBv)
        )
    }

    fun inNormalRange() = ctx.mkBvSignedLessOrEqualExpr(ctx.minNormalExponent(sort), unbiasedExponent)

    fun negate() = with(ctx) { setSign(mkIte(isNaN, sign, !sign)) }

    fun setSign(newSign: KExpr<KBoolSort>) = UnpackedFp(ctx, sort,
        newSign, unbiasedExponent, normalizedSignificand, isNaN, isInf, isZero, packedBv.setSign(newSign))


    fun absolute() = with(ctx) {
        UnpackedFp(ctx, sort,
            falseExpr, unbiasedExponent, normalizedSignificand, isNaN, isInf, isZero, packedBv.setSign(falseExpr))
    }

    fun <T : KFpSort> extend(expWidth: Int, sigExtension: Int, targetFormat: T): UnpackedFp<T> = with(ctx) {

        return UnpackedFp(
            this,
            targetFormat,
            sign,
            unbiasedExponent.extendSigned(expWidth),
            mkBvShiftLeftExpr(
                normalizedSignificand.extendUnsigned(sigExtension),
                mkBv(sigExtension, significandWidth() + sigExtension.toUInt())
            ),
            isNaN,
            isInf,
            isZero,
        )
    }


    companion object {


        fun <Fp : KFpSort> KContext.makeNaN(sort: Fp) = UnpackedFp(
            this, sort, sign = falseExpr, unbiasedExponent = defaultExponent(sort),
            normalizedSignificand = defaultSignificand(sort), isNaN = trueExpr, packedBv = makeBvNaN(sort)
        )

        fun <Fp : KFpSort> KContext.makeInf(
            sort: Fp, sign: KExpr<KBoolSort>
        ) = UnpackedFp(
            this, sort, sign, unbiasedExponent = defaultExponent(sort),
            normalizedSignificand = defaultSignificand(sort), isInf = trueExpr, packedBv = makeBvInf(sort, sign)
        )

        fun <Fp : KFpSort> KContext.makeZero(sort: Fp, sign: KExpr<KBoolSort>) = UnpackedFp(
            this, sort, sign, unbiasedExponent = defaultExponent(sort),
            normalizedSignificand = defaultSignificand(sort), isZero = trueExpr, packedBv = makeBvZero(sort, sign)
        )

        fun <T : KFpSort> KContext.iteOp(
            cond: KExpr<KBoolSort>, l: UnpackedFp<T>, r: UnpackedFp<T>
        ): UnpackedFp<T> {
            return UnpackedFp(
                this,
                l.sort,
                sign = mkIte(cond, l.sign, r.sign),
                unbiasedExponent = mkIte(cond, l.unbiasedExponent, r.unbiasedExponent),
                normalizedSignificand = mkIte(cond, l.normalizedSignificand, r.normalizedSignificand),
                isNaN = mkIte(cond, l.isNaN, r.isNaN),
                isInf = mkIte(cond, l.isInf, r.isInf),
                isZero = mkIte(cond, l.isZero, r.isZero),
                packedBv = PackedFp.packedIte(cond, l.packedBv, r.packedBv)
            )
        }

    }
}
