package io.ksmt.symfpu.operations

import io.ksmt.KContext
import io.ksmt.cache.hash
import io.ksmt.cache.structurallyEqual
import io.ksmt.expr.KExpr
import io.ksmt.expr.printer.ExpressionPrinter
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpSort
import io.ksmt.symfpu.operations.OptionalPackedFp.Companion.makeBvInf
import io.ksmt.symfpu.operations.OptionalPackedFp.Companion.makeBvNaN
import io.ksmt.symfpu.operations.OptionalPackedFp.Companion.makeBvZero
import io.ksmt.symfpu.solver.FpToBvTransformer
import io.ksmt.utils.uncheckedCast


fun KExpr<KBvSort>.extendUnsigned(ext: Int): KExpr<KBvSort> = ctx.mkBvZeroExtensionExpr(ext, this)

fun KExpr<KBvSort>.extendSigned(ext: Int): KExpr<KBvSort> = ctx.mkBvSignExtensionExpr(ext, this)

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
    check(width > reduction) { "Cannot contract to a width smaller than 1" }
    return this.ctx.mkBvExtractExpr((width - 1) - reduction, 0, this)
}

@Suppress("LongParameterList")
class UnpackedFp<Fp : KFpSort> private constructor(
    ctx: KContext, override val sort: Fp,
    val sign: KExpr<KBoolSort>,
    val unbiasedExponent: KExpr<KBvSort>,
    val normalizedSignificand: KExpr<KBvSort>,
    val isNaN: KExpr<KBoolSort> = ctx.mkFalse(),
    val isInf: KExpr<KBoolSort> = ctx.mkFalse(),
    val isZero: KExpr<KBoolSort> = ctx.mkFalse(),
    val packedFp: OptionalPackedFp = NoPackedFp,
) : KExpr<Fp>(ctx) {
    constructor(
        ctx: KContext, sort: Fp, sign: KExpr<KBoolSort>, exponent: KExpr<KBvSort>, significand: KExpr<KBvSort>,
        packedBv: OptionalPackedFp = NoPackedFp,
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

    fun getSignificand(packed: Boolean = false) = if (packed) {
        check(packedFp is PackedFp) { "Cannot get significand from unpackedFp: $this" }
        packedFp.significand
    } else {
        normalizedSignificand
    }

    fun exponentWidth() = unbiasedExponent.sort.sizeBits
    fun significandWidth() = normalizedSignificand.sort.sizeBits

    //same for exponent
    fun getExponent(packed: Boolean = false) =
        if (packed) {
            check(packedFp is PackedFp) { "Cannot get packed exponent from unpackedFp: $this" }
            packedFp.exponent
        } else {
            unbiasedExponent
        }

    fun signBv() = ctx.boolToBv(sign)

    override fun accept(transformer: KTransformerBase): KExpr<Fp> {
        check(transformer is FpToBvTransformer.UnpackedFpRewriter) { "Leaked unpackedFp: $this" }
        return transformer.transform(this).uncheckedCast()
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

    // Moves the leading 1 up to the correct position, adjusting the
    // exponent as required.
    fun normaliseUp(): UnpackedFp<Fp> = with(ctx) {
        val normal = normaliseShift(normalizedSignificand)

        val exponentWidth = unbiasedExponent.sort.sizeBits
        check(normal.shiftAmount.sort.sizeBits < exponentWidth) { "Exponent shift is too large" }

        val signedAlignAmount = normal.shiftAmount.resizeUnsigned(exponentWidth)
        val correctedExponent = mkBvSubExpr(unbiasedExponent, signedAlignAmount)

        // Optimisation : could move the zero detect version in if used in all cases
        //  catch - it zero detection in unpacking is different.
        return UnpackedFp(ctx, sort, sign, correctedExponent, normal.normalised, packedFp)
    }

    fun normaliseUpDetectZero(): UnpackedFp<Fp> = with(ctx) {
        val normal = normaliseShift(normalizedSignificand)

        // May lose data / be incorrect for very small exponents and very large significands
        check(normal.shiftAmount.sort.sizeBits < exponentWidth()) { "Exponent shift is too large" }

        val signedAlignAmount = normal.shiftAmount.resizeUnsigned(exponentWidth())
        val correctedExponent = ctx.mkBvSubExpr(unbiasedExponent, signedAlignAmount)

        return iteOp(
            isAllZeros(normalizedSignificand),
            makeZero(sort, sign),
            UnpackedFp(ctx, sort, sign, correctedExponent, normal.normalised, packedFp)
        )
    }

    fun inNormalRange() = ctx.mkBvSignedLessOrEqualExpr(ctx.minNormalExponent(sort), unbiasedExponent)

    fun negate() = with(ctx) { setSign(mkIte(isNaN, sign, !sign)) }

    fun setSign(newSign: KExpr<KBoolSort>) = UnpackedFp(
        ctx, sort,
        sign = newSign,
        unbiasedExponent = unbiasedExponent,
        normalizedSignificand = normalizedSignificand,
        isNaN = isNaN,
        isInf = isInf,
        isZero = isZero,
        packedFp = packedFp.setSign(newSign)
    )

    fun absolute() = UnpackedFp(
        ctx, sort,
        sign = ctx.falseExpr,
        unbiasedExponent = unbiasedExponent,
        normalizedSignificand = normalizedSignificand,
        isNaN = isNaN,
        isInf = isInf,
        isZero = isZero,
        packedFp = packedFp.setSign(ctx.falseExpr)
    )

    fun <T : KFpSort> extend(expWidth: Int, sigExtension: Int, targetFormat: T): UnpackedFp<T> = UnpackedFp(
        ctx,
        targetFormat,
        sign = sign,
        unbiasedExponent = unbiasedExponent.extendSigned(expWidth),
        normalizedSignificand = ctx.mkBvShiftLeftExpr(
            normalizedSignificand.extendUnsigned(sigExtension),
            ctx.mkBv(sigExtension, significandWidth() + sigExtension.toUInt())
        ),
        isNaN = isNaN,
        isInf = isInf,
        isZero = isZero,
    )

    companion object {
        fun <Fp : KFpSort> KContext.makeNaN(sort: Fp) = UnpackedFp(
            this, sort, sign = falseExpr, unbiasedExponent = defaultExponent(sort),
            normalizedSignificand = defaultSignificand(sort), isNaN = trueExpr, packedFp = makeBvNaN(sort)
        )

        fun <Fp : KFpSort> KContext.makeInf(
            sort: Fp, sign: KExpr<KBoolSort>
        ) = UnpackedFp(
            this, sort, sign, unbiasedExponent = defaultExponent(sort),
            normalizedSignificand = defaultSignificand(sort), isInf = trueExpr, packedFp = makeBvInf(sort, sign)
        )

        fun <Fp : KFpSort> KContext.makeZero(sort: Fp, sign: KExpr<KBoolSort>) = UnpackedFp(
            this, sort, sign, unbiasedExponent = defaultExponent(sort),
            normalizedSignificand = defaultSignificand(sort), isZero = trueExpr, packedFp = makeBvZero(sort, sign)
        )

        fun <T : KFpSort> KContext.iteOp(
            cond: KExpr<KBoolSort>, l: UnpackedFp<T>, r: UnpackedFp<T>
        ): UnpackedFp<T> = UnpackedFp(
            this,
            l.sort,
            sign = mkIte(cond, l.sign, r.sign),
            unbiasedExponent = mkIte(cond, l.unbiasedExponent, r.unbiasedExponent),
            normalizedSignificand = mkIte(cond, l.normalizedSignificand, r.normalizedSignificand),
            isNaN = mkIte(cond, l.isNaN, r.isNaN),
            isInf = mkIte(cond, l.isInf, r.isInf),
            isZero = mkIte(cond, l.isZero, r.isZero),
            packedFp = OptionalPackedFp.packedIte(cond, l.packedFp, r.packedFp)
        )
    }
}
