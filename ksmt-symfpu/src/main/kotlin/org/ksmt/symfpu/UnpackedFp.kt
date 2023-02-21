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

private fun KExpr<KBvSort>.resize(newSize: UInt, ctx: KContext): KExpr<KBvSort> {
    val width = sort.sizeBits
    return if (newSize > width) {
        ctx.mkBvZeroExtensionExpr((newSize - width).toInt(), this)
    } else {
        this
    }
}

class UnpackedFp<Fp : KFpSort> private constructor(
    ctx: KContext, override val sort: Fp,
    val sign: KExpr<KBoolSort>, // negative
    val exponent: KExpr<KBvSort>,
    val significand: KExpr<KBvSort>,
    val isNaN: KExpr<KBoolSort> = ctx.mkFalse(),
    val isInf: KExpr<KBoolSort> = ctx.mkFalse(),
    val isZero: KExpr<KBoolSort> = ctx.mkFalse(),
) : KExpr<Fp>(ctx) {

    constructor(
        ctx: KContext, sort: Fp, sign: KExpr<KBoolSort>, exponent: KExpr<KBvSort>, significand: KExpr<KBvSort>
    ) : this(
        ctx,
        sort,
        sign,
        exponent.matchWidthSigned(ctx, ctx.defaultExponent(sort)),
        significand,
        ctx.mkFalse(),
        ctx.mkFalse(),
        ctx.mkFalse()
    )

    fun signBv() = ctx.boolToBv(sign)

    override fun accept(transformer: KTransformerBase): KExpr<Fp> =
        throw IllegalArgumentException("Leaked unpackedFp: $this")

    override fun print(printer: ExpressionPrinter) = with(printer) {
        append("(unpackedFp ")
        append("sign: ${sign}, exponent: $exponent, significand: $significand")
        append(")")
    }

    override fun internEquals(other: Any): Boolean =
        structurallyEqual(other) { listOf(sign, exponent, significand, isNaN, isInf, isZero) }

    override fun internHashCode(): Int = hash(listOf(sign, exponent, significand, isNaN, isInf, isZero))


    val isNegative = sign
    val isNegativeInfinity by lazy {
        with(ctx) {
            isNegative and isInf
        }
    }
    val isPositiveInfinity by lazy {
        with(ctx) {
            !isNegative and isInf
        }
    }

    // for tests
    internal fun toFp() = ctx.pack(this)

    // Moves the leading 1 up to the correct position, adjusting the
    // exponent as required.
    fun normaliseUp(): UnpackedFp<Fp> = with(ctx) {
        val normal = normaliseShift(significand)

        val exponentWidth = exponent.sort.sizeBits
        check(
            normal.shiftAmount.sort.sizeBits < exponentWidth
        ) // May lose data / be incorrect for very small exponents and very large significands

        val signedAlignAmount = normal.shiftAmount.resize(exponentWidth, ctx)
        val correctedExponent = mkBvSubExpr(exponent, signedAlignAmount)

        // Optimisation : could move the zero detect version in if used in all cases
        //  catch - it zero detection in unpacking is different.
        return UnpackedFp(ctx, sort, sign, correctedExponent, normal.normalised)
    }

    fun inNormalRange() = ctx.mkBvSignedLessOrEqualExpr(ctx.minNormalExponent(sort), exponent)

    companion object {


        fun <Fp : KFpSort> KContext.makeNaN(sort: Fp) = UnpackedFp(
            this, sort, sign = falseExpr, exponent = defaultExponent(sort),
            significand = defaultSignificand(sort), isNaN = trueExpr,
        )

        fun <Fp : KFpSort> KContext.makeInf(
            sort: Fp, sign: KExpr<KBoolSort>
        ) = UnpackedFp(
            this, sort, sign, exponent = defaultExponent(sort),
            significand = defaultSignificand(sort), isInf = trueExpr,
        )

        fun <Fp : KFpSort> KContext.makeZero(sort: Fp, sign: KExpr<KBoolSort>) = UnpackedFp(
            this, sort, sign, exponent = defaultExponent(sort),
            significand = defaultSignificand(sort), isZero = trueExpr,
        )

        fun <T : KFpSort> KContext.iteOp(
            cond: KExpr<KBoolSort>, l: UnpackedFp<T>, r: UnpackedFp<T>
        ): UnpackedFp<T> {
            return UnpackedFp(
                this,
                l.sort,
                sign = mkIte(cond, l.sign, r.sign),
                exponent = mkIte(cond, l.exponent, r.exponent),
                significand = mkIte(cond, l.significand, r.significand),
                isNaN = mkIte(cond, l.isNaN, r.isNaN),
                isInf = mkIte(cond, l.isInf, r.isInf),
                isZero = mkIte(cond, l.isZero, r.isZero)
            )
        }
    }

}