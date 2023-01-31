package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.printer.ExpressionPrinter
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KFpSort

class UnpackedFp<Fp : KFpSort>(
    private val fp: KExpr<Fp>,
    ctx: KContext
) : KExpr<Fp>(ctx) {
    override val sort: Fp = fp.sort

    override fun accept(transformer: KTransformerBase): KExpr<Fp> {
        return fp
    }

    override fun print(printer: ExpressionPrinter) = with(printer) {
        append("(unpackedFp")
        append(fp)
        append(")")
    }


    private val exponentSize = sort.exponentBits.toInt()
    private val significandSize = sort.significandBits.toInt()

    val bv = ctx.mkFpToIEEEBvExpr(fp)
    val shift = sort.exponentShiftSize()

    private val size = bv.sort.sizeBits.toInt()

    val sign = ctx.mkBvExtractExpr(size - 1, size - 1, bv)
    val exponent = ctx.mkBvExtractExpr(size - 2, size - exponentSize - 1, bv)
    val significand = ctx.mkBvExtractExpr(size - exponentSize - 2, 0, bv)


    val isNaN by lazy { ctx.mkFpIsNaNExpr(fp) }
    val isInfinite by lazy { ctx.mkFpIsInfiniteExpr(fp) }
    val isZero by lazy { ctx.mkFpIsZeroExpr(fp) }
    val isNormal by lazy { ctx.mkFpIsNormalExpr(fp) }
    val isSubnormal by lazy { ctx.mkFpIsSubnormalExpr(fp) }
    val isNegative by lazy { ctx.mkFpIsNegativeExpr(fp) }


    val isNegativeInfinity get() = with(ctx) { isInfinite and isNegative }
    val isPositiveInfinity get() = with(ctx) { isInfinite and isNegative.not() }

    companion object {
        fun <Fp : KFpSort> KContext.unpackedFp(fp: KExpr<Fp>) = UnpackedFp(fp, this)
    }


}