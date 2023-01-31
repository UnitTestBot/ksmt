package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.printer.ExpressionPrinter
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KBvSort
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

    val sign: KExpr<KBvSort> =
        ctx.mkBvExtractExpr(exponentSize + significandSize + 1, exponentSize + significandSize + 1, bv)
    val exponent: KExpr<KBvSort> = ctx.mkBvExtractExpr(exponentSize + significandSize, significandSize, bv)
    val significand: KExpr<KBvSort> = ctx.mkBvExtractExpr(significandSize - 1, 0, bv)

    val isNaN = ctx.mkFpIsNaNExpr(fp)
    val isInfinite = ctx.mkFpIsInfiniteExpr(fp)
    val isZero = ctx.mkFpIsZeroExpr(fp)
    val isNormal = ctx.mkFpIsNormalExpr(fp)
    val isSubnormal = ctx.mkFpIsSubnormalExpr(fp)
    val isNegative = sign

    companion object {
        fun <Fp : KFpSort> KContext.unpackedFp(fp: KExpr<Fp>) = UnpackedFp(fp, this)
    }

}