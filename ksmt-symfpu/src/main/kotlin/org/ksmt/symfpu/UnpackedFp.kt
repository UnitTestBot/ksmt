package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.expr.*
import org.ksmt.expr.printer.ExpressionPrinter
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KFpSort
import org.ksmt.utils.cast
import org.ksmt.utils.uncheckedCast

class UnpackedFp<Fp : KFpSort> private constructor(
    ctx: KContext, override val sort: Fp,
    val bv: KFpToIEEEBvExpr<Fp>,
    private val fp: KExpr<Fp>,
) : KExpr<Fp>(ctx) {

    override fun accept(transformer: KTransformerBase): KExpr<Fp> =
        when (fp) {
            is KFp16Value -> transformer.transform(fp)
            is KFp32Value -> transformer.transform(fp)
            is KFp64Value -> transformer.transform(fp)
            is KFp128Value -> transformer.transform(fp)
            is KConst<Fp> -> transformer.transform(fp)
            is UnpackedFp<Fp> -> fp.accept(transformer)
            is KIteExpr<Fp> -> transformer.transform(fp)
            else -> throw IllegalArgumentException("Unknown fp type: $fp")
        }.cast()

    override fun print(printer: ExpressionPrinter) = with(printer) {
        append("(unpackedFp ")
        append(ctx.mkFpFromBvExpr<Fp>(sign.uncheckedCast(), exponent, significand))
        append(")")
    }


    private val exponentSize = sort.exponentBits.toInt()

    private val size = bv.sort.sizeBits.toInt()

    val sign by lazy { ctx.mkBvExtractExpr(size - 1, size - 1, bv) }
    val exponent by lazy { ctx.mkBvExtractExpr(size - 2, size - exponentSize - 1, bv) }
    val significand by lazy { ctx.mkBvExtractExpr(size - exponentSize - 2, 0, bv) }

    val isNaN by lazy {
        with(ctx) {
            val nanBv = mkFpToIEEEBvExpr(mkFpNan(sort))
            nanBv eq bv
        }
    }
    val isZero by lazy {
        with(ctx) {
            val zeroBv = mkFpToIEEEBvExpr(mkFpZero(false, sort))
            val negativeZeroBv = mkFpToIEEEBvExpr(mkFpZero(true, sort))
            zeroBv eq bv or (negativeZeroBv eq bv)
        }
    }
    val isNegative by lazy {
        with(ctx) {
            sign eq mkBv(1, 1u)
        }
    }
    val isNegativeInfinity by lazy {
        with(ctx) {
            val infBv = mkFpToIEEEBvExpr(mkFpInf(true, sort)) // negative
            infBv eq bv
        }
    }
    val isPositiveInfinity by lazy {
        with(ctx) {
            val infBv = mkFpToIEEEBvExpr(mkFpInf(false, sort)) // positive
            infBv eq bv
        }
    }
    val isInfinite: KOrExpr
        get() {
            return with(ctx) {
                isNegativeInfinity or isPositiveInfinity
            }
        }

    companion object {
        fun <Fp : KFpSort> KContext.unpackedFp(fp: KExpr<Fp>) = UnpackedFp(this, fp.sort, mkFpToIEEEBvExpr(fp), fp)
    }


}