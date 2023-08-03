package io.ksmt.symfpu.operations

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpSort
import io.ksmt.utils.FpUtils.fpInfExponentBiased
import io.ksmt.utils.FpUtils.fpInfSignificand
import io.ksmt.utils.FpUtils.fpNaNExponentBiased
import io.ksmt.utils.FpUtils.fpNaNSignificand
import io.ksmt.utils.FpUtils.fpZeroExponentBiased
import io.ksmt.utils.FpUtils.fpZeroSignificand


sealed class OptionalPackedFp {
    fun setSign(sign: KExpr<KBoolSort>) = when (this) {
        is PackedFp -> copy(sign = sign)
        NoPackedFp -> this
    }

    fun hasPackedFp() = this is PackedFp

    companion object {
        fun packedIte(
            cond: KExpr<KBoolSort>,
            trueBranch: OptionalPackedFp,
            falseBranch: OptionalPackedFp,
        ): OptionalPackedFp = with(cond.ctx) {
            when {
                trueBranch is PackedFp && falseBranch is PackedFp -> PackedFp(
                    sign = mkIte(cond, trueBranch.sign, falseBranch.sign),
                    exponent = mkIte(cond, trueBranch.exponent, falseBranch.exponent),
                    significand = mkIte(cond, trueBranch.significand, falseBranch.significand)
                )

                else -> NoPackedFp
            }
        }

        fun KContext.makeBvNaN(sort: KFpSort): OptionalPackedFp = PackedFp(
            sign = falseExpr,
            exponent = fpNaNExponentBiased(sort),
            significand = fpNaNSignificand(sort)
        )

        fun KContext.makeBvInf(sort: KFpSort, sign: KExpr<KBoolSort>): OptionalPackedFp = PackedFp(
            sign = sign,
            exponent = fpInfExponentBiased(sort),
            significand = fpInfSignificand(sort)
        )

        fun KContext.makeBvZero(sort: KFpSort, sign: KExpr<KBoolSort>): OptionalPackedFp = PackedFp(
            sign = sign,
            exponent = fpZeroExponentBiased(sort),
            significand = fpZeroSignificand(sort)
        )
    }
}

data class PackedFp(
    val sign: KExpr<KBoolSort>,
    val exponent: KExpr<KBvSort>,
    val significand: KExpr<KBvSort>
) : OptionalPackedFp() {
    val sort = sign.ctx.mkFpSort(exponent.sort.sizeBits, significand.sort.sizeBits + 1u)

    fun toIEEEBv() = with(sign.ctx) { mkBvConcatExpr(boolToBv(sign), exponent, significand) }

    infix fun eq(packedBv: PackedFp): KExpr<KBoolSort> = with(sign.ctx) {
        mkAnd(sign eq packedBv.sign, exponent eq packedBv.exponent, significand eq packedBv.significand)
    }
}

object NoPackedFp : OptionalPackedFp()
