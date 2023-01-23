package org.ksmt.utils

import org.ksmt.KContext
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KFp32Value
import org.ksmt.expr.KFp64Value
import org.ksmt.expr.KFpValue
import org.ksmt.sort.KFpSort
import org.ksmt.utils.BvUtils.bvMaxValueUnsigned
import org.ksmt.utils.BvUtils.bvOne
import org.ksmt.utils.BvUtils.bvZero
import org.ksmt.utils.BvUtils.unsignedLessOrEqual

object FpUtils {

    @Suppress("MagicNumber")
    fun KFpValue<*>.isZero(): Boolean = when (this) {
        is KFp32Value -> value == 0.0f || value == -0.0f
        is KFp64Value -> value == 0.0 || value == -0.0
        else -> biasedExponent.isZero() && significand.isZero()
    }

    fun KFpValue<*>.isInfinity(): Boolean = when (this) {
        is KFp32Value -> value.isInfinite()
        is KFp64Value -> value.isInfinite()
        else -> biasedExponent.isTopExponent(sort) && significand.isZero()
    }

    fun KFpValue<*>.isNan(): Boolean = when (this) {
        is KFp32Value -> value.isNaN()
        is KFp64Value -> value.isNaN()
        else -> biasedExponent.isTopExponent(sort) && !significand.isZero()
    }

    fun KFpValue<*>.isPositive(): Boolean = !signBit

    fun KFpValue<*>.isNegative(): Boolean = signBit

    // Value is denormalized
    fun KFpValue<*>.isSubnormal(): Boolean =
        biasedExponent.isZero() && !significand.isZero()

    fun KFpValue<*>.isNormal(): Boolean =
        !isZero() && !isSubnormal() && !biasedExponent.isTopExponent(sort)


    fun fpStructurallyEqual(lhs: KFpValue<*>, rhs: KFpValue<*>): Boolean = when {
        lhs.isNan() && rhs.isNan() -> true
        lhs.isZero() && rhs.isZero() -> lhs.signBit == rhs.signBit
        else -> fpEq(lhs, rhs)
    }

    fun fpEq(lhs: KFpValue<*>, rhs: KFpValue<*>): Boolean = lhs.fpCompareOperation(
        other = rhs,
        fp32 = { a, b -> a == b },
        fp64 = { a, b -> a == b },
        default = { a, b ->
            when {
                a.isNan() || b.isNan() -> false
                a.isZero() && b.isZero() -> true
                else -> a.signBit == b.signBit
                        && a.biasedExponent == b.biasedExponent
                        && a.significand == b.significand
            }
        }
    )

    fun fpLeq(lhs: KFpValue<*>, rhs: KFpValue<*>): Boolean = lhs.fpCompareOperation(
        other = rhs,
        fp32 = { a, b -> a <= b },
        fp64 = { a, b -> a <= b },
        default = { a, b ->
            when {
                a.isNan() || b.isNan() -> false
                a.isZero() && b.isZero() -> true
                else -> if (a.signBit == b.signBit) {
                    if (a.isPositive()) {
                        if (a.biasedExponent == b.biasedExponent) {
                            a.significand.unsignedLessOrEqual(b.significand)
                        } else {
                            a.biasedExponent.unsignedLessOrEqual(b.biasedExponent)
                        }
                    } else {
                        if (a.biasedExponent == b.biasedExponent) {
                            b.significand.unsignedLessOrEqual(a.significand)
                        } else {
                            b.biasedExponent.unsignedLessOrEqual(a.biasedExponent)
                        }
                    }
                } else {
                    a.isNegative()
                }
            }
        }
    )

    fun fpLt(lhs: KFpValue<*>, rhs: KFpValue<*>): Boolean = lhs.fpCompareOperation(
        other = rhs,
        fp32 = { a, b -> a < b },
        fp64 = { a, b -> a < b },
        default = { a, b -> fpLeq(a, b) && !fpEq(a, b) }
    )

    fun fpGt(lhs: KFpValue<*>, rhs: KFpValue<*>): Boolean = fpLt(rhs, lhs)

    fun fpGeq(lhs: KFpValue<*>, rhs: KFpValue<*>): Boolean = fpLeq(lhs, rhs)

    fun fpNegate(expr: KFpValue<*>): KFpValue<*> = with(expr) {
        when (this) {
            is KFp32Value -> ctx.mkFp(-value, sort)
            is KFp64Value -> ctx.mkFp(-value, sort)
            else -> if (isNan()) {
                this
            } else {
                ctx.mkFpBiased(significand, biasedExponent, !signBit, sort)
            }
        }
    }

    fun KContext.fpZeroExponentBiased(sort: KFpSort): KBitVecValue<*> =
        bvZero(sort.exponentBits)

    fun KContext.fpInfExponentBiased(sort: KFpSort): KBitVecValue<*> =
        fpTopExponentBiased(sort.exponentBits)

    fun KContext.fpNanExponentBiased(sort: KFpSort): KBitVecValue<*> =
        fpTopExponentBiased(sort.exponentBits)

    fun KContext.fpZeroSignificand(sort: KFpSort): KBitVecValue<*> =
        bvZero(sort.significandBits - 1u)

    fun KContext.fpInfSignificand(sort: KFpSort): KBitVecValue<*> =
        bvZero(sort.significandBits - 1u)

    fun KContext.fpNanSignificand(sort: KFpSort): KBitVecValue<*> =
        bvOne(sort.significandBits - 1u)

    // All 1 bits
    private fun KContext.fpTopExponentBiased(size: UInt): KBitVecValue<*> =
        bvMaxValueUnsigned(size)

    private fun KBitVecValue<*>.isTopExponent(sort: KFpSort): Boolean =
        this == ctx.fpTopExponentBiased(sort.exponentBits)

    private fun KBitVecValue<*>.isZero(): Boolean =
        this == ctx.bvZero(sort.sizeBits)

    private inline fun KFpValue<*>.fpCompareOperation(
        other: KFpValue<*>,
        crossinline fp32: (Float, Float) -> Boolean,
        crossinline fp64: (Double, Double) -> Boolean,
        crossinline default: (KFpValue<*>, KFpValue<*>) -> Boolean
    ): Boolean = when (this) {
        is KFp32Value -> fp32(value, (other as KFp32Value).value)
        is KFp64Value -> fp64(value, (other as KFp64Value).value)
        else -> default(this, other)
    }
}
