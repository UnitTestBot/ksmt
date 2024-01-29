package io.ksmt.utils

import io.ksmt.KContext
import io.ksmt.expr.KBitVecValue
import io.ksmt.expr.KFp32Value
import io.ksmt.expr.KFp64Value
import io.ksmt.expr.KFpRoundingMode
import io.ksmt.expr.KFpValue
import io.ksmt.expr.KInt32NumExpr
import io.ksmt.expr.KInt64NumExpr
import io.ksmt.expr.KIntBigNumExpr
import io.ksmt.expr.KIntNumExpr
import io.ksmt.expr.KRealNumExpr
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpSort
import io.ksmt.utils.ArithUtils.RealValue
import io.ksmt.utils.BvUtils.addMaxValueSigned
import io.ksmt.utils.BvUtils.bigIntValue
import io.ksmt.utils.BvUtils.bvMaxValueUnsigned
import io.ksmt.utils.BvUtils.bvOne
import io.ksmt.utils.BvUtils.bvZero
import io.ksmt.utils.BvUtils.isBvMaxValueUnsigned
import io.ksmt.utils.BvUtils.isBvZero
import io.ksmt.utils.BvUtils.minus
import io.ksmt.utils.BvUtils.subMaxValueSigned
import io.ksmt.utils.BvUtils.unsignedLessOrEqual
import java.math.BigInteger
import kotlin.math.IEEErem
import kotlin.math.absoluteValue
import kotlin.math.round
import kotlin.math.sqrt

@Suppress("LargeClass")
object FpUtils {

    @Suppress("MagicNumber")
    fun KFpValue<*>.isZero(): Boolean = when (this) {
        is KFp32Value -> value == 0.0f
        is KFp64Value -> value == 0.0
        else -> biasedExponent.isBvZero() && significand.isBvZero()
    }

    fun KFpValue<*>.isInfinity(): Boolean = when (this) {
        is KFp32Value -> value.isInfinite()
        is KFp64Value -> value.isInfinite()
        else -> biasedExponent.isTopExponent() && significand.isBvZero()
    }

    fun KFpValue<*>.isNaN(): Boolean = when (this) {
        is KFp32Value -> value.isNaN()
        is KFp64Value -> value.isNaN()
        else -> biasedExponent.isTopExponent() && !significand.isBvZero()
    }

    fun KFpValue<*>.isPositive(): Boolean = !signBit

    fun KFpValue<*>.isNegative(): Boolean = signBit

    // Value is denormalized
    fun KFpValue<*>.isSubnormal(): Boolean =
        biasedExponent.isBvZero() && !significand.isBvZero()

    fun KFpValue<*>.isNormal(): Boolean =
        !isZero() && !isSubnormal() && !biasedExponent.isTopExponent()

    fun fpStructurallyEqual(lhs: KFpValue<*>, rhs: KFpValue<*>): Boolean = when {
        lhs.isNaN() && rhs.isNaN() -> true
        lhs.isZero() && rhs.isZero() -> lhs.signBit == rhs.signBit
        else -> fpEq(lhs, rhs)
    }

    fun fpEq(lhs: KFpValue<*>, rhs: KFpValue<*>): Boolean = lhs.fpCompareOperation(
        other = rhs,
        fp32 = { a, b -> a == b },
        fp64 = { a, b -> a == b },
        default = { a, b ->
            when {
                a.isNaN() || b.isNaN() -> false
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
                a.isNaN() || b.isNaN() -> false
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

    fun fpLt(lhs: KFpValue<*>, rhs: KFpValue<*>): Boolean =
        fpLeq(lhs, rhs) && !fpEq(lhs, rhs)

    fun fpGt(lhs: KFpValue<*>, rhs: KFpValue<*>): Boolean = fpLt(rhs, lhs)

    fun fpGeq(lhs: KFpValue<*>, rhs: KFpValue<*>): Boolean = fpLeq(rhs, lhs)

    fun fpNegate(expr: KFpValue<*>): KFpValue<*> = with(expr) {
        when {
            this.isNaN() -> this
            this is KFp32Value -> ctx.mkFp(-value, sort)
            this is KFp64Value -> ctx.mkFp(-value, sort)
            else -> ctx.mkFpBiased(significand, biasedExponent, !signBit, sort)
        }
    }

    fun fpAdd(rm: KFpRoundingMode, lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> =
        lhs.ctx.fpAdd(rm, lhs, rhs)

    fun fpMul(rm: KFpRoundingMode, lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> =
        lhs.ctx.fpMul(rm, lhs, rhs)

    fun fpDiv(rm: KFpRoundingMode, lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> =
        lhs.ctx.fpDiv(rm, lhs, rhs)

    fun fpRem(lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> =
        lhs.ctx.fpRem(lhs, rhs)

    fun fpFma(rm: KFpRoundingMode, x: KFpValue<*>, y: KFpValue<*>, z: KFpValue<*>): KFpValue<*> =
        x.ctx.fpFma(rm, x, y, z)

    fun fpSqrt(rm: KFpRoundingMode, value: KFpValue<*>): KFpValue<*> =
        value.ctx.fpSqrt(rm, value)

    fun fpRoundToIntegral(rm: KFpRoundingMode, value: KFpValue<*>): KFpValue<*> =
        value.ctx.fpRoundToIntegral(rm, value)

    fun <T : KFpSort> fpToFp(rm: KFpRoundingMode, value: KFpValue<*>, toFpSort: T): KFpValue<T> =
        value.ctx.fpToFp(rm, value, toFpSort)

    fun fpMax(lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> = when {
        lhs.isNaN() -> rhs
        rhs.isNaN() -> lhs

        /**
         * IEEE-754 says that max(+0,-0) = +/-0 (unspecified).
         * Therefore, in the case of a different sign, we can return any of [rhs], [lhs].
         * */
        lhs.isZero() && rhs.isZero() -> rhs
        fpGt(lhs, rhs) -> lhs
        else -> rhs
    }

    fun fpMin(lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> = when {
        lhs.isNaN() -> rhs
        rhs.isNaN() -> lhs

        /**
         * IEEE-754 says that min(+0,-0) = +/-0 (unspecified).
         * Therefore, in the case of a different sign, we can return any of [rhs], [lhs].
         * */
        lhs.isZero() && rhs.isZero() -> rhs
        fpLt(lhs, rhs) -> lhs
        else -> rhs
    }

    fun fpRealValueOrNull(value: KFpValue<*>): KRealNumExpr? =
        value.ctx.fpRealValueOrNull(value)

    fun <T : KBvSort> fpBvValueOrNull(
        value: KFpValue<*>,
        rm: KFpRoundingMode,
        bvSort: T,
        signed: Boolean
    ): KBitVecValue<T>? = value.ctx.fpBvValueOrNull(value, rm, bvSort, signed)

    fun <T : KFpSort> fpValueFromReal(rm: KFpRoundingMode, value: KRealNumExpr, sort: T): KFpValue<T> =
        value.ctx.fpValueFromReal(rm, value, sort).uncheckedCast()

    fun <T : KFpSort> fpValueFromBv(
        rm: KFpRoundingMode,
        value: KBitVecValue<*>,
        signed: Boolean,
        sort: T
    ): KFpValue<T> = value.ctx.fpValueFromBv(rm, value, signed, sort).uncheckedCast()

    fun KContext.fpZeroExponentBiased(sort: KFpSort): KBitVecValue<KBvSort> =
        bvZero(sort.exponentBits)

    fun KContext.fpInfExponentBiased(sort: KFpSort): KBitVecValue<KBvSort> =
        fpTopExponentBiased(sort.exponentBits)

    fun KContext.fpNaNExponentBiased(sort: KFpSort): KBitVecValue<KBvSort> =
        fpTopExponentBiased(sort.exponentBits)

    fun KContext.fpZeroSignificand(sort: KFpSort): KBitVecValue<KBvSort> =
        bvZero(sort.significandBits - 1u)

    fun KContext.fpInfSignificand(sort: KFpSort): KBitVecValue<KBvSort> =
        bvZero(sort.significandBits - 1u)

    fun KContext.fpNaNSignificand(sort: KFpSort): KBitVecValue<KBvSort> =
        bvOne(sort.significandBits - 1u)

    fun <T : KFpSort> KContext.mkFpMaxValue(sort: T, signBit: Boolean): KFpValue<T> {
        val maxSignificand = bvMaxValueUnsigned<KBvSort>(sort.significandBits - 1u)
        val maxExponent = fpTopExponentBiased(sort.exponentBits) - bvOne(sort.exponentBits)
        return mkFpBiased(
            significand = maxSignificand,
            biasedExponent = maxExponent,
            signBit = signBit,
            sort = sort
        )
    }

    fun <T : KFpSort> KContext.mkFpOne(sort: T, signBit: Boolean): KFpValue<T> = mkFp(
        signBit = signBit,
        unbiasedExponent = 0,
        significand = 0,
        sort = sort
    )

    fun biasFpExponent(exponent: KBitVecValue<*>, exponentSize: UInt): KBitVecValue<*> {
        check(exponent.sort.sizeBits == exponentSize) {
            "Incorrect exponent size: expected $exponentSize but ${exponent.sort.sizeBits} provided"
        }
        return exponent.addMaxValueSigned()
    }

    fun unbiasFpExponent(exponent: KBitVecValue<*>, exponentSize: UInt): KBitVecValue<*> {
        check(exponent.sort.sizeBits == exponentSize) {
            "Incorrect exponent size: expected $exponentSize but ${exponent.sort.sizeBits} provided"
        }
        return exponent.subMaxValueSigned()
    }

    // All 1 bits
    private fun KContext.fpTopExponentBiased(size: UInt): KBitVecValue<KBvSort> =
        bvMaxValueUnsigned(size)

    private fun KBitVecValue<*>.isTopExponent(): Boolean =
        isBvMaxValueUnsigned()

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

    @Suppress("ComplexMethod")
    private fun KContext.fpAdd(rm: KFpRoundingMode, lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> = when {
        // RNE is JVM default rounding mode ==> use JVM Float +
        rm == KFpRoundingMode.RoundNearestTiesToEven && lhs is KFp32Value -> {
            mkFp(lhs.value + (rhs as KFp32Value).value, lhs.sort)
        }

        // RNE is JVM default rounding mode ==> use JVM Double +
        rm == KFpRoundingMode.RoundNearestTiesToEven && lhs is KFp64Value -> {
            mkFp(lhs.value + (rhs as KFp64Value).value, lhs.sort)
        }

        lhs.isNaN() || rhs.isNaN() -> mkFpNaN(lhs.sort)

        lhs.isInfinity() -> if (rhs.isInfinity() && lhs.signBit != rhs.signBit) {
            mkFpNaN(lhs.sort)
        } else {
            lhs
        }

        rhs.isInfinity() -> if (lhs.isInfinity() && lhs.signBit != rhs.signBit) {
            mkFpNaN(lhs.sort)
        } else {
            rhs
        }

        lhs.isZero() && rhs.isZero() -> {
            val bothNegative = lhs.isNegative() && rhs.isNegative()
            val roundToNegative = rm == KFpRoundingMode.RoundTowardNegative && (lhs.isNegative() != rhs.isNegative())
            if (bothNegative || roundToNegative) {
                mkFpZero(sort = lhs.sort, signBit = true)
            } else {
                mkFpZero(sort = lhs.sort, signBit = false)
            }
        }

        lhs.isZero() -> rhs
        rhs.isZero() -> lhs
        else -> fpUnpackAndAdd(rm, lhs, rhs)
    }

    @Suppress("ComplexMethod")
    private fun KContext.fpMul(rm: KFpRoundingMode, lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> = when {
        // RNE is JVM default rounding mode ==> use JVM Float *
        rm == KFpRoundingMode.RoundNearestTiesToEven && lhs is KFp32Value -> {
            mkFp(lhs.value * (rhs as KFp32Value).value, lhs.sort)
        }

        // RNE is JVM default rounding mode ==> use JVM Double *
        rm == KFpRoundingMode.RoundNearestTiesToEven && lhs is KFp64Value -> {
            mkFp(lhs.value * (rhs as KFp64Value).value, lhs.sort)
        }

        lhs.isNaN() || rhs.isNaN() -> mkFpNaN(lhs.sort)

        lhs.isInfinity() && lhs.isPositive() -> if (rhs.isZero()) {
            mkFpNaN(lhs.sort)
        } else {
            mkFpInf(rhs.signBit, lhs.sort)
        }

        rhs.isInfinity() && rhs.isPositive() -> if (lhs.isZero()) {
            mkFpNaN(lhs.sort)
        } else {
            mkFpInf(lhs.signBit, lhs.sort)
        }

        lhs.isInfinity() && lhs.isNegative() -> if (rhs.isZero()) {
            mkFpNaN(lhs.sort)
        } else {
            mkFpInf(!rhs.signBit, lhs.sort)
        }

        rhs.isInfinity() && rhs.isNegative() -> if (lhs.isZero()) {
            mkFpNaN(lhs.sort)
        } else {
            mkFpInf(!lhs.signBit, lhs.sort)
        }

        lhs.isZero() || rhs.isZero() -> {
            mkFpZero(sort = lhs.sort, signBit = lhs.signBit != rhs.signBit)
        }

        else -> fpUnpackAndMul(rm, lhs, rhs)
    }

    @Suppress("ComplexMethod")
    private fun KContext.fpDiv(rm: KFpRoundingMode, lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> = when {
        // RNE is JVM default rounding mode ==> use JVM Float /
        rm == KFpRoundingMode.RoundNearestTiesToEven && lhs is KFp32Value -> {
            mkFp(lhs.value / (rhs as KFp32Value).value, lhs.sort)
        }

        // RNE is JVM default rounding mode ==> use JVM Double /
        rm == KFpRoundingMode.RoundNearestTiesToEven && lhs is KFp64Value -> {
            mkFp(lhs.value / (rhs as KFp64Value).value, lhs.sort)
        }

        lhs.isNaN() || rhs.isNaN() -> mkFpNaN(lhs.sort)

        lhs.isInfinity() && lhs.isPositive() -> if (rhs.isInfinity()) {
            mkFpNaN(lhs.sort)
        } else {
            mkFpInf(signBit = rhs.signBit, sort = lhs.sort)
        }

        rhs.isInfinity() && rhs.isPositive() -> if (lhs.isInfinity()) {
            mkFpNaN(lhs.sort)
        } else {
            mkFpZero(signBit = lhs.signBit, sort = lhs.sort)
        }

        lhs.isInfinity() && lhs.isNegative() -> if (rhs.isInfinity()) {
            mkFpNaN(lhs.sort)
        } else {
            mkFpInf(signBit = !rhs.signBit, lhs.sort)
        }

        rhs.isInfinity() && rhs.isNegative() -> if (lhs.isInfinity()) {
            mkFpNaN(lhs.sort)
        } else {
            mkFpZero(signBit = !lhs.signBit, sort = lhs.sort)
        }

        rhs.isZero() -> if (lhs.isZero()) {
            mkFpNaN(lhs.sort)
        } else {
            mkFpInf(signBit = lhs.signBit != rhs.signBit, sort = lhs.sort)
        }

        lhs.isZero() -> mkFpZero(signBit = lhs.signBit != rhs.signBit, sort = lhs.sort)

        else -> fpUnpackAndDiv(rm, lhs, rhs)
    }

    private fun KContext.fpRem(lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> = when {
        lhs is KFp32Value -> mkFp(lhs.value.IEEErem((rhs as KFp32Value).value), lhs.sort)
        lhs is KFp64Value -> mkFp(lhs.value.IEEErem((rhs as KFp64Value).value), lhs.sort)
        lhs.isNaN() || rhs.isNaN() -> mkFpNaN(lhs.sort)
        lhs.isInfinity() -> mkFpNaN(lhs.sort)
        rhs.isInfinity() -> lhs
        rhs.isZero() -> mkFpNaN(lhs.sort)
        lhs.isZero() -> lhs
        else -> fpUnpackAndRem(lhs, rhs)
    }

    @Suppress("ComplexMethod")
    private fun KContext.fpFma(
        rm: KFpRoundingMode,
        x: KFpValue<*>,
        y: KFpValue<*>,
        z: KFpValue<*>
    ): KFpValue<*> = when {
        x.isNaN() || y.isNaN() || z.isNaN() -> mkFpNaN(x.sort)

        x.isInfinity() && x.isPositive() -> when {
            y.isZero() -> mkFpNaN(x.sort)
            z.isInfinity() && (x.signBit xor y.signBit xor z.signBit) -> mkFpNaN(x.sort)
            else -> mkFpInf(y.signBit, x.sort)
        }

        y.isInfinity() && y.isPositive() -> when {
            x.isZero() -> mkFpNaN(x.sort)
            z.isInfinity() && (x.signBit xor y.signBit xor z.signBit) -> mkFpNaN(x.sort)
            else -> mkFpInf(x.signBit, x.sort)
        }

        x.isInfinity() && x.isNegative() -> when {
            y.isZero() -> mkFpNaN(x.sort)
            z.isInfinity() && (x.signBit xor y.signBit xor z.signBit) -> mkFpNaN(x.sort)
            else -> mkFpInf(!y.signBit, x.sort)
        }

        y.isInfinity() && y.isNegative() -> when {
            x.isZero() -> mkFpNaN(x.sort)
            z.isInfinity() && (x.signBit xor y.signBit xor z.signBit) -> mkFpNaN(x.sort)
            else -> mkFpInf(!x.signBit, x.sort)
        }

        z.isInfinity() -> z

        x.isZero() || y.isZero() -> if (z.isZero() && (x.signBit xor y.signBit xor z.signBit)) {
            mkFpZero(signBit = rm == KFpRoundingMode.RoundTowardNegative, sort = x.sort)
        } else {
            z
        }

        else -> fpUnpackAndFma(rm, x, y, z)
    }

    private fun KContext.fpSqrt(rm: KFpRoundingMode, value: KFpValue<*>): KFpValue<*> = when {
        // RNE is JVM default rounding mode ==> use JVM Float sqrt
        rm == KFpRoundingMode.RoundNearestTiesToEven && value is KFp32Value -> {
            mkFp(sqrt(value.value), value.sort)
        }

        // RNE is JVM default rounding mode ==> use JVM Double sqrt
        rm == KFpRoundingMode.RoundNearestTiesToEven && value is KFp64Value -> {
            mkFp(sqrt(value.value), value.sort)
        }

        value.isNaN() -> value
        value.isInfinity() && value.isPositive() -> value
        value.isZero() -> value
        value.isNegative() -> mkFpNaN(value.sort)
        else -> fpUnpackAndSqrt(rm, value)
    }

    private fun KContext.fpRoundToIntegral(rm: KFpRoundingMode, value: KFpValue<*>): KFpValue<*> = when {
        // RNE is JVM default rounding mode ==> use JVM Float round
        rm == KFpRoundingMode.RoundNearestTiesToEven && value is KFp32Value -> {
            mkFp(round(value.value), value.sort)
        }

        // RNE is JVM default rounding mode ==> use JVM Double round
        rm == KFpRoundingMode.RoundNearestTiesToEven && value is KFp64Value -> {
            mkFp(round(value.value), value.sort)
        }

        value.isNaN() -> value
        value.isInfinity() -> value
        value.isZero() -> value
        else -> {
            val exponent = value.unbiasedExponentValue()
            when {
                // Negative exponent -> value is not an integral
                exponent.signum() < 0 -> fpRoundToIntegralNegativeExponent(rm, value, exponent)
                // Big exponent -> value is integral
                exponent >= (value.sort.significandBits - 1u).toInt().toBigInteger() -> value
                else -> fpUnpackAndRoundToIntegral(rm, value)
            }
        }
    }

    private fun KContext.fpRoundToIntegralNegativeExponent(
        rm: KFpRoundingMode,
        value: KFpValue<*>,
        exponent: BigInteger
    ): KFpValue<*> = when (rm) {
        KFpRoundingMode.RoundTowardZero -> mkFpZero(value.signBit, value.sort)
        KFpRoundingMode.RoundTowardNegative -> if (value.isNegative()) {
            mkFpOne(sort = value.sort, signBit = true)
        } else {
            mkFpZero(sort = value.sort, signBit = false)
        }

        KFpRoundingMode.RoundTowardPositive -> if (value.isNegative()) {
            mkFpZero(sort = value.sort, signBit = true)
        } else {
            mkFpOne(sort = value.sort, signBit = false)
        }

        KFpRoundingMode.RoundNearestTiesToEven,
        KFpRoundingMode.RoundNearestTiesToAway -> {
            val tie = value.significand.isBvZero() && exponent == (-BigInteger.ONE)
            if (tie) {
                if (rm == KFpRoundingMode.RoundNearestTiesToEven) {
                    mkFpZero(value.signBit, value.sort)
                } else {
                    mkFpOne(value.sort, value.signBit)
                }
            } else {
                if (exponent < -BigInteger.ONE) {
                    mkFpZero(value.signBit, value.sort)
                } else {
                    mkFpOne(value.sort, value.signBit)
                }
            }
        }
    }

    private fun <T : KFpSort> KContext.fpToFp(rm: KFpRoundingMode, value: KFpValue<*>, toFpSort: T): KFpValue<T> =
        when {
            value.isNaN() -> mkFpNaN(toFpSort)
            value.isInfinity() -> mkFpInf(value.signBit, toFpSort)
            value.isZero() -> mkFpZero(value.signBit, toFpSort)
            value.sort == toFpSort -> value.uncheckedCast()
            else -> fpUnpackAndToFp(rm, value, toFpSort)
        }

    private fun KContext.fpRealValueOrNull(value: KFpValue<*>): KRealNumExpr? = when {
        value.isNaN() || value.isInfinity() -> null // Real value is unspecified for NaN and Inf
        else -> fpUnpackAndGetRealValueOrNull(value)
    }

    private fun <T : KBvSort> KContext.fpBvValueOrNull(
        value: KFpValue<*>,
        rm: KFpRoundingMode,
        bvSort: T,
        signed: Boolean
    ): KBitVecValue<T>? = when {
        value.isNaN() || value.isInfinity() -> null // Bv value is unspecified for NaN and Inf
        else -> fpUnpackAndGetBvValueOrNull(value, rm, bvSort.sizeBits, signed)?.uncheckedCast()
    }

    private fun KIntNumExpr.toBigInteger() = when (this) {
        is KInt32NumExpr -> value.toBigInteger()
        is KInt64NumExpr -> value.toBigInteger()
        is KIntBigNumExpr -> value
        else -> decl.value.toBigInteger()
    }

    private fun KContext.fpValueFromReal(rm: KFpRoundingMode, value: KRealNumExpr, sort: KFpSort): KFpValue<*> {
        val realValue = RealValue.create(value.numerator.toBigInteger(), value.denominator.toBigInteger())
        return fpValueFromReal(rm, realValue, sort)
    }

    private fun KContext.fpValueFromBv(
        rm: KFpRoundingMode,
        value: KBitVecValue<*>,
        signed: Boolean,
        sort: KFpSort
    ): KFpValue<*> {
        var intValue = value.bigIntValue().normalizeValue(value.sort.sizeBits)
        if (signed) {
            val upperLimit = powerOfTwo(value.sort.sizeBits - 1u)
            val lowerLimit = -powerOfTwo(value.sort.sizeBits - 1u)
            if (intValue >= upperLimit) {
                intValue -= powerOfTwo(value.sort.sizeBits)
            }
            if (intValue < lowerLimit) {
                intValue += powerOfTwo(value.sort.sizeBits)
            }
        }

        val realValue = RealValue.create(intValue)
        return fpValueFromReal(rm, realValue, sort)
    }

    private fun KContext.fpUnpackAndAdd(
        rm: KFpRoundingMode,
        lhs: KFpValue<*>,
        rhs: KFpValue<*>
    ): KFpValue<*> {
        // Unpack lhs/rhs, this inserts the hidden bit and adjusts the exponent.
        var unpackedLhs = lhs.unpack(normalizeSignificand = false)
        var unpackedRhs = rhs.unpack(normalizeSignificand = false)

        if (unpackedRhs.unbiasedExponent > unpackedLhs.unbiasedExponent) {
            val tmp = unpackedLhs
            unpackedLhs = unpackedRhs
            unpackedRhs = tmp
        }

        return fpAddUnpacked(rm, unpackedLhs, unpackedRhs)
    }

    private fun KContext.fpUnpackAndMul(
        rm: KFpRoundingMode,
        lhs: KFpValue<*>,
        rhs: KFpValue<*>
    ): KFpValue<*> {
        // Unpack lhs/rhs, this inserts the hidden bit and adjusts the exponent.
        val unpackedLhs = lhs.unpack(normalizeSignificand = true)
        val unpackedRhs = rhs.unpack(normalizeSignificand = true)

        return fpMulUnpacked(rm, unpackedLhs, unpackedRhs)
    }

    private fun KContext.fpUnpackAndDiv(
        rm: KFpRoundingMode,
        lhs: KFpValue<*>,
        rhs: KFpValue<*>
    ): KFpValue<*> {
        // Unpack lhs/rhs, this inserts the hidden bit and adjusts the exponent.
        val unpackedLhs = lhs.unpack(normalizeSignificand = true)
        val unpackedRhs = rhs.unpack(normalizeSignificand = true)

        return fpDivUnpacked(rm, unpackedLhs, unpackedRhs)
    }

    private fun KContext.fpUnpackAndRem(
        lhs: KFpValue<*>,
        rhs: KFpValue<*>
    ): KFpValue<*> {
        // Unpack lhs/rhs, this inserts the hidden bit and adjusts the exponent.
        val unpackedLhs = lhs.unpack(normalizeSignificand = true)
        val unpackedRhs = rhs.unpack(normalizeSignificand = true)

        return fpRemUnpacked(unpackedLhs, unpackedRhs)
    }

    private fun KContext.fpUnpackAndFma(
        rm: KFpRoundingMode,
        x: KFpValue<*>,
        y: KFpValue<*>,
        z: KFpValue<*>
    ): KFpValue<*> {
        // Unpack x/y/z, this inserts the hidden bit and adjusts the exponent.
        val unpackedX = x.unpack(normalizeSignificand = true)
        val unpackedY = y.unpack(normalizeSignificand = true)
        val unpackedZ = z.unpack(normalizeSignificand = true)

        return fpFmaUnpacked(rm, unpackedX, unpackedY, unpackedZ)
    }

    private fun KContext.fpUnpackAndSqrt(
        rm: KFpRoundingMode,
        value: KFpValue<*>
    ): KFpValue<*> {
        val unpackedValue = value.unpack(normalizeSignificand = true)

        return fpSqrtUnpacked(rm, unpackedValue)
    }

    private fun KContext.fpUnpackAndRoundToIntegral(
        rm: KFpRoundingMode,
        value: KFpValue<*>
    ): KFpValue<*> {
        val unpackedValue = value.unpack(normalizeSignificand = true)

        return fpRoundToIntegralUnpacked(rm, unpackedValue)
    }

    private fun <T : KFpSort> KContext.fpUnpackAndToFp(
        rm: KFpRoundingMode,
        value: KFpValue<*>,
        toFpSort: T
    ): KFpValue<T> {
        val unpackedValue = value.unpack(normalizeSignificand = true)

        return fpToFpUnpacked(rm, unpackedValue, toFpSort.exponentBits, toFpSort.significandBits).uncheckedCast()
    }

    private fun KContext.fpUnpackAndGetRealValueOrNull(value: KFpValue<*>): KRealNumExpr? {
        val unpackedValue = value.unpack(normalizeSignificand = true)

        val maxAllowedExponent = Int.MAX_VALUE / 2
        if (unpackedValue.unbiasedExponent.abs() >= maxAllowedExponent.toBigInteger()) {
            // We don't want to compute 2^maxAllowedExponent
            return null
        }

        var numerator = unpackedValue.significand
        if (unpackedValue.sign) {
            numerator = -numerator
        }

        var denominator = powerOfTwo(unpackedValue.significandSize - 1u)

        if (unpackedValue.unbiasedExponent >= BigInteger.ZERO) {
            numerator = numerator.mul2k(unpackedValue.unbiasedExponent)
        } else {
            denominator = denominator.mul2k(-unpackedValue.unbiasedExponent)
        }

        return normalizeAndCreateReal(numerator, denominator)
    }

    private fun KContext.normalizeAndCreateReal(numerator: BigInteger, denominator: BigInteger): KRealNumExpr {
        val realValue = RealValue.create(numerator, denominator)
        return mkRealNum(mkIntNum(realValue.numerator), mkIntNum(realValue.denominator))
    }

    private fun KContext.fpUnpackAndGetBvValueOrNull(
        value: KFpValue<*>,
        rm: KFpRoundingMode,
        bvSize: UInt,
        signed: Boolean
    ): KBitVecValue<*>? {
        val unpackedValue = value.unpack(normalizeSignificand = true)

        val maxAllowedExponent = Int.MAX_VALUE / 2
        if (unpackedValue.unbiasedExponent.abs() >= maxAllowedExponent.toBigInteger()) {
            // We don't want to compute 2^maxAllowedExponent
            return null
        }

        val roundedBvValue = unpackedValue.fpBvValueUnpacked(rm)

        val (upperLimit, lowerLimit) = if (!signed) {
            (powerOfTwo(bvSize) - BigInteger.ONE) to BigInteger.ZERO
        } else {
            (powerOfTwo(bvSize - 1u) - BigInteger.ONE) to -(powerOfTwo(bvSize - 1u))
        }

        if (roundedBvValue > upperLimit || roundedBvValue < lowerLimit) {
            return null
        }

        return mkBv(roundedBvValue, bvSize)
    }

    private fun UnpackedFp.fpBvValueUnpacked(rm: KFpRoundingMode): BigInteger {
        val e = unbiasedExponent - significandSize.toInt().toBigInteger() + BigInteger.ONE

        val roundedSignificandBvValue = if (e < BigInteger.ZERO) {
            var value = significand
            var sticky = false

            // We need only 3 last bits for rounding
            var exponentValue = e.abs()
            val three = 3.toBigInteger()
            if (exponentValue > three) {
                val bitsToDrop = (exponentValue - three).ensureSuitablePowerOfTwo()
                sticky = value.anyOfLastKBitsSet(bitsToDrop)
                value = value.div2k(bitsToDrop)
                exponentValue = three
            }

            var round = false
            var lastBit = !value.isEven()

            while (!exponentValue.isZero()) {
                value = value.div2k(1u)
                sticky = sticky || round
                round = lastBit
                lastBit = !value.isEven()
                exponentValue--
            }

            value.roundSignificandIfNeeded(rm, sign, lastBit, round, sticky)
        } else {
            // Number is integral, no rounding needed
            significand.mul2k(e)
        }

        return if (sign) {
            roundedSignificandBvValue.negate()
        } else {
            roundedSignificandBvValue
        }
    }

    /**
     * Floating point addition algorithm described in
     * Knuth, The Art of Computer Programming, Vol. 2: Seminumerical Algorithms
     * Section 4.2.1 (Single-Precision Calculations)
     * Algorithm A (Floating point addition)
     * */
    private fun KContext.fpAddUnpacked(rm: KFpRoundingMode, lhs: UnpackedFp, rhs: UnpackedFp): KFpValue<*> {
        // lhs.exponent >= rhs.exponent => expDelta >= 0
        var expDelta = lhs.unbiasedExponent - rhs.unbiasedExponent
        val significandSizePlusTwo = (lhs.significandSize.toInt() + 2).toBigInteger()
        if (expDelta > significandSizePlusTwo) {
            expDelta = significandSizePlusTwo
        }

        val resultSignificand = fpAddSignificand(lhs.significand, rhs.significand, lhs.sign, rhs.sign, expDelta)

        if (resultSignificand.isZero()) {
            val sign = rm == KFpRoundingMode.RoundTowardNegative
            return mkFpZero(sign, mkFpSort(lhs.exponentSize, lhs.significandSize))
        }

        val resIsNeg = resultSignificand.signum() < 0
        val resSignificandValue = resultSignificand.abs()

        val resultSign = ((!lhs.sign && rhs.sign && resIsNeg)
                || (lhs.sign && !rhs.sign && !resIsNeg)
                || (lhs.sign && rhs.sign))

        val unpackedResult = UnpackedFp(
            lhs.exponentSize, lhs.significandSize, resultSign, lhs.unbiasedExponent, resSignificandValue
        )

        return fpRound(rm, unpackedResult)
    }

    private fun fpAddSignificand(
        lSignificand: BigInteger,
        rSignificand: BigInteger,
        lSign: Boolean,
        rSign: Boolean,
        expDelta: BigInteger
    ): BigInteger {
        // Introduce 3 extra bits into both numbers
        val lhsSignificand = lSignificand.mul2k(3u)
        val rhsSignificand = rSignificand.mul2k(3u)

        // Alignment shift with sticky bit computation.
        val (shiftedRhs, stickyRem) = rhsSignificand.divAndRem2k(expDelta)

        // Significand addition
        return if (lSign != rSign) {
            var res = lhsSignificand - shiftedRhs
            if (!stickyRem.isZero() && res.isEven()) {
                res--
            }
            res
        } else {
            var res = lhsSignificand + shiftedRhs
            if (!stickyRem.isZero() && res.isEven()) {
                res++
            }
            res
        }
    }

    private fun KContext.fpMulUnpacked(rm: KFpRoundingMode, lhs: UnpackedFp, rhs: UnpackedFp): KFpValue<*> {
        val resultSign = lhs.sign xor rhs.sign
        val resultExponent = lhs.unbiasedExponent + rhs.unbiasedExponent

        /**
         * Since significands are normalized (both have MSB 1),
         * the multiplication result will have 2 * significandSize bits
         * */
        val multipliedSignificand = lhs.significand * rhs.significand

        /**
         *  Remove the extra bits, keeping a sticky bit.
         *  Multiplication result is of the form:
         *  [significandSize result bits][4 special bits][significandSize-4 extra bits]
         *  */
        var (normalizedSignificand, stickyRem) = if (lhs.significandSize >= 4u) {
            multipliedSignificand.divAndRem2k(lhs.significandSize - 4u)
        } else {
            // Ensure significand has at least 4 bits (required for rounding)
            val correctedSignificand = multipliedSignificand.mul2k(4u - lhs.significandSize)
            correctedSignificand to BigInteger.ZERO
        }

        if (!stickyRem.isZero() && normalizedSignificand.isEven()){
           normalizedSignificand++
        }

        val unpackedResult = UnpackedFp(
            lhs.exponentSize, lhs.significandSize, resultSign, resultExponent, normalizedSignificand
        )

        return fpRound(rm, unpackedResult)
    }

    private fun KContext.fpDivUnpacked(rm: KFpRoundingMode, lhs: UnpackedFp, rhs: UnpackedFp): KFpValue<*> =
        fpRound(rm, fpDivUnpackedDenormalized(lhs, rhs))

    private fun fpDivUnpackedDenormalized(lhs: UnpackedFp, rhs: UnpackedFp): UnpackedFp {
        val unRoundedDivisionResult = fpDivUnpackedNoRounding(lhs, rhs)

        /**
         *  Remove the extra bits, keeping a sticky bit.
         *  [normalizedResultSignificand] will have at least 4 bits as required for rounding.
         *  */
        var (normalizedSignificand, stickyRem) = unRoundedDivisionResult.significand.divAndRem2k(lhs.significandSize)
        if (!stickyRem.isZero() && normalizedSignificand.isEven()) {
            normalizedSignificand++
        }

        return UnpackedFp(
            lhs.exponentSize,
            lhs.significandSize,
            unRoundedDivisionResult.sign,
            unRoundedDivisionResult.unbiasedExponent,
            normalizedSignificand
        )
    }

    private fun fpDivUnpackedNoRounding(lhs: UnpackedFp, rhs: UnpackedFp): UnpackedFp {
        val resultSign = lhs.sign xor rhs.sign
        val resultExponent = lhs.unbiasedExponent - rhs.unbiasedExponent

        val extraBits = lhs.significandSize + 2u
        val lhsSignificandWithExtraBits = lhs.significand.mul2k(lhs.significandSize + extraBits)

        /**
         * Since significands are normalized (both have MSB 1),
         * the division result will have at least [extraBits] bits.
         * */
        val divisionResultSignificand = lhsSignificandWithExtraBits.divide(rhs.significand)

        return UnpackedFp(
            lhs.exponentSize,
            lhs.significandSize + extraBits,
            resultSign,
            resultExponent,
            divisionResultSignificand
        )
    }

    /**
     * Floating point IEEE rem algorithm described in
     * Intel® 64 and IA-32 Architectures Software Developer’s Manual
     * Section Vol. 2A 3-407 FPREM1—Partial Remainder
     * */
    private fun KContext.fpRemUnpacked(lhs: UnpackedFp, rhs: UnpackedFp): KFpValue<*> {
        val maxExponentDiff = lhs.significandSize.toInt().toBigInteger()

        var st0 = lhs
        do {
            var expDiff: BigInteger
            if (st0.unbiasedExponent < (rhs.unbiasedExponent - BigInteger.ONE)) {
                expDiff = BigInteger.ZERO
            } else {
                expDiff = st0.unbiasedExponent - rhs.unbiasedExponent
                st0 = fpPartialRemainder(st0, rhs, expDiff >= maxExponentDiff)
            }
        } while (expDiff >= maxExponentDiff && !st0.isZero())

        val unpackedResult = st0.copy(significand = st0.significand.mul2k(3u))
        return fpRound(KFpRoundingMode.RoundNearestTiesToEven, unpackedResult)
    }

    private fun fpPartialRemainder(x: UnpackedFp, y: UnpackedFp, partial: Boolean): UnpackedFp {
        // 1. Compute x/y
        val xDividedByY = fpDivUnpackedNoRounding(x, y)

        // 2. Round x/y to integer Q/QQ
        val q = fpPartialRemainderRoundToIntegral(xDividedByY, partial, x.exponentSize, x.significandSize)

        // Integer part is zero ->  x % y = x
        if ((xDividedByY.unbiasedExponent == (-BigInteger.ONE) || partial) && q.significand.isZero()) {
            return x
        }

        // 3. Compute Y*Q / Y*QQ*2^{D-N}
        val denormalizedYQ = UnpackedFp(
            exponentSize = x.exponentSize,
            significandSize = 2u * x.significandSize - 1u,
            sign = x.sign,
            unbiasedExponent = q.unbiasedExponent + y.unbiasedExponent,
            significand = q.significand.multiply(y.significand)
        )
        val yq = fpNormalize(denormalizedYQ)

        // 4. Compute X-Y*Q
        val denormalizedXMinusYQ = fpPartialRemainderSubtractIntegerQuotient(x, yq)
        if (denormalizedXMinusYQ.significand.isZero()) {
            return UnpackedFp(
                x.exponentSize,
                x.significandSize,
                x.sign,
                fpMinExponentValue(x.exponentSize) - BigInteger.ONE,
                BigInteger.ZERO
            )
        }
        val xMinusYQ = fpNormalize(denormalizedXMinusYQ)

        // 5. Rounding
        return fpPartialRemainderRound(xMinusYQ, x.exponentSize, x.significandSize)
    }

    private fun fpPartialRemainderRoundToIntegral(
        value: UnpackedFp,
        partial: Boolean,
        exponentSize: UInt,
        significandSize: UInt
    ): UnpackedFp {
        val rm = if (partial) KFpRoundingMode.RoundTowardZero else KFpRoundingMode.RoundNearestTiesToEven
        val shift = if (partial) {
            (value.significandSize - (significandSize - 1u)).toInt().toBigInteger()
        } else {
            value.significandSize.toInt().toBigInteger() - value.unbiasedExponent
        }

        val significand = fpRoundSignificandToIntegral(value, shift, rm).div2k(significandSize + 3u)
        val denormalizedResult = UnpackedFp(
            exponentSize, significandSize, value.sign, value.unbiasedExponent, significand
        )
        return fpNormalize(denormalizedResult)
    }

    private fun fpPartialRemainderSubtractIntegerQuotient(dividend: UnpackedFp, quotient: UnpackedFp): UnpackedFp {
        val extraBits = dividend.significandSize - 1u
        val expDelta = dividend.unbiasedExponent - quotient.unbiasedExponent
        val minuend = dividend.significand.mul2k(extraBits)

        var subtrahend = quotient.significand
        if (!expDelta.isZero()) {
            var stickyRem = BigInteger.ZERO
            if (expDelta > (dividend.significandSize + 5u).toInt().toBigInteger()) {
                stickyRem = subtrahend
                subtrahend = BigInteger.ZERO
            } else if (expDelta > BigInteger.ZERO) {
                val (quot, rem) = subtrahend.divAndRem2k(expDelta)
                subtrahend = quot
                stickyRem = rem
            } else {
                subtrahend = subtrahend.mul2k(-expDelta)
            }
            if (!stickyRem.isZero() && subtrahend.isEven()) {
                subtrahend++
            }
        }

        var significand = minuend.subtract(subtrahend)

        val neg = significand.signum() < 0
        if (neg) {
            significand = significand.negate()
        }

        return UnpackedFp(
            exponentSize = dividend.exponentSize,
            significandSize = dividend.significandSize + extraBits,
            sign = dividend.sign xor neg,
            unbiasedExponent = dividend.unbiasedExponent,
            significand = significand
        )
    }

    private fun fpPartialRemainderRound(
        value: UnpackedFp,
        exponentSize: UInt,
        significandSize: UInt
    ): UnpackedFp {
        var (significand, rndBits) = value.significand.divAndRem2k(significandSize - 1u)

        // Round to nearest, ties to even
        val mask = powerOfTwo(5u)
        if (rndBits == mask) {
            // tie
            if (!significand.isEven()) {
                significand++
            }
        } else if (rndBits > mask) {
            significand++
        }

        return UnpackedFp(exponentSize, significandSize, value.sign, value.unbiasedExponent, significand)
    }

    private fun UnpackedFp.isZero(): Boolean {
        if (!significand.isZero()) return false
        val botExp = fpMinExponentValue(exponentSize) - BigInteger.ONE
        return unbiasedExponent == botExp
    }

    private fun KContext.fpFmaUnpacked(rm: KFpRoundingMode, x: UnpackedFp, y: UnpackedFp, z: UnpackedFp): KFpValue<*> {
        val multiplicationResult = UnpackedFp(
            exponentSize = x.exponentSize + 2u,
            significandSize = 2u * x.significandSize - 1u,
            sign = x.sign != y.sign,
            unbiasedExponent = x.unbiasedExponent + y.unbiasedExponent,
            significand = x.significand.multiply(y.significand)
        )

        // Extend z significand size to match `multiplicationResult` size
        val extendedZ = z.addSignificandExtraBits(x.significandSize - 1u)
        val res = fpFmaUnpackedAdd(multiplicationResult, extendedZ)

        return fpFmaRound(rm, res, x.exponentSize, x.significandSize)
    }

    private fun fpFmaUnpackedAdd(a: UnpackedFp, b: UnpackedFp): UnpackedFp {
        // Introduce 3 extra bits
        val extendedA = a.addSignificandExtraBits(3u)
        val extendedB = b.addSignificandExtraBits(3u)

        return if (extendedB.unbiasedExponent > extendedA.unbiasedExponent) {
            fpFmaUnpackedAddExtended(extendedB, extendedA)
        } else {
            fpFmaUnpackedAddExtended(extendedA, extendedB)
        }
    }

    private fun fpFmaUnpackedAddExtended(a: UnpackedFp, b: UnpackedFp): UnpackedFp {
        // Alignment shift with sticky bit computation.
        val expDeltaW = a.unbiasedExponent - b.unbiasedExponent
        val maxExprDelta = (b.significandSize + 1u).toInt().toBigInteger()
        val expDelta = expDeltaW.coerceAtMost(maxExprDelta)

        val (alignedBSignificand, stickyRem) = b.significand.divAndRem2k(expDelta)
        val alignmentSticky = !stickyRem.isZero()

        // Significand addition
        return if (a.sign != b.sign) {
            var resSignificand = a.significand - alignedBSignificand

            if (alignmentSticky && resSignificand.isEven()) {
                resSignificand--
            }

            var sign = a.sign
            if (resSignificand.signum() < 0) {
                resSignificand = resSignificand.abs()
                sign = !sign
            }

            a.copy(sign = sign, significand = resSignificand)
        } else {
            var resSignificand = a.significand + alignedBSignificand

            if (alignmentSticky && resSignificand.isEven()) {
                resSignificand++
            }

            a.copy(significand = resSignificand)
        }
    }

    private fun KContext.fpFmaRound(
        rm: KFpRoundingMode,
        value: UnpackedFp,
        exponentSize: UInt,
        significandSize: UInt
    ): KFpValue<*> {
       val (normalizedValue, normalizationStickyBit) = fpFmaRoundNormalizeSignificand(value, exponentSize)

        val valueWithoutExtraBits = fpFmaRoundRemoveExtraBits(normalizedValue, normalizationStickyBit, significandSize)

        if (valueWithoutExtraBits.significand.isZero()) {
            val sign = rm == KFpRoundingMode.RoundTowardNegative
            return mkFpZero(sign, mkFpSort(exponentSize, significandSize))
        }

        val unpackedRes = valueWithoutExtraBits.copy(exponentSize = exponentSize, significandSize = significandSize)
        return fpRound(rm, unpackedRes)
    }

    /**
     * Normalize exponent and significand after addition similar to [fpRoundExponent].
     * */
    private fun fpFmaRoundNormalizeSignificand(value: UnpackedFp, exponentSize: UInt): Pair<UnpackedFp, Boolean> {
        var stickyBit = false
        var resSignificand = value.significand
        var resExponent = value.unbiasedExponent

        if (resSignificand >= powerOfTwo(value.significandSize + 1u)) {
            resExponent++
            stickyBit = !resSignificand.isEven()
            resSignificand = resSignificand.div2k(1u)
        }

        // Actual significand width.
        val sigWidth = value.significand.log2() + 1

        // Significand bits to add or remove
        val significandExtraBits = (value.significandSize + 1u).toInt() - sigWidth

        val maxExponentShift = resExponent - fpMinExponentValue(exponentSize)
        val exponentShift = minOf(maxExponentShift, significandExtraBits.toBigInteger())

        val normalizationShift = maxOf(BigInteger.ZERO, exponentShift)

        resExponent -= normalizationShift
        resSignificand = resSignificand.mul2k(normalizationShift)

        val normalizedValue = value.copy(unbiasedExponent = resExponent, significand = resSignificand)
        return normalizedValue to stickyBit
    }

    /**
     * Remove extra bits as in [fpMulUnpacked].
     * */
    private fun fpFmaRoundRemoveExtraBits(
        value: UnpackedFp,
        stickyBit: Boolean,
        significandSize: UInt
    ): UnpackedFp {
        val exponent: BigInteger
        var significand: BigInteger
        var sticky: Boolean = stickyBit

        if (significandSize >= 4u) {
            val (sig, rem) = value.significand.divAndRem2k(significandSize + 3u - 4u)
            exponent = value.unbiasedExponent
            significand = sig
            sticky = sticky || !rem.isZero()
        } else {
            val exponentDelta = 4u - significandSize + 3u
            significand = value.significand.mul2k(exponentDelta)
            exponent = value.unbiasedExponent - exponentDelta.toInt().toBigInteger()
        }

        if (sticky && significand.isEven()) {
            significand++
        }

        return normalizeUpperBound(exponent, significand, significandSize + 4u) { exp, sig ->
            value.copy(unbiasedExponent = exp, significand = sig)
        }
    }

    private fun UnpackedFp.addSignificandExtraBits(bits: UInt) = copy(
        significandSize = significandSize + bits,
        significand = significand.mul2k(bits)
    )

    private fun KContext.fpSqrtUnpacked(rm: KFpRoundingMode, value: UnpackedFp): KFpValue<*> {
        /**
         * Add significandSize + 6 bits, to ensure that sqrt
         * will have at least significandSize + 3 bits.
         *
         * Add one more (7) bits in case of even exponent to get more precise result.
         * This extra bit is handled by decrementing the [resultExponent] by one.
         * */
        val extraBits = if (value.unbiasedExponent.isEven()) 7u else 6u
        val extendedSignificand = value.significand.mul2k(value.significandSize + extraBits)

        var resultExponent = value.unbiasedExponent.shiftRight(1)
        if (value.unbiasedExponent.isEven()) {
            resultExponent--
        }

        val (rootIsPrecise, rootValue) = extendedSignificand.impreciseSqrt()

        val resultSignificand = if (!rootIsPrecise) {
            var fixedValue = rootValue

            // If the result is inexact, it is 1 too large.
            // We need a sticky bit in the last position here, so we fix that.
            if (fixedValue.isEven()) {
                fixedValue--
            }

            fixedValue
        } else {
            rootValue
        }

        val unpackedResult = UnpackedFp(
            exponentSize = value.exponentSize,
            significandSize = value.significandSize,
            sign = false, // Sqrt is always positive
            unbiasedExponent = resultExponent,
            significand = resultSignificand
        )

        return fpRound(rm, unpackedResult)
    }

    private fun KContext.fpRoundToIntegralUnpacked(rm: KFpRoundingMode, value: UnpackedFp): KFpValue<*> {
        val shift = (value.significandSize - 1u).toInt().toBigInteger() - value.unbiasedExponent

        val resultSignificand = fpRoundSignificandToIntegral(value, shift, rm)
        val resultExponent = value.unbiasedExponent

        // re-normalize
        return normalizeUpperBound(resultExponent, resultSignificand, value.significandSize) { exp, sig ->
            mkRoundedValue(
                rm, exp, sig, value.sign, value.exponentSize, value.significandSize
            )
        }
    }

    private fun fpNormalize(value: UnpackedFp): UnpackedFp {
        if (value.significand.isZero()) return value
        return normalizeUpperBound(value.unbiasedExponent, value.significand, value.significandSize) { e, s ->
            normalizeLowerBound(e, s, value.significandSize) { exp, sig ->
                UnpackedFp(value.exponentSize, value.significandSize, value.sign, exp, sig)
            }
        }
    }

    private inline fun <T> normalizeUpperBound(
        exponent: BigInteger,
        significand: BigInteger,
        significandSize: UInt,
        body: (exponent: BigInteger, significand: BigInteger) -> T
    ): T {
        val exponentDelta = significand.log2() - significandSize.toInt() + 1
        if (exponentDelta <= 0) {
            return body(exponent, significand)
        }

        return body(
            exponent.plus(exponentDelta.toBigInteger()),
            significand.div2k(exponentDelta.toUInt())
        )
    }

    private inline fun <T> normalizeLowerBound(
        exponent: BigInteger,
        significand: BigInteger,
        significandSize: UInt,
        body: (exponent: BigInteger, significand: BigInteger) -> T
    ): T {
        val exponentDelta = significandSize.toInt() - 1 - significand.log2()
        if (exponentDelta <= 0) {
            return body(exponent, significand)
        }

        return body(
            exponent.minus(exponentDelta.toBigInteger()),
            significand.mul2k(exponentDelta.toUInt())
        )
    }

    private fun fpRoundSignificandToIntegral(
        value: UnpackedFp,
        shift: BigInteger,
        rm: KFpRoundingMode
    ): BigInteger {
        var (div, rem) = value.significand.divAndRem2k(shift)

        when (rm) {
            KFpRoundingMode.RoundNearestTiesToEven,
            KFpRoundingMode.RoundNearestTiesToAway -> {
                val shiftMinusOne = powerOfTwo(shift - BigInteger.ONE)
                val tie = rem == shiftMinusOne
                if (tie) {
                    val roundToEven = rm == KFpRoundingMode.RoundNearestTiesToEven && !div.isEven()
                    if (roundToEven || rm == KFpRoundingMode.RoundNearestTiesToAway) {
                        div++
                    }
                } else {
                    val moreThanTie = rem > shiftMinusOne
                    if (moreThanTie) {
                        div++
                    }
                }
            }

            KFpRoundingMode.RoundTowardPositive -> {
                if (!rem.isZero() && !value.sign) {
                    div++
                }
            }

            KFpRoundingMode.RoundTowardNegative -> {
                if (!rem.isZero() && value.sign) {
                    div++
                }
            }

            KFpRoundingMode.RoundTowardZero -> {}
        }

        return div.mul2k(shift)
    }

    @Suppress("MagicNumber")
    private fun KContext.fpToFpUnpacked(
        rm: KFpRoundingMode,
        value: UnpackedFp,
        toExponentSize: UInt,
        toSignificandSize: UInt
    ): KFpValue<*> {

        val significandSizeDelta = toSignificandSize.toInt() - value.significandSize.toInt() + 3 // plus rounding bits

        val resultSignificand = when {
            significandSizeDelta > 0 -> value.significand.mul2k(significandSizeDelta.toUInt())

            significandSizeDelta < 0 -> {
                val bitsToDrop = significandSizeDelta.absoluteValue.toUInt()
                val sticky = value.significand.anyOfLastKBitsSet(bitsToDrop)
                var significand = value.significand.div2k(bitsToDrop)

                if (sticky && significand.isEven()) {
                    significand++
                }

                significand
            }

            else -> value.significand
        }

        val unpackedResult = UnpackedFp(
            exponentSize = toExponentSize,
            significandSize = toSignificandSize,
            sign = value.sign,
            unbiasedExponent = value.unbiasedExponent,
            significand = resultSignificand
        )

        return fpRound(rm, unpackedResult)
    }

    private fun KContext.fpValueFromReal(rm: KFpRoundingMode, value: RealValue, sort: KFpSort): KFpValue<*> {
        val sign = value.isNegative()
        if (value.isZero()) {
            return mkFpZero(sign, sort)
        }

        // Normalize such that 1.0 <= sig < 2.0
        val denormalizedSignificand = value.abs()
        val (resultExponent, normalizedSignificand) = when {
            denormalizedSignificand < RealValue.one -> {
                val nearestInteger = denormalizedSignificand.inverse().floor()

                var nearestPowerOfTwo = nearestInteger.log2()
                if (denormalizedSignificand.inverse() != RealValue.create(nearestPowerOfTwo.toBigInteger())) {
                    nearestPowerOfTwo++
                }

                val nearestPowerOfTwoValue = powerOfTwo(nearestPowerOfTwo.toUInt())
                val significand = denormalizedSignificand.inverse().div(nearestPowerOfTwoValue).inverse()

                -nearestPowerOfTwo.toBigInteger() to significand
            }

            denormalizedSignificand >= RealValue.two -> {
                val nearestInteger = denormalizedSignificand.floor()
                val nearestPowerOfTwo = nearestInteger.log2()

                val nearestPowerOfTwoValue = powerOfTwo(nearestPowerOfTwo.toUInt())
                val significand = denormalizedSignificand.div(nearestPowerOfTwoValue)

                nearestPowerOfTwo.toBigInteger() to significand
            }

            else -> {
                BigInteger.ZERO to denormalizedSignificand
            }
        }

        val significandWithStickyBitsShift = powerOfTwo(sort.significandBits + 3u - 1u)
        var resultSignificand = normalizedSignificand.mul(significandWithStickyBitsShift).floor()

        val dividend = RealValue.create(resultSignificand).div(significandWithStickyBitsShift)
        val stickyRemainder = normalizedSignificand.sub(dividend)

        // sticky
        if (!stickyRemainder.isZero() && resultSignificand.isEven()) {
            resultSignificand++
        }

        val unpackedResult = UnpackedFp(
            sort.exponentBits, sort.significandBits, sign, resultExponent, resultSignificand
        )

        return fpRound(rm, unpackedResult)
    }

    private fun KContext.fpRound(rm: KFpRoundingMode, value: UnpackedFp): KFpValue<*> {
        val (roundedExponent, significandNormalizationShiftSize) = fpRoundExponent(value)

        val normalizedSignificand = significandNormalizationShift(significandNormalizationShiftSize, value.significand)

        val roundedSignificand = roundSignificand(normalizedSignificand, rm, value.sign)

        val (resultExponent, resultSignificand) = postNormalizeExponentAndSignificand(
            value.significandSize, roundedSignificand, roundedExponent
        )

        return mkRoundedValue(
            rm, resultExponent, resultSignificand, value.sign, value.exponentSize, value.significandSize
        )
    }

    private fun postNormalizeExponentAndSignificand(
        significandSize: UInt,
        significand: BigInteger,
        exponent: BigInteger
    ): Pair<BigInteger, BigInteger> {
        var normalizedSignificand = significand
        var normalizedExponent = exponent

        val pSig = powerOfTwo(significandSize)
        if (normalizedSignificand >= pSig) {
            normalizedSignificand = normalizedSignificand.div2k(1u)
            normalizedExponent++
        }

        return normalizedExponent to normalizedSignificand
    }

    @Suppress("MagicNumber")
    private fun fpRoundExponent(value: UnpackedFp): Pair<BigInteger, BigInteger> {
        val eMin = fpMinExponentValue(value.exponentSize)

        // Actual significand width.
        val sigWidth = value.significand.log2() + 1

        /**
         *  Significand bits to add or remove.
         *
         *  Significand is of the form f[-1:0] . f[1:sbits-1] [round,extra,sticky].
         *  We add +3 for [round, extra, sticky]
         *  Special bits be removed from significand explicitly in [roundSignificand].
         *  */
        val significandExtraBits = value.significandSize.toInt() - sigWidth + 3


        //  Exponent value after significand correction.
        val beta = value.unbiasedExponent - significandExtraBits.toBigInteger()

        var significandNormalizationShift: BigInteger
        val exponent: BigInteger
        if (beta < eMin) {
            // denormal significand/TINY
            significandNormalizationShift = value.unbiasedExponent - eMin
            exponent = eMin
        } else {
            significandNormalizationShift = significandExtraBits.toBigInteger()
            exponent = beta
        }

        // No need to shift more than precision
        val significandRightShiftCap = (-(value.significandSize.toInt() + 2)).toBigInteger()
        if (significandNormalizationShift < significandRightShiftCap) {
            significandNormalizationShift = significandRightShiftCap
        }

        return exponent to significandNormalizationShift
    }

    @Suppress("LongParameterList")
    private fun KContext.mkRoundedValue(
        rm: KFpRoundingMode,
        exponent: BigInteger,
        significand: BigInteger,
        sign: Boolean,
        exponentSize: UInt,
        significandSize: UInt
    ): KFpValue<*> {
        val hasOverflow = exponent > fpMaxExponentValue(exponentSize)
        if (hasOverflow) {
            return fpRoundInf(rm, exponentSize, significandSize, sign)
        }

        val leftMostOneBit = powerOfTwo(significandSize - 1u)
        if (significand >= leftMostOneBit) {
            // normal

            // Strips the hidden bit.
            val correctedSignificand = significand - leftMostOneBit

            val exponentBias = powerOfTwo(exponentSize - 1u) - BigInteger.ONE
            val biasedExponent = exponent + exponentBias

            val significandBv = mkBv(correctedSignificand, significandSize - 1u)
            val exponentBv = mkBv(biasedExponent, exponentSize)
            return mkFpBiased(
                significand = significandBv,
                biasedExponent = exponentBv,
                signBit = sign,
                sort = mkFpSort(exponentSize, significandSize)
            )
        } else {
            // denormal
            val significandBv = mkBv(significand, significandSize - 1u)
            val botBiasedExponent = bvZero<KBvSort>(exponentSize)
            return mkFpBiased(
                significand = significandBv,
                biasedExponent = botBiasedExponent,
                signBit = sign,
                sort = mkFpSort(exponentSize, significandSize)
            )
        }
    }

    private fun significandNormalizationShift(
        normalizationShiftSize: BigInteger,
        significand: BigInteger
    ): BigInteger {
        return if (normalizationShiftSize < BigInteger.ZERO) {
            // Right shift
            var (res, stickyRem) = significand.divAndRem2k(-normalizationShiftSize)
            if (!stickyRem.isZero() && res.isEven()) {
                res++
            }
            res
        } else {
            // Left shift
            significand.mul2k(normalizationShiftSize)
        }
    }

    private fun roundSignificand(
        significand: BigInteger,
        rm: KFpRoundingMode,
        sign: Boolean
    ): BigInteger {
        var result = significand

        // last bit
        var sticky = !result.isEven()
        result = result.div2k(1u)

        // pre-last bit
        sticky = sticky || !result.isEven()
        result = result.div2k(1u)

        // pre-pre-last bit
        val round = !result.isEven()
        result = result.div2k(1u)

        val last = !result.isEven()

        // The significand has the right size now, but we might have to increment it
        // depending on the sign, the last/round/sticky bits, and the rounding mode.
        return result.roundSignificandIfNeeded(rm, sign, last, round, sticky)
    }

    private fun BigInteger.roundSignificandIfNeeded(
        rm: KFpRoundingMode,
        sign: Boolean,
        last: Boolean,
        round: Boolean,
        sticky: Boolean
    ): BigInteger {
        val inc = when (rm) {
            KFpRoundingMode.RoundNearestTiesToEven -> round && (last || sticky)
            KFpRoundingMode.RoundNearestTiesToAway -> round
            KFpRoundingMode.RoundTowardPositive -> (!sign && (round || sticky))
            KFpRoundingMode.RoundTowardNegative -> (sign && (round || sticky))
            KFpRoundingMode.RoundTowardZero -> false
        }
        return if (inc) this.inc() else this
    }

    private fun KContext.fpRoundInf(
        rm: KFpRoundingMode,
        exponentSize: UInt,
        significandSize: UInt,
        sign: Boolean
    ): KFpValue<*> {
        val sort = mkFpSort(exponentSize, significandSize)
        return if (!sign) {
            if (rm == KFpRoundingMode.RoundTowardZero || rm == KFpRoundingMode.RoundTowardNegative) {
                mkFpMaxValue(signBit = false, sort = sort)
            } else {
                mkFpInf(signBit = false, sort = sort)
            }
        } else {
            if (rm == KFpRoundingMode.RoundTowardZero || rm == KFpRoundingMode.RoundTowardPositive) {
                mkFpMaxValue(signBit = true, sort = sort)
            } else {
                mkFpInf(signBit = true, sort = mkFpSort(exponentSize, significandSize))
            }
        }
    }

    private data class UnpackedFp(
        val exponentSize: UInt,
        val significandSize: UInt,
        val sign: Boolean,
        val unbiasedExponent: BigInteger,
        val significand: BigInteger
    )

    private fun KFpValue<*>.unpack(normalizeSignificand: Boolean): UnpackedFp = when {
        isNormal() -> unpackNormalValue()
        !normalizeSignificand -> unpackSubnormalValueWithoutSignificandNormalization()
        else -> unpackSubnormalValueAndNormalizeSignificand()
    }

    private fun KFpValue<*>.unbiasedExponentValue(): BigInteger {
        val exponentBias = powerOfTwo(sort.exponentBits - 1u) - BigInteger.ONE
        val biasedExponentValue = biasedExponent.bigIntValue().normalizeValue(sort.exponentBits)
        return biasedExponentValue - exponentBias
    }

    private fun KFpValue<*>.unpackNormalValue(): UnpackedFp {
        val unbiasedExponent = unbiasedExponentValue()

        val significandValue = significand.bigIntValue().normalizeValue(sort.significandBits - 1u)
        val significandWithHiddenBit = significandValue + powerOfTwo(sort.significandBits - 1u)

        return UnpackedFp(
            sort.exponentBits, sort.significandBits, signBit, unbiasedExponent, significandWithHiddenBit
        )
    }

    private fun KFpValue<*>.unpackSubnormalValueWithoutSignificandNormalization(): UnpackedFp {
        val normalizedExponent = fpMinExponentValue(sort.exponentBits)
        val significandValue = significand.bigIntValue().normalizeValue(sort.significandBits - 1u)
        return UnpackedFp(
            sort.exponentBits, sort.significandBits, signBit, normalizedExponent, significandValue
        )
    }

    private fun KFpValue<*>.unpackSubnormalValueAndNormalizeSignificand(): UnpackedFp {
        var normalizedExponent = fpMinExponentValue(sort.exponentBits)

        val significandValue = significand.bigIntValue().normalizeValue(sort.significandBits - 1u)
        val normalizedSignificand = if (significandValue.isZero()) {
            significandValue
        } else {
            val powerToNormalize = maxOf(0, sort.significandBits.toInt() - significandValue.log2() - 1)
            normalizedExponent -= powerToNormalize.toBigInteger()
            significandValue.mul2k(powerToNormalize.toUInt())
        }

        return UnpackedFp(
            sort.exponentBits, sort.significandBits, signBit, normalizedExponent, normalizedSignificand
        )
    }

    private fun fpMinExponentValue(exponent: UInt): BigInteger {
        val exponentBias = powerOfTwo(exponent - 1u) - BigInteger.ONE
        return (-exponentBias) + BigInteger.ONE
    }

    private fun fpMaxExponentValue(exponent: UInt): BigInteger {
        return powerOfTwo(exponent - 1u) - BigInteger.ONE
    }

    private fun BigInteger.ensureSuitablePowerOfTwo(): UInt {
        check(signum() >= 0) { "Negative power" }
        check(bitLength() < Int.SIZE_BITS) {
            "Number 2^$this is too big to be represented as a BigInteger"
        }
        return intValueExact().toUInt()
    }

    private fun powerOfTwo(power: BigInteger): BigInteger = powerOfTwo(power.ensureSuitablePowerOfTwo())

    private fun BigInteger.isEven(): Boolean = toInt() % 2 == 0
    private fun BigInteger.isZero(): Boolean = signum() == 0
    private fun BigInteger.log2(): Int = bitLength() - 1
    private fun BigInteger.anyOfLastKBitsSet(k: UInt): Boolean =
        lowestSetBit.let { it != -1 && it <= k.toInt() }

    private fun BigInteger.mul2k(k: BigInteger): BigInteger = mul2k(k.ensureSuitablePowerOfTwo())
    private fun BigInteger.mul2k(k: UInt): BigInteger = shiftLeft(k.toInt())

    private fun BigInteger.div2k(k: UInt): BigInteger {
        val result = abs().shiftRight(k.toInt())
        return if (signum() >= 0) result else result.negate()
    }

    private fun BigInteger.divAndRem2k(power: BigInteger): Pair<BigInteger, BigInteger> =
        divAndRem2k(power.ensureSuitablePowerOfTwo())

    private fun BigInteger.divAndRem2k(power: UInt): Pair<BigInteger, BigInteger> {
        val quotient = div2k(power)
        val remainderMask = powerOfTwo(power) - BigInteger.ONE
        val remainder = abs().and(remainderMask)
        return quotient to remainder
    }

    private fun UInt.ceilDiv(other: UInt): UInt =
        if (this % other == 0u) {
            this / other
        } else {
            this / other + 1u
        }

    private fun BigInteger.impreciseSqrt(): Pair<Boolean, BigInteger> {
        check(signum() >= 0) { "Sqrt of negative value" }

        if (isZero()) {
            // precise
            return true to BigInteger.ZERO
        }

        /**
         * Initial approximation.
         * We have that:
         * 2^{log2(this)} <= this <= 2^{(log2(this) + 1)}
         * Thus:
         * 2^{floor_div(log2(this), 2)} <= this^{1/2} <=  2^{ceil_div(log2(this) + 1, 2)}
         * */
        val k = this.log2().toUInt()
        var lower = powerOfTwo(k / 2u)
        var upper = powerOfTwo((k + 1u).ceilDiv(2u))

        if (lower == upper) {
            return true to lower
        }

        // Refine using bisection.
        while (true) {
            val mid = (upper + lower).div2k(1u)
            val midSquared = mid.pow(2)

            // We have a precise square root
            if (this == midSquared) {
                return true to mid
            }

            // No precise square root exists
            if (mid == lower || mid == upper) {
                return false to upper
            }

            // Update search bounds
            if (midSquared < this) {
                lower = mid
            } else {
                upper = mid
            }
        }
    }
}
