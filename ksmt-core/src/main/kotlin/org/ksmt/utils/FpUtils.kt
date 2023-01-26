package org.ksmt.utils

import org.ksmt.KContext
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KFp32Value
import org.ksmt.expr.KFp64Value
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.expr.KFpValue
import org.ksmt.sort.KFpSort
import org.ksmt.utils.BvUtils.bigIntValue
import org.ksmt.utils.BvUtils.bvMaxValueSigned
import org.ksmt.utils.BvUtils.bvMaxValueUnsigned
import org.ksmt.utils.BvUtils.bvOne
import org.ksmt.utils.BvUtils.bvZero
import org.ksmt.utils.BvUtils.isBvMaxValueUnsigned
import org.ksmt.utils.BvUtils.isBvZero
import org.ksmt.utils.BvUtils.minus
import org.ksmt.utils.BvUtils.plus
import org.ksmt.utils.BvUtils.unsignedLessOrEqual
import java.math.BigInteger
import kotlin.math.round
import kotlin.math.sqrt

object FpUtils {

    @Suppress("MagicNumber")
    fun KFpValue<*>.isZero(): Boolean = when (this) {
        is KFp32Value -> value == 0.0f || value == -0.0f
        is KFp64Value -> value == 0.0 || value == -0.0
        else -> biasedExponent.isBvZero() && significand.isBvZero()
    }

    fun KFpValue<*>.isInfinity(): Boolean = when (this) {
        is KFp32Value -> value.isInfinite()
        is KFp64Value -> value.isInfinite()
        else -> biasedExponent.isTopExponent() && significand.isBvZero()
    }

    fun KFpValue<*>.isNan(): Boolean = when (this) {
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

    fun fpLt(lhs: KFpValue<*>, rhs: KFpValue<*>): Boolean =
        fpLeq(lhs, rhs) && !fpEq(lhs, rhs)

    fun fpGt(lhs: KFpValue<*>, rhs: KFpValue<*>): Boolean = fpLt(rhs, lhs)

    fun fpGeq(lhs: KFpValue<*>, rhs: KFpValue<*>): Boolean = fpLeq(rhs, lhs)

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

    fun fpAdd(rm: KFpRoundingMode, lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> =
        lhs.ctx.fpAdd(rm, lhs, rhs)

    fun fpMul(rm: KFpRoundingMode, lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> =
        lhs.ctx.fpMul(rm, lhs, rhs)

    fun fpDiv(rm: KFpRoundingMode, lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> =
        lhs.ctx.fpDiv(rm, lhs, rhs)

    fun fpSqrt(rm: KFpRoundingMode, value: KFpValue<*>): KFpValue<*> =
        value.ctx.fpSqrt(rm, value)

    fun fpRoundToIntegral(rm: KFpRoundingMode, value: KFpValue<*>): KFpValue<*> =
        value.ctx.fpRoundToIntegral(rm, value)

    fun <T : KFpSort> fpToFp(rm: KFpRoundingMode, value: KFpValue<*>, toFpSort: T): KFpValue<T> =
        value.ctx.fpToFp(rm, value, toFpSort)

    fun fpMax(lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> = when {
        lhs.isNan() -> rhs
        rhs.isNan() -> lhs
        lhs.isZero() && rhs.isZero() && lhs.signBit != rhs.signBit -> {
            error("Unspecified: IEEE-754 says that max(+0,-0) = +/-0")
        }

        lhs.isZero() && rhs.isZero() -> rhs
        fpGt(lhs, rhs) -> lhs
        else -> rhs
    }

    fun fpMin(lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> = when {
        lhs.isNan() -> rhs
        rhs.isNan() -> lhs
        lhs.isZero() && rhs.isZero() && lhs.signBit != rhs.signBit -> {
            error("Unspecified: IEEE-754 says that min(+0,-0) = +/-0")
        }

        lhs.isZero() && rhs.isZero() -> rhs
        fpLt(lhs, rhs) -> lhs
        else -> rhs
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

    fun <T : KFpSort> KContext.mkFpMaxValue(sort: T, signBit: Boolean): KFpValue<T> {
        val maxSignificand = bvMaxValueUnsigned(sort.significandBits - 1u)
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

    fun KContext.biasFpExponent(exponent: KBitVecValue<*>, exponentSize: UInt): KBitVecValue<*> =
        exponent + bvMaxValueSigned(exponentSize)

    fun KContext.unbiasFpExponent(exponent: KBitVecValue<*>, exponentSize: UInt): KBitVecValue<*> =
        exponent - bvMaxValueSigned(exponentSize)

    // All 1 bits
    private fun KContext.fpTopExponentBiased(size: UInt): KBitVecValue<*> =
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

    private fun KContext.fpAdd(rm: KFpRoundingMode, lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> = when {
        // RNE is JVM default rounding mode
        rm == KFpRoundingMode.RoundNearestTiesToEven && lhs is KFp32Value -> {
            mkFp(lhs.value + (rhs as KFp32Value).value, lhs.sort)
        }

        rm == KFpRoundingMode.RoundNearestTiesToEven && lhs is KFp64Value -> {
            mkFp(lhs.value + (rhs as KFp64Value).value, lhs.sort)
        }

        lhs.isNan() || rhs.isNan() -> mkFpNan(lhs.sort)

        lhs.isInfinity() -> if (rhs.isInfinity() && lhs.signBit != rhs.signBit) {
            mkFpNan(lhs.sort)
        } else {
            lhs
        }

        rhs.isInfinity() -> if (lhs.isInfinity() && lhs.signBit != rhs.signBit) {
            mkFpNan(lhs.sort)
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

    private fun KContext.fpMul(rm: KFpRoundingMode, lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> = when {
        // RNE is JVM default rounding mode
        rm == KFpRoundingMode.RoundNearestTiesToEven && lhs is KFp32Value -> {
            mkFp(lhs.value * (rhs as KFp32Value).value, lhs.sort)
        }

        rm == KFpRoundingMode.RoundNearestTiesToEven && lhs is KFp64Value -> {
            mkFp(lhs.value * (rhs as KFp64Value).value, lhs.sort)
        }

        lhs.isNan() || rhs.isNan() -> mkFpNan(lhs.sort)

        lhs.isInfinity() && lhs.isPositive() -> if (rhs.isZero()) {
            mkFpNan(lhs.sort)
        } else {
            mkFpInf(rhs.signBit, lhs.sort)
        }

        rhs.isInfinity() && rhs.isPositive() -> if (lhs.isZero()) {
            mkFpNan(lhs.sort)
        } else {
            mkFpInf(lhs.signBit, lhs.sort)
        }

        lhs.isInfinity() && lhs.isNegative() -> if (rhs.isZero()) {
            mkFpNan(lhs.sort)
        } else {
            mkFpInf(!rhs.signBit, lhs.sort)
        }

        rhs.isInfinity() && rhs.isNegative() -> if (lhs.isZero()) {
            mkFpNan(lhs.sort)
        } else {
            mkFpInf(!lhs.signBit, lhs.sort)
        }

        lhs.isZero() || rhs.isZero() -> {
            mkFpZero(sort = lhs.sort, signBit = lhs.signBit != rhs.signBit)
        }

        else -> fpUnpackAndMul(rm, lhs, rhs)
    }

    private fun KContext.fpDiv(rm: KFpRoundingMode, lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> = when {
        // RNE is JVM default rounding mode
        rm == KFpRoundingMode.RoundNearestTiesToEven && lhs is KFp32Value -> {
            mkFp(lhs.value / (rhs as KFp32Value).value, lhs.sort)
        }

        rm == KFpRoundingMode.RoundNearestTiesToEven && lhs is KFp64Value -> {
            mkFp(lhs.value / (rhs as KFp64Value).value, lhs.sort)
        }

        lhs.isNan() || rhs.isNan() -> mkFpNan(lhs.sort)

        lhs.isInfinity() && lhs.isPositive() -> if (rhs.isInfinity()) {
            mkFpNan(lhs.sort)
        } else {
            mkFpInf(signBit = rhs.signBit, sort = lhs.sort)
        }

        rhs.isInfinity() && rhs.isPositive() -> if (lhs.isInfinity()) {
            mkFpNan(lhs.sort)
        } else {
            mkFpZero(signBit = lhs.signBit, sort = lhs.sort)
        }

        lhs.isInfinity() && lhs.isNegative() -> if (rhs.isInfinity()) {
            mkFpNan(lhs.sort)
        } else {
            mkFpInf(signBit = !rhs.signBit, lhs.sort)
        }

        rhs.isInfinity() && rhs.isNegative() -> if (lhs.isInfinity()) {
            mkFpNan(lhs.sort)
        } else {
            mkFpZero(signBit = !lhs.signBit, sort = lhs.sort)
        }

        rhs.isZero() -> if (lhs.isZero()) {
            mkFpNan(lhs.sort)
        } else {
            mkFpInf(signBit = lhs.signBit != rhs.signBit, sort = lhs.sort)
        }

        lhs.isZero() -> mkFpZero(signBit = lhs.signBit != rhs.signBit, sort = lhs.sort)

        else -> fpUnpackAndDiv(rm, lhs, rhs)
    }

    private fun KContext.fpSqrt(rm: KFpRoundingMode, value: KFpValue<*>): KFpValue<*> = when {
        // RNE is JVM default rounding mode
        rm == KFpRoundingMode.RoundNearestTiesToEven && value is KFp32Value -> {
            mkFp(sqrt(value.value), value.sort)
        }

        rm == KFpRoundingMode.RoundNearestTiesToEven && value is KFp64Value -> {
            mkFp(sqrt(value.value), value.sort)
        }

        value.isNan() -> value
        value.isInfinity() && value.isPositive() -> value
        value.isZero() -> value
        value.isNegative() -> mkFpNan(value.sort)
        else -> fpUnpackAndSqrt(rm, value)
    }

    private fun KContext.fpRoundToIntegral(rm: KFpRoundingMode, value: KFpValue<*>): KFpValue<*> = when {
        // RNE is JVM default rounding mode
        rm == KFpRoundingMode.RoundNearestTiesToEven && value is KFp32Value -> {
            mkFp(round(value.value), value.sort)
        }

        rm == KFpRoundingMode.RoundNearestTiesToEven && value is KFp64Value -> {
            mkFp(round(value.value), value.sort)
        }

        value.isNan() -> value
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
            when {
                tie && rm == KFpRoundingMode.RoundNearestTiesToEven -> mkFpZero(value.signBit, value.sort)
                tie && rm == KFpRoundingMode.RoundNearestTiesToAway -> mkFpOne(value.sort, value.signBit)
                exponent < (-BigInteger.ONE) -> mkFpZero(value.signBit, value.sort)
                else -> mkFpOne(value.sort, value.signBit)
            }
        }
    }

    private fun <T : KFpSort> KContext.fpToFp(rm: KFpRoundingMode, value: KFpValue<*>, toFpSort: T): KFpValue<T> =
        when {
            value.isNan() -> mkFpNan(toFpSort)
            value.isInfinity() -> mkFpInf(value.signBit, toFpSort)
            value.isZero() -> mkFpZero(value.signBit, toFpSort)
            value.sort == toFpSort -> value.uncheckedCast()
            else -> fpUnpackAndToFp(rm, value, toFpSort)
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

    private fun KContext.fpAddUnpacked(rm: KFpRoundingMode, lhs: UnpackedFp, rhs: UnpackedFp): KFpValue<*> {
        // lhs.exponent >= rhs.exponent => expDelta >= 0
        var expDelta = (lhs.unbiasedExponent - rhs.unbiasedExponent)
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

        val sign = ((!lhs.sign && rhs.sign && resIsNeg)
                || (lhs.sign && !rhs.sign && !resIsNeg)
                || (lhs.sign && rhs.sign))
        val unpackedResult = UnpackedFp(
            lhs.exponentSize, lhs.significandSize, sign, lhs.unbiasedExponent, resSignificandValue
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
        val (shiftedRhs, stickyRem) = rhsSignificand.divideAndRemainder(powerOfTwo(expDelta))

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

        val multipliedSignificand = lhs.significand * rhs.significand

        // Remove the extra bits, keeping a sticky bit.
        var (normalizedSignificand, stickyRem) = if (lhs.significandSize >= 4u){
            val resultWithReminder = multipliedSignificand.divideAndRemainder(
                powerOfTwo(lhs.significandSize - 4u)
            )
            resultWithReminder[0] to resultWithReminder[1]
        } else {
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

    private fun KContext.fpDivUnpacked(rm: KFpRoundingMode, lhs: UnpackedFp, rhs: UnpackedFp): KFpValue<*> {
        val resultSign = lhs.sign xor rhs.sign
        val resultExponent = lhs.unbiasedExponent - rhs.unbiasedExponent

        val extraBits = lhs.significandSize + 2u
        val lhsSignificandWithExtraBits = lhs.significand.mul2k(lhs.significandSize + extraBits)
        val divisionResultSignificand = lhsSignificandWithExtraBits.divide(rhs.significand)

        // Remove the extra bits, keeping a sticky bit.
        var (normalizedResultSignificand, stickyRem) = divisionResultSignificand.divideAndRemainder(
            powerOfTwo(extraBits - 2u)
        )
        if (!stickyRem.isZero() && normalizedResultSignificand.isEven()) {
            normalizedResultSignificand++
        }

        val unpackedResult = UnpackedFp(
            lhs.exponentSize, lhs.significandSize, resultSign, resultExponent, normalizedResultSignificand
        )

        return fpRound(rm, unpackedResult)
    }

    private fun KContext.fpSqrtUnpacked(rm: KFpRoundingMode, value: UnpackedFp): KFpValue<*> {
        val extraBits = if (value.unbiasedExponent.isEven()) 7u else 6u
        val extendedSignificand = value.significand.mul2k(value.significandSize + extraBits)
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

        var resultExponent = value.unbiasedExponent.shiftRight(1)
        if (value.unbiasedExponent.isEven()) {
            resultExponent--
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

        var resultSignificand = fpRoundSignificandToIntegral(value, shift, rm)
        var resultExponent = value.unbiasedExponent

        // re-normalize
        val maxValue = powerOfTwo(value.significandSize)
        while (resultSignificand >= maxValue) {
            resultSignificand = resultSignificand.div2k(1u)
            resultExponent++
        }

        return mkRoundedValue(
            rm, resultExponent, resultSignificand, value.sign, value.exponentSize, value.significandSize
        )
    }

    private fun fpRoundSignificandToIntegral(
        value: UnpackedFp,
        shift: BigInteger,
        rm: KFpRoundingMode
    ): BigInteger {
        var (div, rem) = value.significand.divideAndRemainder(powerOfTwo(shift))

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

    private fun KContext.fpToFpUnpacked(
        rm: KFpRoundingMode,
        value: UnpackedFp,
        toExponentSize: UInt,
        toSignificandSize: UInt
    ): KFpValue<*> {

        var significandSizeDelta = toSignificandSize.toInt() - value.significandSize.toInt() + 3 // plus rounding bits

        val resultSignificand = when {
            significandSizeDelta > 0 -> value.significand.mul2k(significandSizeDelta.toUInt())

            significandSizeDelta < 0 -> {
                var sticky = false
                var significand = value.significand

                while (significandSizeDelta < 0) {
                    sticky = sticky || !significand.isEven()
                    significand = significand.div2k(1u)
                    significandSizeDelta++
                }

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

    private fun KContext.fpRound(rm: KFpRoundingMode, value: UnpackedFp): KFpValue<*> {
        // Assumptions: significand is of the form f[-1:0] . f[1:sbits-1] [round,extra,sticky],
        // i.e., it has 2 + (sbits-1) + 3 = sbits + 4 bits.

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

    private fun fpRoundExponent(value: UnpackedFp): Pair<BigInteger, BigInteger> {
        val eMin = fpMinExponentValue(value.exponentSize)

        val sigWidth = value.significand.log2() + 1
        val lz = value.significandSize.toInt() + 4 - sigWidth
        val beta = value.unbiasedExponent - lz.toBigInteger() + BigInteger.ONE

        var sigma: BigInteger
        val exponent: BigInteger
        if (beta < eMin) {
            // denormal significand/TINY
            sigma = value.unbiasedExponent - eMin
            exponent = eMin
        } else {
            sigma = (lz - 1).toBigInteger()
            exponent = beta
        }

        val sigmaCap = (-(value.significandSize.toInt() + 2)).toBigInteger()
        if (sigma < sigmaCap) {
            sigma = sigmaCap
        }

        return exponent to sigma
    }

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
            val botBiasedExponent = bvZero(exponentSize)
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
            var (res, stickyRem) = significand.divideAndRemainder(powerOfTwo(-normalizationShiftSize))
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
        val inc = when (rm) {
            KFpRoundingMode.RoundNearestTiesToEven -> round && (last || sticky)
            KFpRoundingMode.RoundNearestTiesToAway -> round
            KFpRoundingMode.RoundTowardPositive -> (!sign && (round || sticky))
            KFpRoundingMode.RoundTowardNegative -> (sign && (round || sticky))
            KFpRoundingMode.RoundTowardZero -> false
        }

        if (inc) {
            result++
        }
        return result
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

        var significandValue = significand.bigIntValue().normalizeValue(sort.significandBits - 1u)
        val normalizedSignificand = if (significandValue == BigInteger.ZERO) {
            significandValue
        } else {
            val p = powerOfTwo(sort.significandBits - 1u)
            while (p > significandValue) {
                normalizedExponent--
                significandValue = significandValue.mul2k(1u)
            }
            significandValue
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

    private fun powerOfTwo(power: BigInteger): BigInteger {
        check(power.signum() >= 0) { "Negative power" }
        val intPower = power.intValueExact()
        val two = BigInteger.valueOf(2L)
        return two.pow(intPower)
    }

    private fun BigInteger.isEven(): Boolean = toInt() % 2 == 0
    private fun BigInteger.isZero(): Boolean = this == BigInteger.ZERO
    private fun BigInteger.log2(): Int = bitLength() - 1
    private fun BigInteger.mul2k(k: UInt): BigInteger = this * powerOfTwo(k)
    private fun BigInteger.mul2k(k: BigInteger): BigInteger = this * powerOfTwo(k)
    private fun BigInteger.div2k(k: UInt): BigInteger = this / powerOfTwo(k)
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
        val two = 2.toBigInteger()
        while (true) {
            val mid = (upper + lower).divide(two)
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
