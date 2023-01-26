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

    private fun fpMinExponentValue(exponent: UInt): BigInteger {
        val exponentBias = powerOfTwo(exponent - 1u) - BigInteger.ONE
        return (-exponentBias) + BigInteger.ONE
    }

    private fun fpMaxExponentValue(exponent: UInt): BigInteger {
        return powerOfTwo(exponent - 1u) - BigInteger.ONE
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

    private fun KFpValue<*>.unpackNormalValue(): UnpackedFp {
        val exponentBias = powerOfTwo(sort.exponentBits - 1u) - BigInteger.ONE
        val biasedExponentValue = biasedExponent.bigIntValue().normalizeValue(sort.exponentBits)
        val unbiasedExponent = biasedExponentValue - exponentBias

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

}
