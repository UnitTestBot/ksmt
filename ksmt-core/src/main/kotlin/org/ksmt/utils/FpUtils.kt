package org.ksmt.utils

import org.ksmt.KContext
import org.ksmt.expr.KBitVec1Value
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KFp32Value
import org.ksmt.expr.KFp64Value
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.expr.KFpValue
import org.ksmt.sort.KFpSort
import org.ksmt.utils.BvUtils.bitwiseAnd
import org.ksmt.utils.BvUtils.bitwiseNot
import org.ksmt.utils.BvUtils.bitwiseOr
import org.ksmt.utils.BvUtils.bvMaxValueSigned
import org.ksmt.utils.BvUtils.bvMaxValueUnsigned
import org.ksmt.utils.BvUtils.bvMinValueSigned
import org.ksmt.utils.BvUtils.bvOne
import org.ksmt.utils.BvUtils.bvZero
import org.ksmt.utils.BvUtils.concatBv
import org.ksmt.utils.BvUtils.extractBv
import org.ksmt.utils.BvUtils.minus
import org.ksmt.utils.BvUtils.plus
import org.ksmt.utils.BvUtils.shiftLeft
import org.ksmt.utils.BvUtils.shiftRightArith
import org.ksmt.utils.BvUtils.shiftRightLogical
import org.ksmt.utils.BvUtils.signExtension
import org.ksmt.utils.BvUtils.signedGreaterOrEqual
import org.ksmt.utils.BvUtils.signedLessOrEqual
import org.ksmt.utils.BvUtils.unaryMinus
import org.ksmt.utils.BvUtils.unsignedGreaterOrEqual
import org.ksmt.utils.BvUtils.unsignedLessOrEqual
import org.ksmt.utils.BvUtils.zeroExtension

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

    fun KContext.biasFpExponent(exponent: KBitVecValue<*>, exponentSize: UInt): KBitVecValue<*> =
        exponent + bvMaxValueSigned(exponentSize)

    fun KContext.unbiasFpExponent(exponent: KBitVecValue<*>, exponentSize: UInt): KBitVecValue<*> =
        exponent - bvMaxValueSigned(exponentSize)

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

    private fun KContext.fpAdd(rm: KFpRoundingMode, lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*> = when {
        lhs.isNan() || rhs.isNan() -> mkFpNan(lhs.sort)
        lhs.isInfinity() -> if (rhs.isInfinity() && lhs.signBit != rhs.signBit) {
            mkFpNan(lhs.sort)
        } else {
            mkFpInf(lhs.signBit, lhs.sort)
        }

        rhs.isInfinity() -> if (lhs.isInfinity() && lhs.signBit != rhs.signBit) {
            mkFpNan(lhs.sort)
        } else {
            mkFpInf(lhs.signBit, lhs.sort)
        }

        lhs.isZero() && rhs.isZero() -> {
            val bothNegative = lhs.isNegative() && rhs.isNegative()
            val roundToNegative = rm == KFpRoundingMode.RoundTowardNegative && (lhs.isNegative() || rhs.isNegative())
            if (bothNegative || roundToNegative) {
                mkFpZero(sort = lhs.sort, signBit = true)
            } else {
                mkFpZero(sort = lhs.sort, signBit = false)
            }
        }

        lhs.isZero() -> rhs
        rhs.isZero() -> lhs
        else -> fpAddUnpacked(rm, lhs.unpack(), rhs.unpack()).pack()
    }

    private fun KContext.fpAddUnpacked(
        rm: KFpRoundingMode,
        lhs: UnpackedFp,
        rhs: UnpackedFp
    ): UnpackedFp {
        val exponentCompare = fpCompareExponentForAdd(lhs.sort, lhs.unbiasedExponent, rhs.unbiasedExponent)
        val (additionResult, rounderInfo) = fpArithmeticAdd(rm, lhs, rhs, exponentCompare)
        return fpCustomRound(lhs.sort, rm, additionResult, rounderInfo)
    }

    private fun fpCustomRound(
        sort: KFpSort,
        rm: KFpRoundingMode,
        value: UnpackedFp,
        rounderInfo: CustomRounderInfo
    ): UnpackedFp {

    }

    private fun KContext.fpArithmeticAdd(
        rm: KFpRoundingMode,
        lhs: UnpackedFp,
        rhs: UnpackedFp,
        ec: ExponentCompareInfo
    ): Pair<UnpackedFp, CustomRounderInfo> {
        val effectiveAdd = (lhs.sign xor rhs.sign) xor true

        // Rounder flags
        val  noOverflow = !effectiveAdd
        val  noUnderflow = true
        val  subnormalExact = true
        val noSignificandOverflow =
            (effectiveAdd && ec.diffIsZero) || (!effectiveAdd && (ec.diffIsZero || ec.diffIsOne))
        val  stickyBitIsZero = ec.diffIsZero || ec.diffIsOne

        val leftLarger = ec.leftIsGreater && if (!ec.diffIsZero) {
            true
        } else {
            lhs.significand.unsignedGreaterOrEqual(rhs.significand)
        }

        val twoZeros =  bvZero(2u)
        val lsig = if (leftLarger) {
            lhs.significand
        } else {
            rhs.significand.zeroExtension(1u).let { concatBv(it, twoZeros) }
        }
        val ssig = if (leftLarger){
            rhs.significand
        } else {
            lhs.significand.zeroExtension(1u).let { concatBv(it, twoZeros) }
        }

        val resultSign = if (leftLarger) {
            lhs.sign
        } else {
            rhs.sign
        }

        val negatedSmaller = if (!effectiveAdd) {
            -ssig
        } else {
            ssig
        }

        val shiftAmount = ec.absoluteExponentDifference.zeroExtension(
            negatedSmaller.sort.sizeBits - ec.absoluteExponentDifference.sort.sizeBits
        )
        val shiftedSignExtendedResult = negatedSmaller.stickyRightShift(shiftAmount)
        val shiftedStickyBitX = negatedSmaller.rightShiftStickyBit(shiftAmount)

        val negatedAlignedSmaller = if (ec.diffIsGreaterThanPrecisionPlusOne){
            if (effectiveAdd) {
                bvZero(negatedSmaller.sort.sizeBits)
            } else {
                bvZero(negatedSmaller.sort.sizeBits).bitwiseNot()
            }
        } else {
            shiftedSignExtendedResult
        }
        val shiftedStickyBit = if (ec.diffIsGreaterThanPrecision) {
            bvOne(negatedSmaller.sort.sizeBits)
        } else {
            shiftedStickyBitX
        }

        val sum = lsig + negatedAlignedSmaller
        val sumWidth = sum.sort.sizeBits.toInt()
        val topBit = sum.extractBv(sumWidth - 1, sumWidth - 1).let { it as KBitVec1Value }.value
        val alignedBit = sum.extractBv(sumWidth - 2, sumWidth - 2).let { it as KBitVec1Value }.value
        val lowerBit = sum.extractBv(sumWidth - 3, sumWidth - 3).let { it as KBitVec1Value }.value

        val  overflow = topBit
        val  cancel = !topBit && !alignedBit
        val  minorCancel = cancel && lowerBit
        val  majorCancel = cancel && !lowerBit
        val  fullCancel = majorCancel && sum.isZero()

        val exact = cancel && (ec.diffIsZero || ec.diffIsOne)

        val oneSumWidth = bvOne(sum.sort.sizeBits)
        val alignedSumWithOverflow = if (overflow) {
            sum.shiftRightLogical(oneSumWidth)
        } else {
            sum
        }
        val alignedSum = if (minorCancel) {
            alignedSumWithOverflow.shiftLeft(oneSumWidth)
        } else {
            alignedSumWithOverflow
        }

        val exponentWidth = lhs.sort.exponentBits + 1u
        val exponentCorrectionTerm = when {
            minorCancel -> -bvOne(exponentWidth)
            overflow -> bvOne(exponentWidth)
            else -> bvZero(exponentWidth)
        }
        val correctedExponent = ec.maxExponent + exponentCorrectionTerm

        val stickyBit = when {
            stickyBitIsZero || majorCancel -> bvZero(alignedSum.sort.sizeBits)
            !overflow -> shiftedStickyBit
            else -> {
                val newBit = sum.extractBv(0, 0).zeroExtension(alignedSum.sort.sizeBits - 1u)
                shiftedStickyBit.bitwiseOr(newBit)
            }
        }

        val extendedFormat = mkFpSort(exponentWidth, lhs.sort.significandBits + 2u)
        val resultSumSignificand = alignedSum.bitwiseOr(stickyBit).extractBv((alignedSum.sort.sizeBits - 2u).toInt(), 0)
        val sumResult = UnpackedFp(extendedFormat, resultSign, correctedExponent, resultSumSignificand)

        val additionResult = if (fullCancel){
            val sign = rm == KFpRoundingMode.RoundTowardNegative
            mkFpZero(sign, extendedFormat).unpack()
        } else if (majorCancel){
            sumResult.normalizeUp(extendedFormat)
        } else {
            sumResult
        }
        val customRounderInfo = CustomRounderInfo(
            noOverflow, noUnderflow, exact, subnormalExact, noSignificandOverflow
        )
        return additionResult to customRounderInfo
    }

    private fun KContext.fpCompareExponentForAdd(
        sort: KFpSort,
        lhs: KBitVecValue<*>,
        rhs: KBitVecValue<*>
    ): ExponentCompareInfo {
        val leftIsGreater = rhs.signedLessOrEqual(lhs)

        val extendedLeft = lhs.signExtension(1u)
        val extendedRight = rhs.signExtension(1u)
        val maxExponent = if (leftIsGreater) extendedLeft else extendedRight

        val zero = bvZero(sort.exponentBits + 1u)
        val one = bvZero(sort.exponentBits + 1u)

        val exponentDifference = extendedLeft - extendedRight
        val absoluteExponentDifference = if (exponentDifference.signedLessOrEqual(zero)) {
             -exponentDifference
        } else {
            exponentDifference
        }

        val diffIsZero = absoluteExponentDifference == zero
        val diffIsOne = absoluteExponentDifference == one

        val significandSize = sort.significandBits.toInt()
        val diffIsGreaterThanPrecision = absoluteExponentDifference.signedGreaterOrEqual(significandSize + 1)
        val diffIsGreaterThanPrecisionPlusOne = absoluteExponentDifference.signedGreaterOrEqual(significandSize + 2)
        val diffIsTwoToPrecision = !diffIsZero && !diffIsOne && !diffIsGreaterThanPrecision

        return ExponentCompareInfo(
            leftIsGreater,
            maxExponent,
            absoluteExponentDifference,
            diffIsZero,
            diffIsOne,
            diffIsGreaterThanPrecision,
            diffIsTwoToPrecision,
            diffIsGreaterThanPrecisionPlusOne
        )
    }

    private fun UnpackedFp.normalizeUp(format: KFpSort): UnpackedFp {
        TODO()
    }

    private data class UnpackedFp(
        val sort: KFpSort,
        val sign: Boolean,
        val unbiasedExponent: KBitVecValue<*>,
        val significand: KBitVecValue<*>
    )

    private fun KContext.leadingOne(size: UInt): KBitVecValue<*> =
        bvMinValueSigned(size)

    private fun KFpValue<*>.unpack(): UnpackedFp {
        val significandWithLeadingZero = significand.zeroExtension(sort.significandBits - significand.sort.sizeBits)
        val leadingOne = ctx.leadingOne(sort.significandBits)
        val significandWithLeadingOne = significandWithLeadingZero.bitwiseOr(leadingOne)
        if (!isSubnormal()) {
            return UnpackedFp(
                sort = sort,
                sign = signBit,
                unbiasedExponent = ctx.unbiasFpExponent(biasedExponent, sort.exponentBits),
                significand = significandWithLeadingOne
            )
        } else {
            val subnormalBase = UnpackedFp(
                sort = sort,
                sign = signBit,
                unbiasedExponent = sort.minNormalExponent(),
                significand = significandWithLeadingZero
            )
            return subnormalBase.normalizeUp(sort)
        }
    }

    private fun UnpackedFp.pack(): KFpValue<*> = with(sort.ctx) {
        val inNormalRange = sort.minNormalExponent().signedLessOrEqual(unbiasedExponent)
        val inSubnormalRange = !inNormalRange

        val biasedExponent = biasFpExponent(unbiasedExponent, sort.exponentBits)


        TODO()
    }

    private fun KFpSort.minNormalExponent(): KBitVecValue<*> = with(ctx) {
        val bias = mkBv(exponentShiftSize(), exponentBits)
        -(bias - bvOne(exponentBits))
    }

    private data class ExponentCompareInfo(
        val leftIsGreater: Boolean,
        val maxExponent: KBitVecValue<*>,
        val absoluteExponentDifference: KBitVecValue<*>,
        val diffIsZero: Boolean,
        val diffIsOne: Boolean,
        val diffIsGreaterThanPrecision: Boolean,
        val diffIsTwoToPrecision: Boolean,
        val diffIsGreaterThanPrecisionPlusOne: Boolean
    )

    data class CustomRounderInfo(
        val noOverflow: Boolean,
        val noUnderflow: Boolean,
        val exact: Boolean,
        val subnormalExact: Boolean,
        val noSignificandOverflow: Boolean
    )

    private fun KBitVecValue<*>.stickyRightShift(shift: KBitVecValue<*>): KBitVecValue<*> = shiftRightArith(shift)

    private fun KBitVecValue<*>.rightShiftStickyBit(shift: KBitVecValue<*>): KBitVecValue<*> {
        val condition = shift.orderEncode().bitwiseAnd(this).isZero()
        return if (condition) ctx.bvZero(sort.sizeBits) else ctx.bvOne(sort.sizeBits)
    }

    private fun KBitVecValue<*>.orderEncode(): KBitVecValue<*> = with(ctx) {
        val width = sort.sizeBits
        val extended = zeroExtension(1u)
        val one = bvOne(width + 1u)
        (one.shiftLeft(extended) - one).extractBv(width.toInt() - 1, 0)
    }
}
