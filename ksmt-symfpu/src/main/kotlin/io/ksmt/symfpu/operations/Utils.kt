package io.ksmt.symfpu.operations

import io.ksmt.KContext
import io.ksmt.expr.KBitVecValue
import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpSort
import io.ksmt.utils.BvUtils.bvMaxValueSigned
import io.ksmt.utils.BvUtils.bvMaxValueUnsigned
import io.ksmt.utils.BvUtils.bvMinValueSigned
import io.ksmt.utils.BvUtils.bvOne
import io.ksmt.utils.BvUtils.bvValue
import io.ksmt.utils.BvUtils.bvZero
import io.ksmt.utils.BvUtils.isBvMaxValueUnsigned
import io.ksmt.utils.BvUtils.isBvZero
import io.ksmt.utils.BvUtils.minus
import io.ksmt.utils.BvUtils.zeroExtension
import io.ksmt.utils.uncheckedCast

fun KContext.bias(format: KFpSort): KBitVecValue<KBvSort> {
    val w = unpackedExponentWidth(format).toUInt()
    val eb = format.exponentBits
    return bvMaxValueSigned<KBvSort>(eb).zeroExtension(w - eb).uncheckedCast()
}

fun KContext.minNormalExponent(format: KFpSort): KBitVecValue<KBvSort> {
    // -(bias - 1)
    val w = unpackedExponentWidth(format).toUInt()
    return bvZero<KBvSort>(w) - (bias(format) - bvOne(w))
}

fun KContext.maxNormalExponent(format: KFpSort): KBitVecValue<KBvSort> {
    return bias(format)
}

fun KContext.maxSubnormalExponent(format: KFpSort): KBitVecValue<KBvSort> {
    return bvZero<KBvSort>(unpackedExponentWidth(format).toUInt()) - bias(format)
}

fun KContext.minSubnormalExponent(format: KFpSort): KBitVecValue<KBvSort> {
    return maxSubnormalExponent(format) - bvValue(
        unpackedExponentWidth(format).toUInt(),
        format.significandBits.toInt() - 2
    )
}

fun KContext.boolToBv(expr: KExpr<KBoolSort>) = mkIte(expr, bvOne(), bvZero())
fun KContext.bvToBool(expr: KExpr<KBvSort>) = mkIte(expr eq bvOne(), trueExpr, falseExpr)

fun KContext.bvOne() = mkBv(1, 1u)
fun KContext.bvZero() = mkBv(0, 1u)


fun <T : KBvSort, S : KBvSort, F : KBvSort> KContext.mkBvConcatExpr(
    arg0: KExpr<T>, arg1: KExpr<S>, arg2: KExpr<F>,
) = mkBvConcatExpr(mkBvConcatExpr(arg0, arg1), arg2)

fun <Fp : KFpSort> KContext.defaultSignificand(sort: Fp) = leadingOne(sort.significandBits.toInt())
fun <Fp : KFpSort> KContext.defaultExponent(sort: Fp): KBitVecValue<KBvSort> =
    bvZero(unpackedExponentWidth(sort).toUInt())

fun unpackedSignificandWidth(format: KFpSort): Int = format.significandBits.toInt()

/**
 * Get the number of bits in the unpacked format corresponding to a
 * given packed format.  These are the unpacked counter-parts of
 * format.exponentWidth() and format.significandWidth()
 */

fun unpackedExponentWidth(format: KFpSort): Int {

    /**
     * This calculation is a little more complex than you might think...
     *
     * Note that there is one more exponent above 0 than there is
     * below.  This is the opposite of 2's compliment but this is not
     * a problem because the highest packed exponent corresponds to
     * inf and NaN and is thus does not need to be represented in the
     * unpacked format.
     * However, we do need to increase it to allow subnormals (packed)
     * to be normalised.
     *
     *
     * The smallest exponent is:
     *    -2^(format.exponentWidth() - 1) - 2  -  (format.significandWidth() - 1)
     *    We need an unpacked exponent width u such that
     *     -2^(u-1) <= -2^(format.exponentWidth() - 1) - 2  -     (format.significandWidth() - 1)
     *              i.e.
     *      2^(u-1) >=  2^(format.exponentWidth() - 1)      +     (format.significandWidth() - 3)
     */


    val formatExponentWidth = format.exponentBits.toInt()
    val formatSignificandWidth = format.significandBits

    if (formatSignificandWidth <= 3u) {
        // Subnormals fit into the gap between minimum normal exponent and what is representable
        // using a signed number
        return formatExponentWidth
    }

    val bitsNeededForSubnormals = bitsToRepresent((format.significandBits - 3u).toInt())
    return if (bitsNeededForSubnormals < formatExponentWidth - 1) {
        // Significand is short compared to exponent range,
        // one extra bit should be sufficient
        formatExponentWidth + 1
    } else {
        // Significand is long compared to exponent range
        bitsToRepresent(((1) shl (formatExponentWidth - 1)) + (formatSignificandWidth - 3u).toInt()) + 1
    }
}

/**
 * The number of bits required to represent a number
 * == the position of the leading 0 + 1
 * == ceil(log_2(value + 1))
 */
fun bitsToRepresent(value: Int): Int {
    var i = 0
    var working = value

    while (working != 0) {
        ++i
        working = working shr 1
    }

    return i
}

fun KContext.ones(width: UInt): KBitVecValue<KBvSort> = bvMaxValueUnsigned(width)

fun KContext.leadingOne(width: Int): KBitVecValue<KBvSort> = bvMinValueSigned(width.toUInt())

fun KContext.isAllZeros(expr: KExpr<KBvSort>): KExpr<KBoolSort> =
    if (expr is KBitVecValue<KBvSort>) {
        mkBool(expr.isBvZero())
    } else {
        expr eq bvZero(expr.sort.sizeBits)
    }

fun KContext.isAllOnes(expr: KExpr<KBvSort>): KExpr<KBoolSort> =
    if (expr is KBitVecValue<KBvSort>) {
        mkBool(expr.isBvMaxValueUnsigned())
    } else {
        expr eq bvMaxValueUnsigned(expr.sort.sizeBits)
    }

data class NormaliseShiftResult(
    val normalised: KExpr<KBvSort>, val shiftAmount: KExpr<KBvSort>,
)

private fun previousPowerOfTwo(x: UInt): UInt {
    var result = x.takeHighestOneBit()
    // x is power of two
    if (result == x) {
        result /= 2u
    }
    return result
}

/* CLZ https://en.wikipedia.org/wiki/Find_first_set */
fun KContext.normaliseShift(input: KExpr<KBvSort>): NormaliseShiftResult {
    val inputWidth = input.sort.sizeBits
    val startingMask = previousPowerOfTwo(inputWidth)
    check(startingMask < inputWidth) { "Start has to be less than width" }


    // We need to shift the input to the left until the first bit is set
    var currentMantissa = input
    var shiftAmount: KExpr<KBvSort>? = null
    var curMaskLen = startingMask
    while (curMaskLen > 0u) {
        val mask = mkBvConcatExpr(ones(curMaskLen), bvZero(inputWidth - curMaskLen))
        val shiftNeeded = isAllZeros(mkBvAndExpr(mask, currentMantissa))

        currentMantissa = mkIte(
            shiftNeeded,
            mkBvShiftLeftExpr(currentMantissa, curMaskLen.toInt().toBv(inputWidth)),
            currentMantissa
        )

        shiftAmount = if (shiftAmount == null) {
            boolToBv(shiftNeeded)
        } else {
            mkBvConcatExpr(shiftAmount, boolToBv(shiftNeeded))
        }

        curMaskLen /= 2u
    }
    val res = NormaliseShiftResult(currentMantissa, shiftAmount!!)


    val shiftAmountWidth = res.shiftAmount.sort.sizeBits
    val widthBits = bitsToRepresent(inputWidth.toInt())
    check(shiftAmountWidth.toInt() == widthBits || shiftAmountWidth.toInt() == widthBits - 1) {
        "Shift amount width should be $widthBits or ${widthBits - 1} but was ${shiftAmountWidth.toInt()}"
    }

    return res

}

fun KContext.max(op1: KExpr<KBvSort>, op2: KExpr<KBvSort>) = mkIte(mkBvSignedLessOrEqualExpr(op1, op2), op2, op1)

fun KContext.conditionalNegate(cond: KExpr<KBoolSort>, bv: KExpr<KBvSort>): KExpr<KBvSort> {
    return mkIte(cond, mkBvNegationExpr(bv), bv)
}

fun KContext.conditionalIncrement(cond: KExpr<KBoolSort>, bv: KExpr<KBvSort>): KExpr<KBvSort> {
    val inc = mkIte(cond, mkBv(1, bv.sort.sizeBits), mkBv(0, bv.sort.sizeBits))
    return mkBvAddExpr(bv, inc)
}

fun KContext.conditionalDecrement(cond: KExpr<KBoolSort>, bv: KExpr<KBvSort>): KExpr<KBvSort> {
    val inc = mkIte(cond, mkBv(1, bv.sort.sizeBits), mkBv(0, bv.sort.sizeBits))
    return mkBvSubExpr(bv, inc)
}

fun <Fp : KFpSort> KContext.makeMinValue(sort: Fp, sign: KExpr<KBoolSort>) = UnpackedFp(
    this, sort, sign, minSubnormalExponent(sort), leadingOne(sort.significandBits.toInt())
)

fun <Fp : KFpSort> KContext.makeMaxValue(sort: Fp, sign: KExpr<KBoolSort>) = UnpackedFp(
    this, sort, sign, maxNormalExponent(sort), ones(sort.significandBits)
)

fun KContext.expandingSubtractUnsigned(op1: KExpr<KBvSort>, op2: KExpr<KBvSort>): KExpr<KBvSort> {
    check(op1.sort.sizeBits == op2.sort.sizeBits) { "Operands must be the same size" }
    val x = mkBvZeroExtensionExpr(1, op1)
    val y = mkBvZeroExtensionExpr(1, op2)
    return mkBvSubExpr(x, y)
}


fun KContext.expandingSubtractSigned(op1: KExpr<KBvSort>, op2: KExpr<KBvSort>): KExpr<KBvSort> {
    check(op1.sort.sizeBits == op2.sort.sizeBits) { "Operands must be the same size" }
    val x = mkBvSignExtensionExpr(1, op1)
    val y = mkBvSignExtensionExpr(1, op2)
    return mkBvSubExpr(x, y)
}
