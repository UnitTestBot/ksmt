package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort
import org.ksmt.utils.BvUtils.bvOne
import org.ksmt.utils.BvUtils.bvValue
import org.ksmt.utils.BvUtils.bvZero
import org.ksmt.utils.BvUtils.minus
import org.ksmt.utils.cast


fun KContext.bias(format: KFpSort): KBitVecValue<KBvSort> {
    val w = exponentWidth(format)
    val binaryValue = "0".repeat(w - format.exponentBits.toInt() + 1) + "1".repeat(format.exponentBits.toInt() - 1)
    val res = mkBv(binaryValue, w.toUInt())
    check(res.sort.sizeBits.toInt() == w)
    return res
}


fun KContext.minNormalExponent(format: KFpSort): KBitVecValue<KBvSort> {
    // -(bias - 1)
    val w = exponentWidth(format).toUInt()
    return (bvZero(w) - (bias(format) - bvOne(w))).cast()
}

fun KContext.maxNormalExponent(format: KFpSort): KBitVecValue<KBvSort> {
    return bias(format)
}

fun KContext.maxSubnormalExponent(format: KFpSort): KBitVecValue<KBvSort> {
    return (bvZero(exponentWidth(format).toUInt()) - bias(format)).cast()
}

fun KContext.minSubnormalExponent(format: KFpSort): KBitVecValue<KBvSort> {
    return (maxSubnormalExponent(format) - bvValue(
        exponentWidth(format).toUInt(),
        format.significandBits.toInt() - 2
    )).cast()
}

fun KContext.boolToBv(expr: KExpr<KBoolSort>) = mkIte(expr, bvOne(), bvZero())
fun KContext.bvToBool(expr: KExpr<KBvSort>) = mkIte(expr eq bvOne(), trueExpr, falseExpr)

fun KContext.bvOne() = mkBv(1, 1u)
fun KContext.bvZero() = mkBv(0, 1u)


fun <T : KBvSort, S : KBvSort, F : KBvSort> KContext.mkBvConcatExpr(
    arg0: KExpr<T>, arg1: KExpr<S>, arg2: KExpr<F>
) = mkBvConcatExpr(mkBvConcatExpr(arg0, arg1), arg2)

fun KContext.leadingOne(width: Int): KBitVecValue<KBvSort> {
    val binaryValue = "1" + "0".repeat(width - 1)
    return mkBv(binaryValue, width.toUInt())
}

fun <Fp : KFpSort> KContext.defaultSignificand(sort: Fp) = leadingOne(sort.significandBits.toInt())
fun <Fp : KFpSort> KContext.defaultExponent(sort: Fp): KBitVecValue<KBvSort> {
    return bvZero(exponentWidth(sort).toUInt()).cast()
}

fun significandWidth(format: KFpSort): Int = format.significandBits.toInt()

// Get the number of bits in the unpacked format corresponding to a
// given packed format.  These are the unpacked counter-parts of
//  format.exponentWidth() and format.significandWidth()

fun exponentWidth(format: KFpSort): Int {

    // This calculation is a little more complex than you might think...

    // Note that there is one more exponent above 0 than there is
    // below.  This is the opposite of 2's compliment but this is not
    // a problem because the highest packed exponent corresponds to
    // inf and NaN and is thus does not need to be represented in the
    // unpacked format.
    // However, we do need to increase it to allow subnormals (packed)
    // to be normalised.

    // The smallest exponent is:
    //  -2^(format.exponentWidth() - 1) - 2  -  (format.significandWidth() - 1)
    //
    // We need an unpacked exponent width u such that
    //  -2^(u-1) <= -2^(format.exponentWidth() - 1) - 2  -     (format.significandWidth() - 1)
    //           i.e.
    //   2^(u-1) >=  2^(format.exponentWidth() - 1)      +     (format.significandWidth() - 3)

    val formatExponentWidth = format.exponentBits.toInt()
    val formatSignificandWidth = format.significandBits.toInt()

    if (formatSignificandWidth <= 3) {
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
        bitsToRepresent(((1) shl (formatExponentWidth - 1)) + formatSignificandWidth - 3) + 1
    }
}

// The number of bits required to represent a number
//  == the position of the leading 0 + 1
//  == ceil(log_2(value + 1))
fun bitsToRepresent(value: Int): Int {
    var i = 0
    var working = value

    while (working != 0) {
        ++i
        working = working shr 1
    }

    return i
}

fun KContext.ones(width: UInt): KExpr<KBvSort> = mkBv(-1, width)

fun KContext.isAllZeros(expr: KExpr<KBvSort>) = expr eq bvZero(expr.sort.sizeBits).cast()
fun KContext.isAllOnes(expr: KExpr<KBvSort>): KExpr<KBoolSort> = expr eq mkBv(-1, expr.sort.sizeBits)


data class NormaliseShiftResult(
    val normalised: KExpr<KBvSort>, val shiftAmount: KExpr<KBvSort>
)

/* CLZ https://en.wikipedia.org/wiki/Find_first_set */
fun KContext.normaliseShift(input: KExpr<KBvSort>): NormaliseShiftResult {
    val inputWidth = input.sort.sizeBits
    val startingMask = previousPowerOfTwo(inputWidth)
    check(startingMask < inputWidth)


    // We need to shift the input to the left until the first bit is set
    var currentMantissa = input
    var shiftAmount: KExpr<KBvSort>? = null
    var curMaskLen = previousPowerOfTwo(inputWidth)
    while (curMaskLen > 0u) {
        val mask = mkBvConcatExpr(ones(curMaskLen), bvZero(inputWidth - curMaskLen))
        val shiftNeeded = isAllZeros(mkBvAndExpr(mask, currentMantissa))

        currentMantissa = mkIte(shiftNeeded,
            mkBvShiftLeftExpr(currentMantissa, curMaskLen.toInt().toBv(inputWidth)),
            currentMantissa)

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
    check(shiftAmountWidth.toInt() == widthBits || shiftAmountWidth.toInt() == widthBits - 1)

    return res

}

fun previousPowerOfTwo(x: UInt): UInt {
    check(x > 1u)

    var current = 1u
    var next = current * 2u

    while (next < x) {
        current = next
        next *= 2u
    }
    return current
}

fun KContext.max(op1: KExpr<KBvSort>, op2: KExpr<KBvSort>) = mkIte(mkBvSignedLessOrEqualExpr(op1, op2), op2, op1)

fun KContext.conditionalIncrement(cond: KExpr<KBoolSort>, bv: KExpr<KBvSort>): KExpr<KBvSort> {
    val inc = mkIte(cond, mkBv(1, bv.sort.sizeBits), mkBv(0, bv.sort.sizeBits))
    return mkBvAddExpr(bv, inc)
}

fun KContext.conditionalDecrement(cond: KExpr<KBoolSort>, bv: KExpr<KBvSort>): KExpr<KBvSort> {
    val inc = mkIte(cond, mkBv(1, bv.sort.sizeBits), mkBv(0, bv.sort.sizeBits))
    return mkBvSubExpr(bv, inc)
}

fun <Fp : KFpSort> KContext.makeMin(sort: Fp, sign: KExpr<KBoolSort>) = UnpackedFp(
    this, sort, sign, minSubnormalExponent(sort), leadingOne(sort.significandBits.toInt())
)

fun <Fp : KFpSort> KContext.makeMax(sort: Fp, sign: KExpr<KBoolSort>) = UnpackedFp(
    this, sort, sign, maxNormalExponent(sort), ones(sort.significandBits)
)

fun KContext.expandingSubtractUnsigned(op1: KExpr<KBvSort>, op2: KExpr<KBvSort>): KExpr<KBvSort> {
    check(op1.sort.sizeBits == op2.sort.sizeBits)
    val x = mkBvZeroExtensionExpr(1, op1)
    val y = mkBvZeroExtensionExpr(1, op2)
    return mkBvSubExpr(x, y)
}


fun KContext.expandingSubtractSigned(op1: KExpr<KBvSort>, op2: KExpr<KBvSort>): KExpr<KBvSort> {
    check(op1.sort.sizeBits == op2.sort.sizeBits)
    val x = mkBvSignExtensionExpr(1, op1)
    val y = mkBvSignExtensionExpr(1, op2)
    return mkBvSubExpr(x, y)
}