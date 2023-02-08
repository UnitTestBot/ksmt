package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.expr.KBvConcatExpr
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort


fun KContext.bias(format: KFpSort): KExpr<KBvSort> {
    val w = exponentWidth(format)
    val res = mkBvConcatExpr(zeros(w.toUInt() - format.exponentBits + 1u), ones(format.exponentBits - 1u))
    check(res.sort.sizeBits.toInt() == w)
    return res
}


fun KContext.minNormalExponent(format: KFpSort): KExpr<KBvSort> {
    // -(bias - 1)
    return mkBvNegationExpr(mkBvSubExpr(bias(format), one(exponentWidth(format).toUInt())))
}

fun KContext.maxNormalExponent(format: KFpSort): KExpr<KBvSort> {
    return bias(format)
}

fun KContext.maxSubnormalExponent(format: KFpSort): KExpr<KBvSort> {
    return mkBvNegationExpr(bias(format))
}

fun KContext.minSubnormalExponent(format: KFpSort): KExpr<KBvSort> {
    return mkBvSubExpr(
        maxSubnormalExponent(format),
        mkBv(format.significandBits.toInt() - 2, exponentWidth(format).toUInt())
    )
}

fun KContext.boolToBv(expr: KExpr<KBoolSort>) = mkIte(expr, bvOne(), bvZero())
fun KContext.bvToBool(expr: KExpr<KBvSort>) = mkIte(expr eq bvOne(), trueExpr, falseExpr)

fun KContext.bvOne() = mkBv(1, 1u)
fun KContext.bvZero() = mkBv(0, 1u)


fun <T : KBvSort, S : KBvSort, F : KBvSort> KContext.mkBvConcatExpr(
    arg0: KExpr<T>, arg1: KExpr<S>, arg2: KExpr<F>
): KBvConcatExpr = mkBvConcatExpr(mkBvConcatExpr(arg0, arg1), arg2)

fun KContext.leadingOne(width: Int): KExpr<KBvSort> = mkBvConcatExpr(bvOne(), mkBv(0, (width - 1).toUInt()))
fun <Fp : KFpSort> KContext.defaultSignificand(sort: Fp) = leadingOne(sort.significandBits.toInt())
fun <Fp : KFpSort> KContext.defaultExponent(sort: Fp): KExpr<KBvSort> {
    return zeros(exponentWidth(sort).toUInt())
}

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

fun KContext.ones(width: UInt) = mkBv(-1, width)
fun KContext.one(width: UInt) = mkBv(1, width)
fun KContext.zero(width: Int) = mkBv(0, width.toUInt())
fun KContext.zeros(width: UInt) = mkBv(0, width)

fun KContext.isAllZeros(expr: KExpr<KBvSort>) = expr eq zeros(expr.sort.sizeBits)
fun KContext.isAllOnes(expr: KExpr<KBvSort>): KExpr<KBoolSort> = expr eq mkBv(-1, expr.sort.sizeBits)


data class NormaliseShiftResult(
    val normalised: KExpr<KBvSort>,
    val shiftAmount: KExpr<KBvSort>,
    val isZero: KExpr<KBoolSort>
)

fun KContext.normaliseShift(input: KExpr<KBvSort>): NormaliseShiftResult {
    val width = input.sort.sizeBits
    val startingMask = previousPowerOfTwo(width)
    check(startingMask < width)

    // Catch the zero case
    val zeroCase = isAllZeros(input)
    var working = input
    var shiftAmount: KExpr<KBvSort>? = null
    var deactivateShifts: KExpr<KBoolSort> = zeroCase
    // We need to shift the input to the right until the first bit is set
    // We need to shift the input to the right by i bits
    var i = startingMask
    check(i > 0u)
    while (i > 0u) {
        deactivateShifts =
            deactivateShifts or isAllOnes(mkBvExtractExpr(width.toInt() - 1, width.toInt() - 1, working))
        val mask = mkBvConcatExpr(ones(i), zeros(width - i))
        val shiftNeeded = !deactivateShifts and isAllZeros(mkBvAndExpr(mask, working))

        // Modular is safe because of the mask comparison
        working = mkIte(shiftNeeded, mkBvShiftLeftExpr(working, i.toInt().toBv(width)), working)
        shiftAmount = if (shiftAmount == null) {
            boolToBv(shiftNeeded)
        } else {
            mkBvConcatExpr(shiftAmount, boolToBv(shiftNeeded))
        }

        i /= 2u
    }
    val res = NormaliseShiftResult(working, shiftAmount!!, zeroCase)
    val shiftAmountWidth = res.shiftAmount.sort.sizeBits
    val widthBits = bitsToRepresent(width.toInt())
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

fun KContext.max(op1: KExpr<KBvSort>, op2: KExpr<KBvSort>) =
    mkIte(mkBvSignedLessOrEqualExpr(op1, op2), op2, op1)

fun KContext.conditionalIncrement(cond: KExpr<KBoolSort>, bv: KExpr<KBvSort>): KExpr<KBvSort> {
    val one = one(bv.sort.sizeBits)
    return mkIte(cond, mkBvAddExpr(bv, one), bv)
}

fun <Fp : KFpSort> KContext.makeMin(sort: Fp, sign: KExpr<KBoolSort>) = UnpackedFp(
    this, sort, sign, minSubnormalExponent(sort),
    leadingOne(sort.significandBits.toInt())
)

fun <Fp : KFpSort> KContext.makeMax(sort: Fp, sign: KExpr<KBoolSort>) = UnpackedFp(
    this, sort, sign, maxNormalExponent(sort),
    ones(sort.significandBits)
)