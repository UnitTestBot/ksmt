package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.symfpu.UnpackedFp.Companion.iteOp
import org.ksmt.symfpu.UnpackedFp.Companion.makeInf
import org.ksmt.symfpu.UnpackedFp.Companion.makeNaN
import org.ksmt.symfpu.UnpackedFp.Companion.makeZero
import org.ksmt.utils.BvUtils.bvZero


fun <T : KFpSort> KContext.divide(
    left: UnpackedFp<T>,
    right: UnpackedFp<T>,
    roundingMode: KExpr<KFpRoundingModeSort>,
): UnpackedFp<T> {
    val divideResult = arithmeticDivide(left, right)
    val roundedDivideResult = round(divideResult, roundingMode, left.sort)
    return addDivideSpecialCases(left, right, roundedDivideResult.sign, roundedDivideResult)
}

fun <T : KFpSort> KContext.addDivideSpecialCases(
    left: UnpackedFp<T>,
    right: UnpackedFp<T>,
    sign: KExpr<KBoolSort>,
    divideResult: UnpackedFp<T>
): UnpackedFp<T> {
    val format = left.sort
    val eitherArgumentNaN = left.isNaN or right.isNaN
    val generateNaN = (left.isInf and right.isInf) or (left.isZero and right.isZero)

    val isNaN = eitherArgumentNaN or generateNaN

    val isInf = (!left.isZero and right.isZero) or (left.isInf and !right.isInf)

    val isZero = (!left.isInf and right.isInf) or (left.isZero and !right.isZero)

    return iteOp(
        isNaN,
        makeNaN(format),
        iteOp(isInf, makeInf(format, sign), iteOp(isZero, makeZero(format, sign), divideResult))
    )
}

fun <T : KFpSort> KContext.arithmeticDivide(
    left: UnpackedFp<T>,
    right: UnpackedFp<T>
): UnpackedFp<KFpSort> {
    val format = left.sort
    val divideSign = left.sign xor right.sign

    val exponentDiff = expandingSubtractSigned(left.getExponent(), right.getExponent())


    val extendedNumerator = mkBvConcatExpr(left.getSignificand(), bvZero(2u))
    val extendedDenominator = mkBvConcatExpr(right.getSignificand(), bvZero(2u))

    val divided = fixedPointDivide(extendedNumerator, extendedDenominator)

    val resWidth = divided.result.sort.sizeBits.toInt()

    val topBit = mkBvExtractExpr(resWidth - 1, resWidth - 1, divided.result)

    val topBitSet = isAllOnes(topBit)


    val alignedExponent = conditionalDecrement(!topBitSet, exponentDiff)
    val alignedSignificand = conditionalLeftShiftOne(!topBitSet, divided.result)

    val finishedSignificand =
        mkBvOrExpr(alignedSignificand, mkBvZeroExtensionExpr(resWidth - 1, boolToBv(divided.remainderBit)))

    val extendedFormat = mkFpSort(format.exponentBits + 1u, format.significandBits + 2u)


    return UnpackedFp(this, extendedFormat, divideSign, alignedExponent, finishedSignificand)
}


data class ResultWithRemainderBit(val result: KExpr<KBvSort>, val remainderBit: KExpr<KBoolSort>)

private fun KContext.fixedPointDivide(x: KExpr<KBvSort>, y: KExpr<KBvSort>): ResultWithRemainderBit {
    val w = x.sort.sizeBits

    check(y.sort.sizeBits == w)
    // contract: first bits are ones

    val ex = mkBvConcatExpr(x, bvZero(w - 1u))
    val ey = mkBvZeroExtensionExpr(w.toInt() - 1, y)

    val div = mkBvUnsignedDivExpr(ex, ey)
    val rem = mkBvUnsignedRemExpr(ex, ey)

    return ResultWithRemainderBit(mkBvExtractExpr(w.toInt() - 1, 0, div), !isAllZeros(rem))
}
