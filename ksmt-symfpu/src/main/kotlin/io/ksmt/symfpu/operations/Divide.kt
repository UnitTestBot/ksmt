package io.ksmt.symfpu.operations

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.symfpu.operations.UnpackedFp.Companion.iteOp
import io.ksmt.symfpu.operations.UnpackedFp.Companion.makeInf
import io.ksmt.symfpu.operations.UnpackedFp.Companion.makeNaN
import io.ksmt.symfpu.operations.UnpackedFp.Companion.makeZero
import io.ksmt.utils.BvUtils.bvZero


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

    check(y.sort.sizeBits == w) { "Divisor must have same width as dividend" }
    // contract: first bits are ones

    val ex = mkBvConcatExpr(x, bvZero(w - 1u))
    val ey = mkBvZeroExtensionExpr(w.toInt() - 1, y)

    val div = mkBvUnsignedDivExpr(ex, ey)
    val rem = mkBvUnsignedRemExpr(ex, ey)

    return ResultWithRemainderBit(mkBvExtractExpr(w.toInt() - 1, 0, div), !isAllZeros(rem))
}
