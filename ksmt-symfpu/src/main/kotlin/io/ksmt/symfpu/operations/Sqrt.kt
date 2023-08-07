package io.ksmt.symfpu.operations

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFpRoundingMode
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.symfpu.operations.UnpackedFp.Companion.iteOp
import io.ksmt.symfpu.operations.UnpackedFp.Companion.makeInf
import io.ksmt.symfpu.operations.UnpackedFp.Companion.makeNaN
import io.ksmt.symfpu.operations.UnpackedFp.Companion.makeZero

private fun KContext.fixedPointSqrt(
    x: KExpr<KBvSort>,
): ResultWithRemainderBit {
    val inputWidth = x.sort.sizeBits
    val outputWidth = inputWidth - 1u

    // To compare against, we need to pad x to 2/2p
    val xcomp = mkBvConcatExpr(x, mkBv(0, inputWidth - 2u))

    // Start at 1
    var working = mkBvConcatExpr(mkBv(1, 1u), mkBv(0, outputWidth - 1u))
    for (location in outputWidth - 1u downTo 1u) {
        val shift = mkBv(location.toInt() - 1, outputWidth)
        val candidate = mkBvOrExpr(working, mkBvShiftLeftExpr(mkBv(1, outputWidth), shift))
        val addBit = mkBvUnsignedLessOrEqualExpr(expandingMultiply(candidate, candidate), xcomp)
        working = mkBvOrExpr(
            working,
            mkBvShiftLeftExpr(mkBvZeroExtensionExpr(outputWidth.toInt() - 1, boolToBv(addBit)), shift)
        )
    }
    return ResultWithRemainderBit(working, expandingMultiply(working, working) neq xcomp)
}

private fun <Fp : KFpSort> KContext.arithmeticSqrt(
    uf: UnpackedFp<Fp>,
): UnpackedFp<KFpSort> {
    val exponent = uf.getExponent()
    val exponentWidth = uf.exponentWidth()
    val exponentEven = isAllZeros(mkBvAndExpr(exponent, mkBv(1, exponentWidth)))
    val exponentHalved = mkBvArithShiftRightExpr(exponent, mkBv(1, exponentWidth))

    val alignedSignificand = conditionalLeftShiftOne(
        !exponentEven,
        mkBvConcatExpr(mkBv(0, 1u), uf.getSignificand(), mkBv(0, 1u)),
    )

    val sqrtd = fixedPointSqrt(alignedSignificand)

    val finishedSignificand = mkBvConcatExpr(sqrtd.result, boolToBv(sqrtd.remainderBit))
    val extendedFormat = mkFpSort(uf.sort.exponentBits, uf.sort.significandBits + 2u)

    return UnpackedFp(
        this,
        extendedFormat,
        uf.sign,
        exponentHalved,
        finishedSignificand,
    )
}

internal fun <Fp : KFpSort> KContext.sqrt(
    roundingMode: KExpr<KFpRoundingModeSort>,
    uf: UnpackedFp<Fp>,
): UnpackedFp<Fp> {
    val sqrtResult = arithmeticSqrt(uf)

    // Exponent is divided by two, thus it can't overflow, underflow or generate a subnormal number.
    // The last one is quite subtle but you can show that the largest number generatable
    // by arithmeticSqrt is 111...111:0:1 with the last two as the guard and sticky bits.
    // Round up (when the sign is positive) and round down (when the sign is negative --
    // the result will be computed but then discarded) are the only cases when this can increment the significand.

    val rtp = roundingEq(roundingMode, KFpRoundingMode.RoundTowardPositive) and !sqrtResult.sign
    val rtn = roundingEq(roundingMode, KFpRoundingMode.RoundTowardNegative) and sqrtResult.sign
    val cri = CustomRounderInfo(
        noOverflow = trueExpr,
        noUnderflow = trueExpr,
        exact = falseExpr,
        subnormalExact = trueExpr,
        noSignificandOverflow = !(rtp or rtn)
    )
    val roundedSqrtResult = round(sqrtResult, roundingMode, uf.sort, cri)

    return addSqrtSpecialCases(uf, roundedSqrtResult.sign, roundedSqrtResult)
}

private fun <Fp : KFpSort> KContext.addSqrtSpecialCases(
    uf: UnpackedFp<Fp>,
    sign: KExpr<KBoolSort>,
    sqrtResult: UnpackedFp<Fp>,
): UnpackedFp<Fp> {
    val generateNaN = uf.sign and !uf.isZero
    val isNaN = uf.isNaN or generateNaN
    val isInf = uf.isInf and !uf.sign
    val isZero = uf.isZero

    return iteOp(
        isNaN,
        makeNaN(uf.sort),
        iteOp(
            isInf,
            makeInf(uf.sort, falseExpr),
            iteOp(
                isZero,
                makeZero(uf.sort, sign),
                sqrtResult
            )
        )
    )
}
