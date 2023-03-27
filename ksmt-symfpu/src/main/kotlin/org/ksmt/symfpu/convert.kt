package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFalse
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.symfpu.UnpackedFp.Companion.iteOp
import org.ksmt.symfpu.UnpackedFp.Companion.makeZero


fun <T : KFpSort> roundToIntegral(
    roundingMode: KExpr<KFpRoundingModeSort>,
    input: UnpackedFp<T>,
): UnpackedFp<T> = with(input.ctx) {
    val exponent = input.getExponent()
    val exponentWidth = input.exponentWidth()

    val packedSigWidth = mkBv(input.sort.significandBits.toInt() - 1, exponentWidth)
    val unpackedSigWidth = mkBv(input.significandWidth().toInt(), exponentWidth)

    val isIntegral = mkBvSignedGreaterOrEqualExpr(exponent, packedSigWidth)
    val isSpecial = mkOr(input.isNaN, input.isInf, input.isZero)
    val isID = isIntegral or isSpecial

    val initialRoundingPoint = expandingSubtractSigned(packedSigWidth, exponent)
    val collaredRoundingPoint = collar(
        initialRoundingPoint, mkBv(0, exponentWidth + 1u),
        increment(mkBvSignExtensionExpr(1, unpackedSigWidth))
    )


    val significand = input.getSignificand()
    val significandWidth = input.significandWidth().toInt()
    val roundingPoint = if (significandWidth.toUInt() >= exponentWidth) {
        collaredRoundingPoint.matchWidthUnsigned(this, significand)
    } else {
        mkBvExtractExpr(significandWidth - 1, 0, collaredRoundingPoint)
    }


    val roundedResult = variablePositionRound(
        roundingMode, input.sign, significand, roundingPoint,
        mkFalse(), isID
    )

    val reconstructed = UnpackedFp(
        this,
        input.sort,
        input.sign,
        max(
            conditionalIncrement(roundedResult.incrementExponent, exponent),
            mkBv(0, exponentWidth)
        ),
        roundedResult.significand
    )


    return iteOp(
        isID,
        input,
        iteOp(
            isAllZeros(roundedResult.significand),
            makeZero(input.sort, input.sign),
            reconstructed
        )
    )
}


data class SignificandRounderResult(val significand: KExpr<KBvSort>, val incrementExponent: KExpr<KBoolSort>)


private fun KContext.variablePositionRound(
    roundingMode: KExpr<KFpRoundingModeSort>,
    sign: KExpr<KBoolSort>,
    significand: KExpr<KBvSort>,
    roundPosition: KExpr<KBvSort>,
    knownLeadingOne: KFalse,
    knownRoundDown: KExpr<KBoolSort>
): SignificandRounderResult {


    val sigWidth = significand.sort.sizeBits

    val expandedSignificand = mkBvConcatExpr(mkBv(0, 2u), significand, mkBv(0, 2u))
    val exsigWidth = expandedSignificand.sort.sizeBits

    val incrementLocation = mkBvShiftLeftExpr(
        mkBv(1 shl 2, exsigWidth),
        roundPosition.matchWidthUnsigned(this, expandedSignificand)
    )

    val guardLocation = mkBvLogicalShiftRightExpr(incrementLocation, mkBv(1, exsigWidth))
    val stickyLocation = decrement(guardLocation)

    val significandEven = isAllZeros(mkBvAndExpr(incrementLocation, expandedSignificand))
    val guardBit = !isAllZeros(mkBvAndExpr(guardLocation, expandedSignificand))
    val stickyBit = !isAllZeros(mkBvAndExpr(stickyLocation, expandedSignificand))

    val roundUp = roundingDecision(
        roundingMode, sign, significandEven,
        guardBit, stickyBit, knownRoundDown
    )

    val roundedSignificand = mkBvAddExpr(
        expandedSignificand,
        mkIte(
            roundUp,
            incrementLocation,
            mkBv(0, exsigWidth)
        )
    )

    val maskedRoundedSignificand = mkBvAndExpr(
        roundedSignificand,
        mkBvNotExpr(mkBvShiftLeftExpr(stickyLocation, mkBv(1, exsigWidth)))
    )

    val roundUpFromSticky = mkBvExtractExpr(exsigWidth.toInt() - 1, exsigWidth.toInt() - 1, roundedSignificand)

    val overflowBit = mkBvExtractExpr(exsigWidth.toInt() - 2, exsigWidth.toInt() - 2, roundedSignificand)
    val maskTrigger = mkBvAndExpr(
        mkBvOrExpr(roundUpFromSticky, overflowBit),
        boolToBv(roundUp)
    )
    val carryUpMask = mkBvConcatExpr(
        mkBvOrExpr(maskTrigger, boolToBv(knownLeadingOne)),
        mkBv(0, sigWidth - 1u)
    )

    return SignificandRounderResult(
        mkBvOrExpr(mkBvExtractExpr(sigWidth.toInt() + 1, 2, maskedRoundedSignificand), carryUpMask),
        isAllOnes(maskTrigger)
    )
}











