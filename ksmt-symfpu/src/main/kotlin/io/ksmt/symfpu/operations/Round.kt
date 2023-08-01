package io.ksmt.symfpu.operations

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFalse
import io.ksmt.expr.KFpRoundingMode
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.symfpu.operations.CustomRounderInfo.Companion.defaultRounderInfo
import io.ksmt.symfpu.operations.UnpackedFp.Companion.iteOp
import io.ksmt.symfpu.operations.UnpackedFp.Companion.makeInf
import io.ksmt.symfpu.operations.UnpackedFp.Companion.makeZero
import io.ksmt.utils.BvUtils.bvOne
import io.ksmt.utils.BvUtils.bvZero

@Suppress("LongParameterList")
private fun <Fp : KFpSort> KContext.rounderSpecialCases(
    format: Fp,
    roundingMode: KExpr<KFpRoundingModeSort>,
    roundedResult: UnpackedFp<Fp>,
    overflow: KExpr<KBoolSort>,
    underflow: KExpr<KBoolSort>,
    isZero: KExpr<KBoolSort>,
): UnpackedFp<Fp> {
    /*** Underflow and overflow ***/

    // On overflow either return inf or max
    val returnInf = mkOr(
        mkEq(roundingMode, mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven)),
        mkEq(roundingMode, mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToAway)),
        mkAnd(
            mkEq(roundingMode, mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardPositive)), !roundedResult.sign
        ),
        mkAnd(
            mkEq(roundingMode, mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardNegative)), roundedResult.sign
        )
    )

    // On underflow either return 0 or minimum subnormal
    val returnZero = mkOr(
        mkEq(roundingMode, mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven)),
        mkEq(roundingMode, mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToAway)),
        mkEq(roundingMode, mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardZero)),
        mkAnd(
            mkEq(roundingMode, mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardPositive)), roundedResult.sign
        ),
        mkAnd(
            mkEq(roundingMode, mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardNegative)), !roundedResult.sign
        )
    )


    /*** Reconstruct ***/
    val inf = makeInf(format, roundedResult.sign)
    val max = makeMaxValue(format, roundedResult.sign)
    val min = makeMinValue(format, roundedResult.sign)
    val zero = makeZero(format, roundedResult.sign)

    return iteOp(
        isZero, zero, iteOp(
        underflow, iteOp(returnZero, zero, min), iteOp(overflow, iteOp(returnInf, inf, max), roundedResult)
    )
    )
}

@Suppress("LongParameterList")
fun KContext.roundingDecision(
    roundingMode: KExpr<KFpRoundingModeSort>,
    sign: KExpr<KBoolSort>,
    significandEven: KExpr<KBoolSort>,
    guardBit: KExpr<KBoolSort>,
    stickyBit: KExpr<KBoolSort>,
    knownRoundDown: KExpr<KBoolSort>,
): KExpr<KBoolSort> {
    val roundUpRNE = mkAnd(
        roundingEq(roundingMode, KFpRoundingMode.RoundNearestTiesToEven), guardBit, mkOr(stickyBit, !significandEven)
    )
    val roundUpRNA = mkAnd(
        roundingEq(roundingMode, KFpRoundingMode.RoundNearestTiesToAway), guardBit
    )
    val roundUpRTP = mkAnd(
        roundingEq(roundingMode, KFpRoundingMode.RoundTowardPositive), !sign, mkOr(guardBit, stickyBit)
    )
    val roundUpRTN = mkAnd(
        roundingEq(roundingMode, KFpRoundingMode.RoundTowardNegative), sign, mkOr(guardBit, stickyBit)
    )
    val roundUpRTZ = mkAnd(
        roundingEq(roundingMode, KFpRoundingMode.RoundTowardZero), false.expr
    )

    return mkAnd(
        !knownRoundDown, mkOr(roundUpRNE, roundUpRNA, roundUpRTP, roundUpRTN, roundUpRTZ)
    )
}

fun KContext.roundingEq(roundingMode: KExpr<KFpRoundingModeSort>, other: KFpRoundingMode) =
    mkEq(roundingMode, mkFpRoundingModeExpr(other))


data class CustomRounderInfo(
    val noOverflow: KExpr<KBoolSort>,
    val noUnderflow: KExpr<KBoolSort>,
    val exact: KExpr<KBoolSort>,
    val subnormalExact: KExpr<KBoolSort>,
    val noSignificandOverflow: KExpr<KBoolSort>,
) {
    companion object {
        fun KContext.defaultRounderInfo() = CustomRounderInfo(
            noOverflow = falseExpr,
            noUnderflow = falseExpr,
            exact = falseExpr,
            subnormalExact = falseExpr,
            noSignificandOverflow = falseExpr
        )
    }
}

fun KContext.decrement(expr: KExpr<KBvSort>): KExpr<KBvSort> = mkBvSubExpr(expr, bvOne(expr.sort.sizeBits))
fun KContext.increment(expr: KExpr<KBvSort>): KExpr<KBvSort> = mkBvAddExpr(expr, bvOne(expr.sort.sizeBits))


@Suppress("LongMethod")
fun <Fp : KFpSort, S : KFpSort> KContext.round(
    uf: UnpackedFp<S>,
    roundingMode: KExpr<KFpRoundingModeSort>,
    format: Fp,
    known: CustomRounderInfo = defaultRounderInfo(),
): UnpackedFp<Fp> {
    val sigWidth = uf.normalizedSignificand.sort.sizeBits.toInt()
    val sig = mkBvOrExpr(uf.normalizedSignificand, leadingOne(sigWidth))

    val targetSignificandWidth = format.significandBits.toInt()
    check(sigWidth >= targetSignificandWidth + 2) { "Significand width is too small" }

    val exp = uf.unbiasedExponent
    val expWidth = exp.sort.sizeBits.toInt()
    val targetExponentWidth = unpackedExponentWidth(format)
    check(expWidth >= targetExponentWidth) { "Exponent width is too small" }

    /*** Early underflow and overflow detection ***/
    val exponentExtension = expWidth - targetExponentWidth
    val earlyOverflow = mkBvSignedGreaterExpr(exp, mkBvSignExtensionExpr(exponentExtension, maxNormalExponent(format)))
    val earlyUnderflow =
        mkBvSignedLessExpr(exp, mkBvSignExtensionExpr(exponentExtension, decrement(minSubnormalExponent(format))))


    val potentialLateOverflow = exp eq mkBvSignExtensionExpr(exponentExtension, maxNormalExponent(format))
    val potentialLateUnderflow =
        exp eq decrement(mkBvSignExtensionExpr(exponentExtension, minSubnormalExponent(format)))

    /*** Normal or subnormal rounding? ***/
    val normalRoundingRange =
        mkBvSignedGreaterOrEqualExpr(exp, mkBvSignExtensionExpr(exponentExtension, minNormalExponent(format)))
    val normalRounding = normalRoundingRange or known.subnormalExact


    /*** Round to correct significand. ***/
    val extractedSignificand =
        mkBvZeroExtensionExpr(1, mkBvExtractExpr(sigWidth - 1, sigWidth - targetSignificandWidth, sig))

    // guard bit is the bit after the target significand, sticky bit is the rest
    val guardBitPosition = sigWidth - (targetSignificandWidth + 1)
    val guardBit = isAllOnes(mkBvExtractExpr(guardBitPosition, guardBitPosition, sig))
    val stickyBit = !isAllZeros(mkBvExtractExpr(guardBitPosition - 1, 0, sig))


    // For subnormals, locating the guard and stick bits is a bit more involved
    val subnormalAmount = expandingSubtractUnsigned(
        minNormalExponent(format).matchWidthSigned(this, exp), exp
    )
    val extractedSignificandWidth = extractedSignificand.sort.sizeBits.toInt()
    val subnormalShiftPrepared = if (extractedSignificandWidth >= expWidth + 1) {
        subnormalAmount.matchWidthUnsigned(extractedSignificand)
    } else {
        mkBvExtractExpr(extractedSignificandWidth - 1, 0, subnormalAmount)
    }


    val subnormalMask = orderEncode(subnormalShiftPrepared)
    val subnormalStickyMask =
        mkBvLogicalShiftRightExpr(subnormalMask, bvOne(targetSignificandWidth.toUInt() + 1u))


    val subnormalMaskedSignificand = mkBvAndExpr(extractedSignificand, mkBvNotExpr(subnormalMask))
    val subnormalMaskRemoved = mkBvAndExpr(extractedSignificand, subnormalMask)


    val subnormalGuardBit = !isAllZeros(mkBvAndExpr(subnormalMaskRemoved, mkBvNotExpr(subnormalStickyMask)))
    val subnormalStickyBit =
        mkOr(guardBit, stickyBit, !isAllZeros(mkBvAndExpr(subnormalMaskRemoved, subnormalStickyMask)))

    val subnormalIncrementAmount = mkBvAndExpr(
        mkBvShiftLeftExpr(subnormalMask, bvOne(targetSignificandWidth.toUInt() + 1u)), mkBvNotExpr(subnormalMask)
    )


    // Have to choose the right one dependent on rounding mode
    val chosenGuardBit = mkIte(normalRounding, guardBit, subnormalGuardBit)
    val chosenStickyBit = mkIte(normalRounding, stickyBit, subnormalStickyBit)

    val significandEven = mkIte(
        normalRounding,
        isAllZeros(mkBvExtractExpr(0, 0, extractedSignificand)),
        isAllZeros(mkBvAndExpr(extractedSignificand, subnormalIncrementAmount))
    )

    val roundUp = roundingDecision(
        roundingMode,
        uf.sign,
        significandEven,
        chosenGuardBit,
        chosenStickyBit,
        known.exact or (known.subnormalExact and !normalRoundingRange)
    )


    val leadingOne = leadingOne(targetSignificandWidth)
    val normalRoundUpAmount = boolToBv(roundUp).matchWidthUnsigned(extractedSignificand)

    val subnormalRoundUpMask = mkBvArithShiftRightExpr(
        mkBvConcatExpr(
            boolToBv(roundUp), bvZero(targetSignificandWidth.toUInt())
        ), mkBv(targetSignificandWidth, targetSignificandWidth.toUInt() + 1u)
    )
    val subnormalRoundUpAmount = mkBvAndExpr(subnormalRoundUpMask, subnormalIncrementAmount)


    val rawRoundedSignificand = mkBvAddExpr(
        mkIte(normalRounding, extractedSignificand, subnormalMaskedSignificand),
        mkIte(normalRounding, normalRoundUpAmount, subnormalRoundUpAmount)
    )
    val significandOverflow =
        isAllOnes(mkBvExtractExpr(targetSignificandWidth, targetSignificandWidth, rawRoundedSignificand))
    val extractedRoundedSignificand = mkBvExtractExpr(targetSignificandWidth - 1, 0, rawRoundedSignificand)
    val roundedSignificand = mkBvOrExpr(extractedRoundedSignificand, leadingOne)

    val extendedExponent = mkBvSignExtensionExpr(1, exp)
    val incrementExponentNeeded = (roundUp and significandOverflow)
    val incrementExponent = !(known.noSignificandOverflow) and incrementExponentNeeded

    val correctedExponent = conditionalIncrement(incrementExponent, extendedExponent)

    val maxNormal = maxNormalExponent(format).matchWidthSigned(this, correctedExponent)
    val minSubnormal = minSubnormalExponent(format).matchWidthSigned(this, correctedExponent)

    val correctedExponentInRange = collar(correctedExponent, minSubnormal, maxNormal)

    val roundedExponent = mkBvExtractExpr(
        targetExponentWidth - 1, 0, correctedExponentInRange
    )


    val computedOverflow = potentialLateOverflow and incrementExponentNeeded
    val computedUnderflow = potentialLateUnderflow and !incrementExponentNeeded

    val lateOverflow = !earlyOverflow and computedOverflow
    val lateUnderflow = !earlyUnderflow and computedUnderflow

    val overflow = !(known.noOverflow) and (lateOverflow or earlyOverflow)
    val underflow = !(known.noUnderflow) and (lateUnderflow or earlyUnderflow)

    val roundedResult = UnpackedFp(
        this, format, uf.sign, roundedExponent, roundedSignificand
    )
    return rounderSpecialCases(
        format, roundingMode, roundedResult, overflow, underflow, uf.isZero
    )
}

fun KContext.collar(op: KExpr<KBvSort>, lower: KExpr<KBvSort>, upper: KExpr<KBvSort>): KExpr<KBvSort> {
    return mkIte(
        mkBvSignedLessExpr(op, lower), lower, mkIte(
        mkBvSignedLessExpr(upper, op), upper, op
    )
    )
}


data class SignificandRounderResult(val significand: KExpr<KBvSort>, val incrementExponent: KExpr<KBoolSort>)

@Suppress("LongParameterList")
fun KContext.variablePositionRound(
    roundingMode: KExpr<KFpRoundingModeSort>,
    sign: KExpr<KBoolSort>,
    significand: KExpr<KBvSort>,
    roundPosition: KExpr<KBvSort>,
    knownLeadingOne: KFalse,
    knownRoundDown: KExpr<KBoolSort>,
): SignificandRounderResult {


    val sigWidth = significand.sort.sizeBits

    val expandedSignificand = mkBvConcatExpr(mkBv(0, 2u), significand, mkBv(0, 2u))
    val exsigWidth = expandedSignificand.sort.sizeBits

    val incrementLocation = mkBvShiftLeftExpr(
        mkBv(1 shl 2, exsigWidth),
        roundPosition.matchWidthUnsigned(expandedSignificand)
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

@Suppress("LongParameterList")
fun KContext.fixedPositionRound(
    roundingMode: KExpr<KFpRoundingModeSort>,
    sign: KExpr<KBoolSort>,
    significand: KExpr<KBvSort>,
    targetWidth: Int,
    knownLeadingOne: KFalse,
    knownRoundDown: KExpr<KBoolSort>,
): SignificandRounderResult {
    val sigWidth = significand.sort.sizeBits.toInt()
    check(sigWidth >= targetWidth + 2) {
        "Significand width ($sigWidth) must be at least target width + 2 = ${targetWidth + 2}"
    }
    // Extract
    val extractedSignificand =
        mkBvZeroExtensionExpr(1, mkBvExtractExpr(sigWidth - 1, sigWidth - targetWidth, significand))


    val significandEven = isAllZeros(mkBvExtractExpr(0, 0, extractedSignificand))
    // Normal guard and sticky bits
    val guardBitPosition = sigWidth - (targetWidth + 1)
    val guardBit = isAllOnes(mkBvExtractExpr(guardBitPosition, guardBitPosition, significand))


    val stickyBit = !isAllZeros(mkBvExtractExpr(guardBitPosition - 1, 0, significand))
    // Rounding decision
    val roundUp = roundingDecision(roundingMode, sign, significandEven, guardBit, stickyBit, knownRoundDown)

    // Conditional increment
    val roundedSignificand = conditionalIncrement(roundUp, extractedSignificand)
    val overflowBit = mkBvAndExpr(mkBvExtractExpr(targetWidth, targetWidth, roundedSignificand), boolToBv(roundUp))
    val carryUpMask = mkBvConcatExpr(
        mkBvOrExpr(overflowBit, boolToBv(knownLeadingOne)),
        mkBv(0, targetWidth.toUInt() - 1u)
    )

    return SignificandRounderResult(
        mkBvOrExpr(mkBvExtractExpr(targetWidth - 1, 0, roundedSignificand), carryUpMask),
        isAllOnes(overflowBit)
    )
}
