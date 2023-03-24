package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.symfpu.CustomRounderInfo.Companion.defaultRounderInfo
import org.ksmt.symfpu.UnpackedFp.Companion.iteOp
import org.ksmt.symfpu.UnpackedFp.Companion.makeInf
import org.ksmt.symfpu.UnpackedFp.Companion.makeZero
import org.ksmt.utils.BvUtils.bvOne
import org.ksmt.utils.BvUtils.bvZero
import org.ksmt.utils.cast

private fun <Fp : KFpSort> KContext.rounderSpecialCases(
    format: Fp,
    roundingMode: KExpr<KFpRoundingModeSort>,
    roundedResult: UnpackedFp<Fp>,
    overflow: KExpr<KBoolSort>,
    underflow: KExpr<KBoolSort>,
    isZero: KExpr<KBoolSort>
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
    val max = makeMax(format, roundedResult.sign)
    val min = makeMin(format, roundedResult.sign)
    val zero = makeZero(format, roundedResult.sign)

    return iteOp(
        isZero, zero, iteOp(
            underflow, iteOp(returnZero, zero, min), iteOp(overflow, iteOp(returnInf, inf, max), roundedResult)
        )
    )
}

private fun KContext.roundingDecision(
    roundingMode: KExpr<KFpRoundingModeSort>,
    sign: KExpr<KBoolSort>,
    significandEven: KExpr<KBoolSort>,
    guardBit: KExpr<KBoolSort>,
    stickyBit: KExpr<KBoolSort>,
    knownRoundDown: KExpr<KBoolSort>
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
    val noSignificandOverflow: KExpr<KBoolSort>
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

fun KContext.decrement(expr: KExpr<KBvSort>): KExpr<KBvSort> = mkBvSubExpr(expr, bvOne(expr.sort.sizeBits).cast())

// something wrong with. looks like shift misses by one
// - normal exponent
// - subnormals
fun <Fp : KFpSort> KContext.round(
    uf: UnpackedFp<KFpSort>,
    roundingMode: KExpr<KFpRoundingModeSort>,
    format: Fp,
    known: CustomRounderInfo = defaultRounderInfo()
): UnpackedFp<Fp> {
    val sigWidth = uf.normalizedSignificand.sort.sizeBits.toInt()
    val sig = mkBvOrExpr(uf.normalizedSignificand, leadingOne(sigWidth))

    val targetSignificandWidth = format.significandBits.toInt()
    check(sigWidth >= targetSignificandWidth + 2)

    val exp = uf.unbiasedExponent
    val expWidth = exp.sort.sizeBits.toInt()
    val targetExponentWidth = exponentWidth(format)
    check(expWidth >= targetExponentWidth)

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
    val subnormalAmount = expandingSubtract(
        minNormalExponent(format).matchWidthSigned(this, exp), exp
    )
    val extractedSignificandWidth = extractedSignificand.sort.sizeBits.toInt()
    val subnormalShiftPrepared = if (extractedSignificandWidth >= expWidth + 1) {
        subnormalAmount.matchWidthUnsigned(this, extractedSignificand)
    } else {
        mkBvExtractExpr(extractedSignificandWidth - 1, 0, subnormalAmount)
    }


    val subnormalMask = orderEncode(subnormalShiftPrepared)
    val subnormalStickyMask =
        mkBvLogicalShiftRightExpr(subnormalMask, bvOne(targetSignificandWidth.toUInt() + 1u).cast())


    val subnormalMaskedSignificand = mkBvAndExpr(extractedSignificand, mkBvNotExpr(subnormalMask))
    val subnormalMaskRemoved = mkBvAndExpr(extractedSignificand, subnormalMask)


    val subnormalGuardBit = !isAllZeros(mkBvAndExpr(subnormalMaskRemoved, mkBvNotExpr(subnormalStickyMask)))
    val subnormalStickyBit =
        mkOr(guardBit, stickyBit, !isAllZeros(mkBvAndExpr(subnormalMaskRemoved, subnormalStickyMask)))

    val subnormalIncrementAmount = mkBvAndExpr(
        mkBvShiftLeftExpr(subnormalMask, bvOne(targetSignificandWidth.toUInt() + 1u).cast()), mkBvNotExpr(subnormalMask)
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
    val normalRoundUpAmount = boolToBv(roundUp).matchWidthUnsigned(this, extractedSignificand)

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