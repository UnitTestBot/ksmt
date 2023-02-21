package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.symfpu.UnpackedFp.Companion.iteOp
import org.ksmt.symfpu.UnpackedFp.Companion.makeInf
import org.ksmt.symfpu.UnpackedFp.Companion.makeZero
import org.ksmt.utils.BvUtils.bvOne
import org.ksmt.utils.BvUtils.bvZero
import org.ksmt.utils.cast

fun <Fp : KFpSort> KContext.round(
    uf: UnpackedFp<KFpSort>, roundingMode: KExpr<KFpRoundingModeSort>, format: Fp
): UnpackedFp<Fp> {
    val sigWidth = uf.significand.sort.sizeBits.toInt()
    val sig = mkBvOrExpr(uf.significand, leadingOne(sigWidth))

    val targetSignificandWidth = format.significandBits.toInt()
    check(sigWidth >= targetSignificandWidth + 2)

    val exp = uf.exponent
    val expWidth = exp.sort.sizeBits.toInt()
    val targetExponentWidth = exponentWidth(format)
    check(expWidth >= targetExponentWidth)

    /*** Early underflow and overflow detection ***/
    val exponentExtension = expWidth - targetExponentWidth
    val earlyOverflow = mkBvSignedGreaterExpr(exp, mkBvSignExtensionExpr(exponentExtension, maxNormalExponent(format)))
    val earlyUnderflow = mkBvSignedLessExpr(exp, mkBvSignExtensionExpr(exponentExtension, minSubnormalExponent(format)))


    /*** Normal or subnormal rounding? ***/
    val normalRounding =
        mkBvSignedGreaterOrEqualExpr(exp, mkBvSignExtensionExpr(exponentExtension, minNormalExponent(format)))


    /*** Round to correct significand. ***/
    val extractedSignificand = mkBvExtractExpr(sigWidth - 1, sigWidth - targetSignificandWidth, sig)

    // guard bit is the bit after the target significand, sticky bit is the rest
    val guardBitPosition = sigWidth - (targetSignificandWidth + 1)
    val guardBit = bvToBool(mkBvExtractExpr(guardBitPosition, guardBitPosition, sig))
    val stickyBit = !isAllZeros(mkBvExtractExpr(guardBitPosition - 1, 0, sig))


    // For subnormals, locating the guard and stick bits is a bit more involved
    val subnormalAmount = mkBvSubExpr(mkBvSignExtensionExpr(exponentExtension, maxSubnormalExponent(format)), exp)
    val extractedSignificandWidth = extractedSignificand.sort.sizeBits.toInt()
    val subnormalShiftPrepared = if (extractedSignificandWidth >= expWidth + 1) {
        mkBvZeroExtensionExpr(targetSignificandWidth - expWidth, subnormalAmount)
    } else {
        mkBvExtractExpr(extractedSignificandWidth - 1, 0, subnormalAmount)
    }
    val guardLocation = mkBvShiftLeftExpr(bvOne(targetSignificandWidth.toUInt()).cast(), subnormalShiftPrepared)
    val stickyMask = mkBvSubExpr(guardLocation, bvOne(targetSignificandWidth.toUInt()).cast())

    val subnormalGuardBit = !isAllZeros(mkBvAndExpr(extractedSignificand, guardLocation))
    val subnormalStickyBit = mkOr(guardBit, stickyBit, !isAllZeros(mkBvAndExpr(extractedSignificand, stickyMask)))

    // Can overflow but is handled
    val incrementedSignificand = mkBvAddExpr(extractedSignificand, bvOne(targetSignificandWidth.toUInt()).cast())
    val incrementedSignificandOverflow = isAllZeros(incrementedSignificand)

    check(incrementedSignificand.sort == leadingOne(format.significandBits.toInt()).sort)

    // Optimisation : use top bit of significand to increment the exponent
    val correctedIncrementedSignificand = mkIte(
        !incrementedSignificandOverflow, incrementedSignificand, leadingOne(format.significandBits.toInt())
    )


    val incrementAmount = mkBvShiftLeftExpr(guardLocation, bvOne(guardLocation.sort.sizeBits).cast())
    val mask = mkBvOrExpr(guardLocation, stickyMask)
    val maskedSignificand = mkBvAndExpr(extractedSignificand, mkBvNotExpr(mask))


    val subnormalIncrementedSignificand = mkBvAddExpr(maskedSignificand, incrementAmount)
    val subnormalIncrementedSignificandOverflow = isAllZeros(subnormalIncrementedSignificand)
    val subnormalCorrectedIncrementedSignificand = mkIte(
        !subnormalIncrementedSignificandOverflow,
        subnormalIncrementedSignificand,
        leadingOne(format.significandBits.toInt())
    )


    // Have to choose the right one dependent on rounding mode
    val chosenGuardBit = mkIte(normalRounding, guardBit, subnormalGuardBit)
    val chosenStickyBit = mkIte(normalRounding, stickyBit, subnormalStickyBit)

    val significandEven = mkIte(
        normalRounding,
        isAllZeros(mkBvExtractExpr(0, 0, extractedSignificand)),
        isAllZeros(mkBvAndExpr(extractedSignificand, incrementAmount))
    )
    val roundUp = roundingDecision(
        roundingMode, uf.sign, significandEven, chosenGuardBit, chosenStickyBit, false.expr
    )
    val roundedSignificand =
        mkIte(
            normalRounding, mkIte(
                !roundUp, extractedSignificand, correctedIncrementedSignificand
            ), mkIte(
                !roundUp, maskedSignificand, subnormalCorrectedIncrementedSignificand
            )
        )

    val extendedExponent = mkBvSignExtensionExpr(1, exp)
    val incrementExponent = mkAnd(
        mkIte(
            normalRounding, incrementedSignificandOverflow, subnormalIncrementedSignificandOverflow
        ), roundUp
    )
    val correctedExponent = conditionalIncrement(incrementExponent, extendedExponent)
    val roundedExponent = mkBvExtractExpr(
        targetExponentWidth - 1, 0, correctedExponent
    )
    val currentExponentWidth = correctedExponent.sort.sizeBits.toInt()


    /*** Finish ***/
    val computedOverflow = mkBvSignedGreaterExpr(
        correctedExponent, mkBvSignExtensionExpr(
            currentExponentWidth - targetExponentWidth, maxNormalExponent(format)
        )
    )
    val computedUnderflow = mkBvSignedLessExpr(
        correctedExponent, mkBvSignExtensionExpr(
            currentExponentWidth - targetExponentWidth, minSubnormalExponent(format)
        )
    )
    val overflow = mkOr(earlyOverflow, computedOverflow)
    val underflow = mkOr(earlyUnderflow, computedUnderflow)
    val roundedResult = UnpackedFp(
        this, format, uf.sign, roundedExponent, roundedSignificand
    )
    return rounderSpecialCases(
        format, roundingMode, roundedResult, overflow, underflow, uf.isZero
    )
}

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
        mkEq(roundingMode, mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven)) and
                (roundedResult.significand eq bvZero(roundedResult.sort.significandBits).cast()),
        mkEq(roundingMode, mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToAway)) and
                (roundedResult.significand eq bvZero(roundedResult.sort.significandBits).cast()),
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
        isZero,
        zero,
        iteOp(
            underflow,
            iteOp(returnZero, zero, min),
            iteOp(overflow, iteOp(returnInf, inf, max), roundedResult)
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
        mkEq(roundingMode, mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven)),
        guardBit,
        mkOr(stickyBit, !significandEven)
    )
    val roundUpRNA = mkAnd(
        mkEq(roundingMode, mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToAway)), guardBit
    )
    val roundUpRTP = mkAnd(
        mkEq(roundingMode, mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardPositive)), !sign, mkOr(guardBit, stickyBit)
    )
    val roundUpRTN = mkAnd(
        mkEq(roundingMode, mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardNegative)), sign, mkOr(guardBit, stickyBit)
    )
    val roundUpRTZ = mkAnd(
        mkEq(roundingMode, mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardZero)), false.expr
    )

    return mkAnd(
        !knownRoundDown, mkOr(roundUpRNE, roundUpRNA, roundUpRTP, roundUpRTN, roundUpRTZ)
    )
}



