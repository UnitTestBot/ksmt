package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.symfpu.UnpackedFp.Companion.iteOp
import org.ksmt.symfpu.UnpackedFp.Companion.makeNaN
import java.math.BigInteger
import java.math.BigInteger.ONE
import java.math.BigInteger.valueOf

internal fun <Fp : KFpSort> remainder(
    left: UnpackedFp<Fp>,
    right: UnpackedFp<Fp>,
): UnpackedFp<Fp> = with(left.ctx) {
    remainderWithRounding(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), left, right)
}

internal fun <Fp : KFpSort> remainderWithRounding(
    roundingMode: KExpr<KFpRoundingModeSort>,
    left: UnpackedFp<Fp>,
    right: UnpackedFp<Fp>,
): UnpackedFp<Fp> = with(left.ctx) {
    val remainderResult = arithmeticRemainder(roundingMode, left, right)
    return addRemainderSpecialCases(left, right, remainderResult)
}

private fun <Fp : KFpSort> KContext.addRemainderSpecialCases(
    left: UnpackedFp<Fp>,
    right: UnpackedFp<Fp>,
    remainderResult: UnpackedFp<Fp>
): UnpackedFp<Fp> {
    val eitherArgumentNan = left.isNaN or right.isNaN
    val generateNan = left.isInf or right.isZero
    val isNan = eitherArgumentNan or generateNan
    val passThrough = (!(left.isInf or left.isNaN) and right.isInf) or left.isZero

    return iteOp(
        isNan,
        makeNaN(left.sort),
        iteOp(
            passThrough,
            left,
            remainderResult
        )
    )
}

fun <Fp : KFpSort> KContext.arithmeticRemainder(
    roundingMode: KExpr<KFpRoundingModeSort>,
    left: UnpackedFp<Fp>,
    right: UnpackedFp<Fp>
): UnpackedFp<Fp> {
    // Compute sign
    val remainderSign = left.sign

    // Compute exponent difference
    val exponentDifference = expandingSubtractSigned(left.getExponent(), right.getExponent())
    val edWidth = exponentDifference.sort.sizeBits

    // Extend for divide steps
    val lsig = left.getSignificand().extendUnsigned(1)
    val rsig = right.getSignificand().extendUnsigned(1)
    var running = divideStep(lsig, rsig).result

    // The first step is a little different as we need the result bit for the even flag
    // and the actual result for the final
    val maxDifference = maximumExponentDifference(left.sort)
    var i = maxDifference - ONE
    while (i > valueOf(0)) {
        val needPrevious = mkBvSignedGreaterExpr(exponentDifference, mkBv(i, edWidth))
        val r = mkIte(needPrevious, running, lsig)
        running = divideStep(r, rsig).result
        i -= ONE
    }
    // The zero exponent difference case is a little different
    // as we need the result bit for the even flag
    // and the actual result for the final
    val lsbRoundActive = mkBvSignedGreaterExpr(exponentDifference, mkBvNegationExpr(mkBv(1, edWidth)))
    val needPrevious = mkBvSignedGreaterExpr(exponentDifference, mkBv(0, edWidth))
    val r0 = mkIte(needPrevious, running, lsig)


    val dsr = divideStep(r0, rsig)

    val integerEven = !lsbRoundActive or !dsr.remainderBit


    // The same to get the guard flag
    val guardRoundActive = mkBvSignedGreaterExpr(exponentDifference, mkBvNegationExpr(mkBv(2, edWidth)))

    val rm1 = mkIte(lsbRoundActive, dsr.result, lsig)
    val dsrg = divideStep(rm1, rsig)
    val guardBit = guardRoundActive and dsrg.remainderBit
    val stickyBit = !mkIte(guardRoundActive, dsrg.result, lsig).isAllZeros()

    // The base result if lsbRoundActive
    val reconstruct = UnpackedFp(
        this, left.sort,
        remainderSign,
        right.getExponent(),
        dsr.result.extract(lsig.sort.sizeBits.toInt() - 1, 1)
    )

    val candidateResult = iteOp(lsbRoundActive, reconstruct.normaliseUpDetectZero(), left)

    // The final subtract is a little different as previous ones were
    // guaranteed to be positive

    // From the rounding of the big integer multiple
    val bonusSubtract = roundingDecision(
        roundingMode,
        remainderSign,
        integerEven,
        guardBit,
        stickyBit,
        falseExpr
    )

    // The big integer has sign left.getSign() ^ right.getSign() so we subtract something of left.getSign().
    // For the integer part we handle this by working with absolutes (ignoring the sign) and
    // adding it back in at the end.
    // However, for the correction for the rounded part we need to take it into account

    val signCorrectedRight = right.setSign(left.sign)

    return iteOp(
        bonusSubtract,
        sub(candidateResult, signCorrectedRight, roundingMode),
        candidateResult
    )

}

fun <Fp : KFpSort> maximumExponentDifference(sort: Fp): BigInteger {
    val maxNormalExp = (ONE shl (sort.exponentBits - 1u).toInt()) - ONE
    val minSubnormalExp = -maxNormalExp - valueOf((sort.significandBits - 2u).toLong())
    return maxNormalExp - minSubnormalExp
}

// One step of a divider
// Here the "remainder bit" is actual the result bit and
// The result is the remainder
private fun KContext.divideStep(x: KExpr<KBvSort>, y: KExpr<KBvSort>): ResultWithRemainderBit {
    val xWidth = x.sort.sizeBits
    val yWidth = y.sort.sizeBits

    check(xWidth == yWidth)
    check(yWidth >= 2u)
    val canSubtract = mkBvUnsignedGreaterOrEqualExpr(x, y)
    val sub = mkBvAddExpr(x, mkBvNegationExpr(y)) // TODO : modular subtract or better
    val step = mkIte(canSubtract, sub, x)

    return ResultWithRemainderBit(mkBvShiftLeftExpr(step, mkBv(1, xWidth)), canSubtract)
}

private fun KExpr<KBvSort>.extendUnsigned(i: Int) = ctx.mkBvZeroExtensionExpr(i, this)
private fun KExpr<KBvSort>.extract(high: Int, low: Int) = ctx.mkBvExtractExpr(high, low, this)
private fun KExpr<KBvSort>.isAllZeros() = ctx.mkEq(this, ctx.mkBv(0, sort.sizeBits))

