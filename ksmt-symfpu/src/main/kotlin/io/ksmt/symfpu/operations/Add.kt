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
import io.ksmt.utils.BvUtils.bvOne
import io.ksmt.utils.BvUtils.bvZero


internal fun <Fp : KFpSort> KContext.sub(
    left: UnpackedFp<Fp>,
    right: UnpackedFp<Fp>,
    roundingMode: KExpr<KFpRoundingModeSort>
) = add(left, right, roundingMode, falseExpr)

internal fun <Fp : KFpSort> KContext.add(
    left: UnpackedFp<Fp>,
    right: UnpackedFp<Fp>,
    roundingMode: KExpr<KFpRoundingModeSort>,
    isAdd: KExpr<KBoolSort> = trueExpr
): UnpackedFp<Fp> {

    val knownInCorrectOrder = falseExpr
    val ec = addExponentCompare(
        left.normalizedSignificand.sort.sizeBits.toInt(),
        left.unbiasedExponent, right.unbiasedExponent, knownInCorrectOrder
    )

    val additionResult = arithmeticAdd(
        roundingMode, left, right, isAdd, knownInCorrectOrder, ec
    )

    val roundedAdditionResult = round(additionResult.unpackedFp, roundingMode, left.sort, additionResult.rounderInfo)
    return addAdditionSpecialCases(left.sort, roundingMode, left, right, roundedAdditionResult, isAdd)
}

data class ExponentCompareInfo(
    val leftIsMax: KExpr<KBoolSort>,
    val maxExponent: KExpr<KBvSort>,
    val absoluteExponentDifference: KExpr<KBvSort>,
    val diffIsZero: KExpr<KBoolSort>,
    val diffIsOne: KExpr<KBoolSort>,
    val diffIsGreaterThanPrecision: KExpr<KBoolSort>,
    val diffIsTwoToPrecision: KExpr<KBoolSort>,
    val diffIsGreaterThanPrecisionPlusOne: KExpr<KBoolSort>
)

fun KContext.addExponentCompare(
    significandWidth: Int,
    leftExponent: KExpr<KBvSort>,
    rightExponent: KExpr<KBvSort>,
    knownInCorrectOrder: KExpr<KBoolSort>
): ExponentCompareInfo {
    val exponentWidth = leftExponent.sort.sizeBits.toInt() + 1
    check(exponentWidth == rightExponent.sort.sizeBits.toInt() + 1) { "Exponent widths must be the same" }

    val exponentDifference = mkBvSubExpr(
        mkBvSignExtensionExpr(1, leftExponent),
        mkBvSignExtensionExpr(1, rightExponent)
    )

    val signBit = isAllOnes(mkBvExtractExpr(exponentWidth - 1, exponentWidth - 1, exponentDifference))
    val leftIsMax = knownInCorrectOrder or !signBit

    val maxExponent = mkIte(
        leftIsMax, mkBvSignExtensionExpr(1, leftExponent),
        mkBvSignExtensionExpr(1, rightExponent)
    )

    val absoluteExponentDifference = mkIte(
        leftIsMax,
        exponentDifference,
        mkBvNegationExpr(exponentDifference)
    )  // Largest negative value not obtainable so negate is safe

    val diffIsZero = absoluteExponentDifference eq bvZero(exponentWidth.toUInt())
    val diffIsOne = absoluteExponentDifference eq bvOne(exponentWidth.toUInt())
    val diffIsGreaterThanPrecision = mkBvSignedLessExpr(
        mkBv(significandWidth, exponentWidth.toUInt()),
        absoluteExponentDifference
    )  // Assumes this is representable

    val diffIsTwoToPrecision = mkAnd(
        !diffIsZero,
        !diffIsOne,
        !diffIsGreaterThanPrecision
    )
    val diffIsGreaterThanPrecisionPlusOne = mkBvSignedLessExpr(
        mkBv(significandWidth + 1, exponentWidth.toUInt()),
        absoluteExponentDifference
    )

    return ExponentCompareInfo(
        leftIsMax,
        maxExponent,
        absoluteExponentDifference,
        diffIsZero,
        diffIsOne,
        diffIsGreaterThanPrecision,
        diffIsTwoToPrecision,
        diffIsGreaterThanPrecisionPlusOne
    )
}

@Suppress("LongParameterList")
private fun <Fp : KFpSort> KContext.addAdditionSpecialCasesComplete(
    format: Fp,
    roundingMode: KExpr<KFpRoundingModeSort>,
    left: UnpackedFp<Fp>,
    leftID: UnpackedFp<Fp>,
    right: UnpackedFp<Fp>,
    returnLeft: KExpr<KBoolSort>,
    returnRight: KExpr<KBoolSort>,
    additionResult: UnpackedFp<Fp>,
    isAdd: KExpr<KBoolSort>
): UnpackedFp<Fp> {
    val eitherArgumentNan = left.isNaN or right.isNaN
    val bothInfinity = left.isInf and right.isInf
    val signsMatch = left.sign eq right.sign
    val compatibleSigns = isAdd xor !signsMatch // ITE(isAdd, signsMatch, !signsMatch)
    val generatesNaN = eitherArgumentNan or (bothInfinity and !compatibleSigns)

    val generatesInf = (bothInfinity and compatibleSigns) or
            (left.isInf and !right.isInf) or
            (!left.isInf and right.isInf)

    val signOfInf = mkIte(left.isInf, left.sign, isAdd xor !right.sign)

    val bothZero = left.isZero and right.isZero
    val flipRightSign = !isAdd xor right.sign
    val signOfZero = mkIte(
        roundingEq(roundingMode, KFpRoundingMode.RoundTowardNegative),
        left.sign or flipRightSign,
        left.sign and flipRightSign
    )

    val idLeft = !left.isZero and right.isZero
    val idRight = left.isZero and !right.isZero

    return iteOp(
        idRight or returnRight,
        iteOp(
            isAdd,
            right,
            (right).negate()
        ),
        iteOp(
            idLeft or returnLeft,
            leftID,
            iteOp(
                generatesNaN,
                makeNaN(format),
                iteOp(
                    generatesInf,
                    makeInf(format, signOfInf),
                    iteOp(
                        bothZero,
                        makeZero(format, signOfZero),
                        additionResult
                    )
                )
            )
        )
    )
}

@Suppress("LongParameterList")
private fun <Fp : KFpSort> KContext.addAdditionSpecialCases(
    format: Fp,
    roundingMode: KExpr<KFpRoundingModeSort>,
    left: UnpackedFp<Fp>,
    right: UnpackedFp<Fp>,
    additionResult: UnpackedFp<Fp>,
    isAdd: KExpr<KBoolSort>
): UnpackedFp<Fp> = addAdditionSpecialCasesComplete(
    format = format,
    roundingMode = roundingMode,
    left = left,
    leftID = left,
    right = right,
    returnLeft = falseExpr,
    returnRight = falseExpr,
    additionResult = additionResult,
    isAdd = isAdd
)

@Suppress("LongParameterList")
fun <Fp : KFpSort> KContext.addAdditionSpecialCasesWithID(
    format: Fp,
    roundingMode: KExpr<KFpRoundingModeSort>,
    left: UnpackedFp<Fp>,
    leftID: UnpackedFp<Fp>,
    right: UnpackedFp<Fp>,
    additionResult: UnpackedFp<Fp>,
    isAdd: KExpr<KBoolSort>
): UnpackedFp<Fp> = addAdditionSpecialCasesComplete(
    format = format,
    roundingMode = roundingMode,
    left = left,
    leftID = leftID,
    right = right,
    returnLeft = falseExpr,
    returnRight = falseExpr,
    additionResult = additionResult,
    isAdd = isAdd
)

data class FloatWithCustomRounderInfo<Fp : KFpSort>(
    val unpackedFp: UnpackedFp<Fp>,
    val rounderInfo: CustomRounderInfo
)

@Suppress("LongMethod", "LongParameterList", "MagicNumber")
fun <Fp : KFpSort> KContext.arithmeticAdd(
    roundingMode: KExpr<KFpRoundingModeSort>,
    left: UnpackedFp<Fp>,
    right: UnpackedFp<Fp>,
    isAdd: KExpr<KBoolSort>,
    knownInCorrectOrder: KExpr<KBoolSort>,
    ec: ExponentCompareInfo
): FloatWithCustomRounderInfo<KFpSort> {
    val effectiveAdd = left.sign xor right.sign xor isAdd
    val exponentWidth = left.exponentWidth() + 1u
    val significandWidth = left.significandWidth()

    // Rounder flags
    val noOverflow = !effectiveAdd
    val noUnderflow = trueExpr

    val subnormalExact = trueExpr

    val noSignificandOverflow = (effectiveAdd and ec.diffIsZero) or
            (!effectiveAdd and (ec.diffIsZero or ec.diffIsOne))

    val stickyBitIsZero = ec.diffIsZero or ec.diffIsOne

    // Work out ordering
    val leftSigLarger = mkIte(
        !ec.diffIsZero,
        trueExpr,
        mkBvUnsignedGreaterOrEqualExpr(left.getSignificand(), right.getSignificand())
    )
    val leftLarger = knownInCorrectOrder or (ec.leftIsMax and leftSigLarger)

    // Extend the significands to give room for carry plus guard and sticky bits
    val largerSig = mkBvConcatExpr(
        bvZero(),
        mkIte(leftLarger, left.getSignificand(), right.getSignificand()),
        bvZero(2u)
    )
    val smallerSig = mkBvConcatExpr(
        bvZero(),
        mkIte(leftLarger, right.getSignificand(), left.getSignificand()),
        bvZero(2u)
    )

    val resultSign = mkIte(
        leftLarger,
        left.sign,
        !isAdd xor right.sign
    )

    // Extended so no info lost, negate before shift so that sign-extension works
    val negatedSmaller = mkIte(effectiveAdd, smallerSig, mkBvNegationExpr(smallerSig))

    val shiftAmount = ec.absoluteExponentDifference // Safe as >= 0
        .resizeUnsigned(
            negatedSmaller.sort.sizeBits
        ) // Safe as long as the significand has more bits than the exponent


    // Shift the smaller significand
    val shifted = stickyRightShift(negatedSmaller, shiftAmount)

    val negatedAlignedSmaller =
        mkIte(
            ec.diffIsGreaterThanPrecisionPlusOne,
            mkIte(
                effectiveAdd,
                bvZero(negatedSmaller.sort.sizeBits),
                ones(negatedSmaller.sort.sizeBits)
            ),
            shifted.signExtendedResult
        )

    val shiftedStickyBit = mkIte(
        ec.diffIsGreaterThanPrecision,
        bvOne(negatedSmaller.sort.sizeBits),
        shifted.stickyBit
    )  // Have to separate otherwise align up may convert it to the guard bit

    // Sum and re-align
    val sum = mkBvAddExpr(largerSig, negatedAlignedSmaller)

    val sumWidth = sum.sort.sizeBits.toInt()
    val topBit = sum.nthBit(sumWidth - 1)
    val alignedBit = sum.nthBit(sumWidth - 2)
    val lowerBit = sum.nthBit(sumWidth - 3)

    val overflow = !isAllZeros(topBit)
    val cancel = isAllZeros(topBit) and isAllZeros(alignedBit)
    val minorCancel = cancel and isAllOnes(lowerBit)
    val majorCancel = cancel and isAllZeros(lowerBit)

    val fullCancel = majorCancel and isAllZeros(sum)
    val exact = cancel and (ec.diffIsZero or ec.diffIsOne) // For completeness

    val alignedSum = conditionalLeftShiftOne(
        minorCancel,
        conditionalRightShiftOne(overflow, sum)
    )
    val one = bvOne<KBvSort>(exponentWidth)
    val exponentCorrectionTerm = mkIte(
        minorCancel,
        mkBvNegationExpr(one),
        mkIte(
            overflow,
            one,
            bvZero(exponentWidth)
        )
    )

    val correctedExponent = mkBvAddExpr(ec.maxExponent, exponentCorrectionTerm)
    val stickyBit = mkIte(
        stickyBitIsZero or majorCancel,
        bvZero(alignedSum.sort.sizeBits),
        mkBvOrExpr(
            shiftedStickyBit,
            mkBvZeroExtensionExpr(
                alignedSum.sort.sizeBits.toInt() - 1,
                mkIte(
                    !overflow,
                    bvZero(),
                    sum.nthBit(0)
                )
            )
        )
    )

    val extendedFormat = mkFpSort(exponentWidth + 1u, significandWidth + 2u)
    val sumResult = UnpackedFp(
        this,
        extendedFormat,
        resultSign,
        correctedExponent,
        mkBvOrExpr(alignedSum, stickyBit).contract(1)
    )

    val rtnRounding = mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardNegative)

    val additionResult = iteOp(
        fullCancel,
        makeZero(extendedFormat, roundingMode eq rtnRounding),
        iteOp(
            majorCancel,
            sumResult.normaliseUp(),
            sumResult
        )
    )

    return FloatWithCustomRounderInfo(
        additionResult,
        CustomRounderInfo(
            noOverflow,
            noUnderflow,
            exact,
            subnormalExact,
            noSignificandOverflow
        )
    )
}

private fun KExpr<KBvSort>.nthBit(pos: Int) =
    ctx.mkBvExtractExpr(pos, pos, this)

data class StickyRightShiftResult(val signExtendedResult: KExpr<KBvSort>, val stickyBit: KExpr<KBvSort>)

fun KContext.stickyRightShift(input: KExpr<KBvSort>, shiftAmount: KExpr<KBvSort>): StickyRightShiftResult =
    StickyRightShiftResult(mkBvArithShiftRightExpr(input, shiftAmount), rightShiftStickyBit(input, shiftAmount))

fun KContext.rightShiftStickyBit(op: KExpr<KBvSort>, shift: KExpr<KBvSort>): KExpr<KBvSort> = mkIte(
    isAllZeros(mkBvAndExpr(orderEncode(shift), op)),
    bvZero(op.sort.sizeBits),
    bvOne(op.sort.sizeBits)
)

fun KContext.orderEncode(op: KExpr<KBvSort>): KExpr<KBvSort> {
    val w = op.sort.sizeBits
    return mkBvExtractExpr(
        high = (w - 1u).toInt(),
        low = 0,
        value = decrement(
            mkBvShiftLeftExpr(
                bvOne(w + 1u),
                op.resizeUnsigned(w + 1u)
            )
        )
    )
}
