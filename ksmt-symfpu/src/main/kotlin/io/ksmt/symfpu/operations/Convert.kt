package io.ksmt.symfpu.operations

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.symfpu.operations.UnpackedFp.Companion.iteOp
import io.ksmt.symfpu.operations.UnpackedFp.Companion.makeInf
import io.ksmt.symfpu.operations.UnpackedFp.Companion.makeNaN
import io.ksmt.symfpu.operations.UnpackedFp.Companion.makeZero
import kotlin.math.max

fun <T : KFpSort> bvToFp(
    roundingMode: KExpr<KFpRoundingModeSort>,
    input: KExpr<KBvSort>,
    targetFormat: T,
    signed: Boolean,
): UnpackedFp<T> = with(input.ctx) {
    // In the case of a 1 bit input(?) extend to 2 bits so that the intermediate float is a sensible format
    val inputBv = if (input.sort.sizeBits == 1u) input.extendUnsigned(1) else input
    val inputWidth = inputBv.sort.sizeBits
    if (signed) {
        val initialExponentWidth = bitsToRepresent(inputWidth.toInt()) + 1
        val initialFormat = mkFpSort(initialExponentWidth.toUInt(), inputWidth + 1u)
        val actualExponentWidth = unpackedExponentWidth(initialFormat).toUInt()

        // Work out the sign
        val negative = mkBvSignedLessExpr(inputBv, mkBv(0, inputWidth))
        val significand = mkBvSignExtensionExpr(1, inputBv).let { mkIte(negative, mkBvNegationExpr(it), it) }

        val initial = UnpackedFp(
            ctx = this,
            sort = initialFormat,
            sign = negative,
            exponent = mkBv(inputWidth.toInt(), actualExponentWidth),
            significand = significand,
        )

        val normalised = initial.normaliseUpDetectZero()
        // Round (the conversion will catch the cases where no rounding is needed)
        return fpToFp(targetFormat, roundingMode, normalised)
    } else {
        val initialExponentWidth = bitsToRepresent(inputWidth.toInt()) + 1
        val initialFormat = mkFpSort(initialExponentWidth.toUInt(), inputWidth)
        val actualExponentWidth = unpackedExponentWidth(initialFormat).toUInt()

        val initial = UnpackedFp(
            ctx = this,
            sort = initialFormat,
            sign = falseExpr,
            exponent = mkBv(inputWidth.toInt() - 1, actualExponentWidth),
            significand = inputBv,
        )

        val normalised = initial.normaliseUpDetectZero()
        // Round (the conversion will catch the cases where no rounding is needed)
        return fpToFp(targetFormat, roundingMode, normalised)
    }
}


fun <T : KFpSort, S : KFpSort> fpToFp(
    targetFormat: T,
    roundingMode: KExpr<KFpRoundingModeSort>,
    input: UnpackedFp<S>,
): UnpackedFp<T> = with(input.ctx) {
    val sourceFormat = input.sort

    val exponentIncreased = unpackedExponentWidth(sourceFormat) <= unpackedExponentWidth(targetFormat)
    val significandIncreased = unpackedSignificandWidth(sourceFormat) <= unpackedSignificandWidth(targetFormat)


    val expExpression = max(0, unpackedExponentWidth(targetFormat) - unpackedExponentWidth(sourceFormat))
    val sigExpression = max(0, unpackedSignificandWidth(targetFormat) - unpackedSignificandWidth(sourceFormat))

    val extended = input.extend(expExpression, sigExpression, targetFormat)

    return if (exponentIncreased && significandIncreased) {
        extended
    } else {
        val rounded = round(extended, roundingMode, targetFormat)
        iteOp(
            input.isNaN,
            makeNaN(targetFormat),
            iteOp(
                input.isInf,
                makeInf(targetFormat, input.sign),
                iteOp(
                    input.isZero,
                    makeZero(targetFormat, input.sign),
                    rounded
                )
            )
        )
    }
}

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
        collaredRoundingPoint.matchWidthUnsigned(significand)
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

fun fpToBv(
    roundingMode: KExpr<KFpRoundingModeSort>,
    input: UnpackedFp<*>,
    targetWidth: Int,
    signed: Boolean,
): KExpr<KBvSort> = with(input.ctx) {
    val specialValue = input.isInf or input.isNaN
    val maxExponentBits = bitsToRepresent(targetWidth) + 1
    val exponentWidth = input.exponentWidth().toInt()
    val workingExponentWidth = max(exponentWidth, maxExponentBits)
    val maxExponent = mkBv(targetWidth, workingExponentWidth.toUInt())
    val exponent = input.getExponent().matchWidthSigned(this, maxExponent)
    val tooLarge = mkBvSignedGreaterOrEqualExpr(exponent, maxExponent)
    val rounded = fpToBvCommon(roundingMode, input, targetWidth)

    val unspecified = mkBv(-1, targetWidth.toUInt())

    if (signed) {
        val earlyUndefinedResult = specialValue or tooLarge
        val roundSigWidth = rounded.significand.sort.sizeBits.toInt()
        val undefinedResult = earlyUndefinedResult or rounded.incrementExponent or (isAllOnes(
            mkBvExtractExpr(
                roundSigWidth - 1,
                roundSigWidth - 1,
                rounded.significand
            )
        ) and !input.sign and !isAllZeros(
            mkBvExtractExpr(
                roundSigWidth - 2, 0, rounded.significand
            )
        ))
        mkIte(
            undefinedResult,
            unspecified,
            conditionalNegate(
                input.sign, rounded.significand
            )
        )
    } else {
        val tooNegative = input.sign and !input.isZero and mkBvSignedLessOrEqualExpr(
            mkBv(0, workingExponentWidth.toUInt()),
            exponent
        )

        val earlyUndefinedResult = specialValue or tooLarge or tooNegative
        val undefinedResult = earlyUndefinedResult or rounded.incrementExponent or (input.sign and !isAllZeros(
            rounded.significand
        ))
        mkIte(
            undefinedResult,
            unspecified,
            rounded.significand
        )
    }
}

private fun fpToBvCommon(
    roundingMode: KExpr<KFpRoundingModeSort>,
    input: UnpackedFp<*>,
    targetWidth: Int,
): SignificandRounderResult =
    with(input.ctx) {

        val maxShift = targetWidth + 1
        val maxShiftBits = bitsToRepresent(maxShift) + 1

        val exponentWidth = input.exponentWidth().toInt()
        val workingExponentWidth = if (exponentWidth >= maxShiftBits) {
            exponentWidth
        } else {
            maxShiftBits
        }

        val maxShiftAmount = mkBv(maxShift, workingExponentWidth.toUInt())
        val exponent = input.getExponent().matchWidthSigned(this, maxShiftAmount)


        val inputSignificand = input.getSignificand()
        val inputSignificandWidth = input.significandWidth().toInt()
        val significand = if (targetWidth + 2 < inputSignificandWidth) {
            val dataAndGuard =
                mkBvExtractExpr(
                    inputSignificandWidth - 1,
                    (inputSignificandWidth - targetWidth) - 1,
                    inputSignificand
                )
            val sticky =
                !isAllZeros(mkBvExtractExpr((inputSignificandWidth - targetWidth) - 2, 0, inputSignificand))
            mkBvConcatExpr(dataAndGuard, boolToBv(sticky))
        } else {
            inputSignificand
        }
        val significandWidth = significand.sort.sizeBits
        val zeroedSignificand = mkBvAndExpr(
            significand,
            mkIte(
                input.isZero,
                mkBv(0, significandWidth),
                ones(significandWidth)
            )
        )
        val expandedSignificand = mkBvZeroExtensionExpr(maxShift, zeroedSignificand)


        val shiftAmount = collar(
            expandingAdd(
                exponent,
                mkBv(2, workingExponentWidth.toUInt())
            ),
            mkBv(0, workingExponentWidth.toUInt() + 1u),
            mkBvSignExtensionExpr(1, maxShiftAmount)
        )


        val convertedShiftAmount = shiftAmount.resizeSigned(bitsToRepresent(maxShift).toUInt() + 1u)
            .matchWidthUnsigned(expandedSignificand)

        val aligned = mkBvShiftLeftExpr(expandedSignificand, convertedShiftAmount)

        return fixedPositionRound(
            roundingMode, input.sign, aligned, targetWidth,
            mkFalse(), mkFalse()
        )
    }


private fun KContext.expandingAdd(
    left: KExpr<KBvSort>, right: KExpr<KBvSort>
): KExpr<KBvSort> {
    val x = mkBvSignExtensionExpr(1, left)
    val y = mkBvSignExtensionExpr(1, right)

    return mkBvAddExpr(x, y)
}








