package io.ksmt.symfpu.operations

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFpRoundingMode
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.symfpu.operations.UnpackedFp.Companion.makeZero
import io.ksmt.utils.cast

internal fun <Fp : KFpSort> KContext.fma(
    leftMultiply: UnpackedFp<Fp>,
    rightMultiply: UnpackedFp<Fp>,
    addArgument: UnpackedFp<Fp>,
    roundingMode: KExpr<KFpRoundingModeSort>,
): UnpackedFp<Fp> {
    val format = leftMultiply.sort

    /* First multiply */
    val arithmeticMultiplyResult = arithmeticMultiply(leftMultiply, rightMultiply)

    val extendedFormat = mkFpSort(
        format.exponentBits + 1u,
        format.significandBits * 2u,
    )

    /* Then add */

    // Rounding mode doesn't matter as this is a strict extension
    val extendedAddArgument = fpToFp(extendedFormat, mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardZero), addArgument)

    val ec = addExponentCompare(
        arithmeticMultiplyResult.significandWidth().toInt(),
        arithmeticMultiplyResult.getExponent(),
        extendedAddArgument.getExponent(),
        falseExpr,
    )

    val additionResult =
        arithmeticAdd(roundingMode, arithmeticMultiplyResult, extendedAddArgument, trueExpr, falseExpr, ec).uf


    /* Then round */
    val roundedResult = round(additionResult, roundingMode, format)

    // This result is correct as long as neither of multiplyResult or extendedAddArgument is
    // 0, Inf or NaN.  Note that roundedResult may be zero from cancellation or underflow
    // or infinity due to rounding. If it is, it has the correct sign.


    /* Finally, the special cases */


    // One disadvantage to having a flag for zero and default exponents and significands for zero
    // that are not (min, 0) is that the x + (+/-)0 case has to be handled by the addition special cases.
    // This means that you need the value of x, rounded to the correct format.
    // arithmeticMultiplyResult is in extended format, thus we have to use a second rounder just for this case.
    // It is not zero, inf or NaN, so it only matters when addArgument is zero when it would be returned.
    val roundedMultiplyResult = round(arithmeticMultiplyResult, roundingMode, format)

    val fullMultiplyResult = addMultSpecialCases(roundedMultiplyResult, leftMultiply, rightMultiply)


    // We need the flags from the multiply special cases, determined on the arithmetic result,
    // i.e. handling special values and not the underflow / overflow of the result.
    // But we will use roundedMultiplyResult instead of the value so ...
    val dummyZero = makeZero(format, arithmeticMultiplyResult.sign)
    val dummyValue = UnpackedFp(
        this,
        dummyZero.sort,
        sign = dummyZero.sign,
        exponent = dummyZero.getExponent(),
        significand = dummyZero.getSignificand()
    )

    val multiplyResultWithSpecialCases = addMultSpecialCases(dummyValue, leftMultiply, rightMultiply)

    return addAdditionSpecialCasesWithID(
        format,
        roundingMode,
        multiplyResultWithSpecialCases,
        fullMultiplyResult,
        addArgument.cast<UnpackedFp<Fp>, UnpackedFp<Fp>>(),
        roundedResult,
        trueExpr
    )
}
