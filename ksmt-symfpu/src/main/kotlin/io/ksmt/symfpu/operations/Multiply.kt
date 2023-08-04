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
import io.ksmt.utils.BvUtils.bvOne

internal fun <Fp : KFpSort> KContext.multiply(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>, roundingMode: KExpr<KFpRoundingModeSort>
): KExpr<Fp> {
    val multiplyResult = arithmeticMultiply(left, right)
    val rounded = round(multiplyResult, roundingMode, left.sort)
    return addMultSpecialCases(rounded, left, right)
}

fun <Fp : KFpSort, T : KFpSort> KContext.addMultSpecialCases(
    multiplyResult: UnpackedFp<Fp>, left: UnpackedFp<T>, right: UnpackedFp<T>
): UnpackedFp<Fp> {
    val eitherArgumentNan = left.isNaN or right.isNaN

    val generateNan = ((left.isInf and right.isZero) or (left.isZero and right.isInf))

    val isNan = eitherArgumentNan or generateNan

    val isInf = left.isInf or right.isInf
    val isZero = left.isZero or right.isZero

    return iteOp(
        isNan,
        makeNaN(multiplyResult.sort),
        iteOp(
            isInf,
            makeInf(multiplyResult.sort, multiplyResult.sign),
            iteOp(
                isZero,
                makeZero(multiplyResult.sort, multiplyResult.sign),
                multiplyResult
            )
        )
    )
}

fun KContext.expandingMultiply(left: KExpr<KBvSort>, right: KExpr<KBvSort>): KExpr<KBvSort> {
    val width = left.sort.sizeBits.toInt()
    val x = mkBvZeroExtensionExpr(width, left)
    val y = mkBvZeroExtensionExpr(width, right)
    return mkBvMulExpr(x, y)
}

private fun KContext.expandingAddWithCarryIn(
    left: KExpr<KBvSort>, right: KExpr<KBvSort>, carry: KExpr<KBoolSort>
): KExpr<KBvSort> {
    val x = mkBvSignExtensionExpr(1, left)
    val y = mkBvSignExtensionExpr(1, right)
    val sum = mkBvAddExpr(x, y)

    val carryBv = mkIte(carry, mkBv(1, sum.sort), mkBv(0, sum.sort))
    return mkBvAddExpr(sum, carryBv)
}

fun <Fp : KFpSort> KContext.arithmeticMultiply(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>
): UnpackedFp<KFpSort> {
    // Compute sign
    val multiplySign = left.sign xor right.sign

    // Multiply the significands
    val significandProduct = expandingMultiply(left.normalizedSignificand, right.normalizedSignificand)

    val spWidth = significandProduct.sort.sizeBits.toInt()
    val topBit = mkBvExtractExpr(spWidth - 1, spWidth - 1, significandProduct)
    val topBitSet = isAllOnes(topBit)

    val alignedSignificand = conditionalLeftShiftOne(!topBitSet, significandProduct)

    val alignedExponent = expandingAddWithCarryIn(left.unbiasedExponent, right.unbiasedExponent, topBitSet)

    val sort = mkFpSort(left.sort.exponentBits + 1u, left.sort.significandBits * 2u)
    return UnpackedFp(
        this, sort, multiplySign, alignedExponent, alignedSignificand
    )
}

fun KContext.conditionalLeftShiftOne(
    condition: KExpr<KBoolSort>, expr: KExpr<KBvSort>
) = mkIte(
    condition, mkBvShiftLeftExpr(expr, bvOne(expr.sort.sizeBits)), expr
)

fun KContext.conditionalRightShiftOne(
    condition: KExpr<KBoolSort>, expr: KExpr<KBvSort>
) = mkIte(
    condition, mkBvLogicalShiftRightExpr(expr, bvOne(expr.sort.sizeBits)), expr
)
