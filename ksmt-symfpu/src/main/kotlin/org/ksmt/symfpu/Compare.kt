package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KFpSort
import org.ksmt.symfpu.UnpackedFp.Companion.iteOp

internal fun <Fp : KFpSort> KContext.less(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>
): KExpr<KBoolSort> {
    val infCase =
        (left.isNegativeInfinity and right.isNegativeInfinity.not()) or (left.isPositiveInfinity.not() and right.isPositiveInfinity)

    val zeroCase =
        (left.isZero and right.isZero.not() and right.isNegative.not()) or (left.isZero.not() and left.isNegative and right.isZero)

    return lessHelper(left, right, infCase, zeroCase)
}

internal fun <Fp : KFpSort> KContext.lessOrEqual(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>
): KExpr<KBoolSort> {
    val infCase = (left.isInf and right.isInf and (left.isNegative eq right.isNegative)) or
            left.isNegativeInfinity or right.isPositiveInfinity


    val zeroCase =
        (left.isZero and right.isZero) or (left.isZero and right.isNegative.not()) or (left.isNegative and right.isZero)


    return lessHelper(
        left, right, infCase, zeroCase, mkBvUnsignedLessOrEqualExpr(left.significand, right.significand),
        mkBvUnsignedLessOrEqualExpr(right.significand, left.significand),
    )
}

// common logic for less and lessOrEqual
private fun <Fp : KFpSort> KContext.lessHelper(
    left: UnpackedFp<Fp>,
    right: UnpackedFp<Fp>,
    infCase: KExpr<KBoolSort>,
    zeroCase: KExpr<KBoolSort>,
    positiveCaseSignificandComparison: KExpr<KBoolSort> = mkBvUnsignedLessExpr(left.significand, right.significand),
    negativeCaseSignificandComparison: KExpr<KBoolSort> = mkBvUnsignedLessExpr(right.significand, left.significand),
): KExpr<KBoolSort> {
    val neitherNan = left.isNaN.not() and right.isNaN.not()

    // Infinities are bigger than everything but themselves
    val eitherInf = left.isInf or right.isInf

    // Both zero are equal
    val eitherZero = left.isZero or right.isZero

    // Normal and subnormal
    val negativeLessThanPositive = left.isNegative and right.isNegative.not()
    val positiveCase = left.isNegative.not() and right.isNegative.not() and (mkBvSignedLessExpr(
        left.exponent, right.exponent
    ) or (left.exponent eq right.exponent and positiveCaseSignificandComparison))


    val negativeCase = left.isNegative and right.isNegative and (mkBvSignedLessExpr(
        right.exponent, left.exponent
    ) or (left.exponent eq right.exponent and negativeCaseSignificandComparison))


    return neitherNan and mkIte(
        eitherInf, infCase, mkIte(
            eitherZero, zeroCase, negativeLessThanPositive or positiveCase or negativeCase
        )
    )
}

internal fun <Fp : KFpSort> KContext.greater(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>
): KExpr<KBoolSort> = less(right, left)


internal fun <Fp : KFpSort> KContext.greaterOrEqual(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>
) = lessOrEqual(right, left)

internal fun <Fp : KFpSort> KContext.equal(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>
): KExpr<KBoolSort> {
    // All comparison with NaN are false
    val neitherNan = !left.isNaN and !right.isNaN

    val bothZero = left.isZero and right.isZero
    val neitherZero = !left.isZero and !right.isZero

    val flagsAndExponent = neitherNan and
            (bothZero or (neitherZero and
                    (left.isInf eq right.isInf and (left.sign eq right.sign) and (left.exponent eq right.exponent))))

    if (left.packedBv != null) {
        if (right.packedBv != null) {
            return neitherNan and (bothZero or (left.packedBv eq right.packedBv))
        }
    }
    return flagsAndExponent and (left.significand eq right.significand)
}

internal fun <Fp : KFpSort> KContext.min(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>
): KExpr<Fp> {
    return iteOp(right.isNaN or less(left, right), left, right)
}


internal fun <Fp : KFpSort> KContext.max(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>
): KExpr<Fp> = iteOp(right.isNaN or greater(left, right), left, right)
