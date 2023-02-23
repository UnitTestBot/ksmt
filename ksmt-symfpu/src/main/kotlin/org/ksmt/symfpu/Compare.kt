package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort
import org.ksmt.symfpu.UnpackedFp.Companion.iteOp

internal fun <Fp : KFpSort> KContext.less(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>
): KExpr<KBoolSort> {
    val infCase =
        (left.isNegativeInfinity and !right.isNegativeInfinity) or (!left.isPositiveInfinity and right.isPositiveInfinity)

    val zeroCase =
        (left.isZero and !right.isZero and !right.isNegative) or (!left.isZero and left.isNegative and right.isZero)

    val packedExists = left.packedBv != null && right.packedBv != null
    println("packedExists: $packedExists")

    return lessHelper(
        left, right, infCase, zeroCase, packedExists, positiveCaseSignificandComparison = mkBvUnsignedLessExpr(
            left.getSignificand(packedExists), right.getSignificand(packedExists)
        ), negativeCaseSignificandComparison = mkBvUnsignedLessExpr(
            right.getSignificand(packedExists), left.getSignificand(packedExists)
        )
    )
}


internal fun <Fp : KFpSort> KContext.lessOrEqual(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>
): KExpr<KBoolSort> {
    val infCase =
        (left.isInf and right.isInf and (left.isNegative eq right.isNegative)) or left.isNegativeInfinity or right.isPositiveInfinity


    val zeroCase =
        (left.isZero and right.isZero) or (left.isZero and right.isNegative.not()) or (left.isNegative and right.isZero)

    val packedExists = left.packedBv != null && right.packedBv != null
    println("packedExists: $packedExists")

    return lessHelper(
        left,
        right,
        infCase,
        zeroCase,
        packedExists,
        mkBvUnsignedLessOrEqualExpr(left.getSignificand(packedExists), right.getSignificand(packedExists)),
        mkBvUnsignedLessOrEqualExpr(right.getSignificand(packedExists), left.getSignificand(packedExists)),
    )
}

// common logic for less and lessOrEqual
private fun <Fp : KFpSort> KContext.lessHelper(
    left: UnpackedFp<Fp>,
    right: UnpackedFp<Fp>,
    infCase: KExpr<KBoolSort>,
    zeroCase: KExpr<KBoolSort>,
    packedExists: Boolean,
    positiveCaseSignificandComparison: KExpr<KBoolSort> = mkBvUnsignedLessExpr(
        left.getSignificand(packedExists), right.getSignificand(packedExists)
    ),
    negativeCaseSignificandComparison: KExpr<KBoolSort> = mkBvUnsignedLessExpr(
        right.getSignificand(packedExists), left.getSignificand(packedExists)
    ),
): KExpr<KBoolSort> {
    val neitherNan = !left.isNaN and !right.isNaN

    // Infinities are bigger than everything but themselves
    val eitherInf = left.isInf or right.isInf

    // Both zero are equal
    val eitherZero = left.isZero or right.isZero

    val exponentLessExpr = { a: KExpr<KBvSort>, b : KExpr<KBvSort>->
        if (packedExists)
            mkBvUnsignedLessExpr(a, b)
        else
            mkBvSignedLessExpr(a, b)
    }
    // Normal and subnormal
    val negativeLessThanPositive = left.isNegative and !right.isNegative
    val positiveCase = !left.isNegative and !right.isNegative and (exponentLessExpr(
        left.getExponent(packedExists), right.getExponent(packedExists)
    ) or (left.getExponent(packedExists) eq right.getExponent(packedExists) and positiveCaseSignificandComparison))


    val negativeCase = left.isNegative and right.isNegative and (exponentLessExpr(
        right.getExponent(packedExists), left.getExponent(packedExists)
    ) or (left.getExponent(packedExists) eq right.getExponent(packedExists) and negativeCaseSignificandComparison))


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

    val flagsAndExponent =
        neitherNan and (bothZero or (neitherZero and (left.isInf eq right.isInf and (left.sign eq right.sign) and (left.unbiasedExponent eq right.unbiasedExponent))))

    if (left.packedBv != null) {
        if (right.packedBv != null) {
            return neitherNan and (bothZero or (left.packedBv eq right.packedBv))
        }
    }
    return flagsAndExponent and (left.normalizedSignificand eq right.normalizedSignificand)
}

internal fun <Fp : KFpSort> KContext.min(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>
): KExpr<Fp> {
    return iteOp(right.isNaN or less(left, right), left, right)
}


internal fun <Fp : KFpSort> KContext.max(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>
): KExpr<Fp> = iteOp(right.isNaN or greater(left, right), left, right)
