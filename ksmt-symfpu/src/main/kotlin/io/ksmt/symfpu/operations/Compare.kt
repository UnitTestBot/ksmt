package io.ksmt.symfpu.operations

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpSort
import io.ksmt.symfpu.operations.UnpackedFp.Companion.iteOp

// doesn't matter if significand is normalized or not
internal fun <Fp : KFpSort> KContext.less(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>,
): KExpr<KBoolSort> {
    val infCase = (left.isNegativeInfinity and !right.isNegativeInfinity) or
            (!left.isPositiveInfinity and right.isPositiveInfinity)

    val zeroCase = (left.isZero and !right.isZero and !right.isNegative) or
            (!left.isZero and left.isNegative and right.isZero)

    val packedExists = left.packedFp.hasPackedFp() && right.packedFp.hasPackedFp()

    return lessHelper(
        left, right, infCase, zeroCase, packedExists,
        positiveCaseSignificandComparison = mkBvUnsignedLessExpr(
            left.getSignificand(packedExists), right.getSignificand(packedExists)
        ),
        negativeCaseSignificandComparison = mkBvUnsignedLessExpr(
            right.getSignificand(packedExists), left.getSignificand(packedExists)
        )
    )
}

internal fun <Fp : KFpSort> KContext.lessOrEqual(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>,
): KExpr<KBoolSort> {
    val infCase = (left.isInf and right.isInf and (left.isNegative eq right.isNegative)) or
            left.isNegativeInfinity or right.isPositiveInfinity

    val zeroCase = (left.isZero and right.isZero) or
            (left.isZero and right.isNegative.not()) or (left.isNegative and right.isZero)

    val packedExists = left.packedFp.hasPackedFp() && right.packedFp.hasPackedFp()

    return lessHelper(
        left,
        right,
        infCase,
        zeroCase,
        packedExists,
        positiveCaseSignificandComparison = mkBvUnsignedLessOrEqualExpr(
            left.getSignificand(packedExists), right.getSignificand(packedExists)
        ),
        negativeCaseSignificandComparison = mkBvUnsignedLessOrEqualExpr(
            right.getSignificand(packedExists), left.getSignificand(packedExists)
        ),
    )
}

// common logic for less and lessOrEqual
@Suppress("LongParameterList")
private fun <Fp : KFpSort> KContext.lessHelper(
    left: UnpackedFp<Fp>,
    right: UnpackedFp<Fp>,
    infCase: KExpr<KBoolSort>,
    zeroCase: KExpr<KBoolSort>,
    packedExists: Boolean,
    positiveCaseSignificandComparison: KExpr<KBoolSort>,
    negativeCaseSignificandComparison: KExpr<KBoolSort>,
): KExpr<KBoolSort> {
    val neitherNan = !left.isNaN and !right.isNaN

    // Infinities are bigger than everything but themselves
    val eitherInf = left.isInf or right.isInf

    // Both zero are equal
    val eitherZero = left.isZero or right.isZero

    val exponentLessExpr = { a: KExpr<KBvSort>, b: KExpr<KBvSort> ->
        if (packedExists)
            mkBvUnsignedLessExpr(a, b)
        else
            mkBvSignedLessExpr(a, b)
    }
    // Normal and subnormal
    val negativeLessThanPositive = left.isNegative and !right.isNegative
    val exponentEqual = left.getExponent(packedExists) eq right.getExponent(packedExists)

    val leftExponentLess = exponentLessExpr(
        left.getExponent(packedExists), right.getExponent(packedExists)
    )
    val positiveCase = !left.isNegative and !right.isNegative and
            (leftExponentLess or (exponentEqual and positiveCaseSignificandComparison))

    val rightExponentLess = exponentLessExpr(
        right.getExponent(packedExists), left.getExponent(packedExists)
    )
    val negativeCase = left.isNegative and right.isNegative and
            (rightExponentLess or (exponentEqual and negativeCaseSignificandComparison))

    return neitherNan and mkIte(
        eitherInf, infCase, mkIte(
            eitherZero, zeroCase, negativeLessThanPositive or positiveCase or negativeCase
        )
    )
}

internal fun <Fp : KFpSort> KContext.greater(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>,
): KExpr<KBoolSort> = less(right, left)

internal fun <Fp : KFpSort> KContext.greaterOrEqual(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>,
) = lessOrEqual(right, left)

internal fun <Fp : KFpSort> KContext.equal(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>,
): KExpr<KBoolSort> {
    // All comparison with NaN are false
    val neitherNan = !left.isNaN and !right.isNaN

    val bothZero = left.isZero and right.isZero
    val neitherZero = !left.isZero and !right.isZero

    val flagsAndExponent = neitherNan and (bothZero or
            (neitherZero and (left.isInf eq right.isInf and (left.sign eq right.sign)
                    and (left.unbiasedExponent eq right.unbiasedExponent))))

    if (left.packedFp is PackedFp && right.packedFp is PackedFp) {
        return neitherNan and (bothZero or (left.packedFp eq right.packedFp))
    }

    return flagsAndExponent and (left.normalizedSignificand eq right.normalizedSignificand)
}

internal fun <Fp : KFpSort> KContext.min(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>,
): KExpr<Fp> = iteOp(right.isNaN or less(left, right), left, right)

internal fun <Fp : KFpSort> KContext.max(
    left: UnpackedFp<Fp>, right: UnpackedFp<Fp>,
): KExpr<Fp> = iteOp(right.isNaN or greater(left, right), left, right)
