package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KBitVec1Value
import io.ksmt.expr.KBitVecValue
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFpNegationExpr
import io.ksmt.expr.KFpRoundingMode
import io.ksmt.expr.KFpRoundingModeExpr
import io.ksmt.expr.KFpValue
import io.ksmt.expr.KRealNumExpr
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KRealSort
import io.ksmt.utils.FpUtils.fpAdd
import io.ksmt.utils.FpUtils.fpBvValueOrNull
import io.ksmt.utils.FpUtils.fpDiv
import io.ksmt.utils.FpUtils.fpEq
import io.ksmt.utils.FpUtils.fpFma
import io.ksmt.utils.FpUtils.fpLeq
import io.ksmt.utils.FpUtils.fpLt
import io.ksmt.utils.FpUtils.fpMax
import io.ksmt.utils.FpUtils.fpMin
import io.ksmt.utils.FpUtils.fpMul
import io.ksmt.utils.FpUtils.fpNegate
import io.ksmt.utils.FpUtils.fpRealValueOrNull
import io.ksmt.utils.FpUtils.fpRem
import io.ksmt.utils.FpUtils.fpRoundToIntegral
import io.ksmt.utils.FpUtils.fpSqrt
import io.ksmt.utils.FpUtils.fpToFp
import io.ksmt.utils.FpUtils.fpValueFromBv
import io.ksmt.utils.FpUtils.fpValueFromReal
import io.ksmt.utils.FpUtils.isInfinity
import io.ksmt.utils.FpUtils.isNaN
import io.ksmt.utils.FpUtils.isNegative
import io.ksmt.utils.FpUtils.isNormal
import io.ksmt.utils.FpUtils.isPositive
import io.ksmt.utils.FpUtils.isSubnormal
import io.ksmt.utils.FpUtils.isZero
import io.ksmt.utils.uncheckedCast

fun <T : KFpSort> KContext.simplifyFpAbsExpr(value: KExpr<T>): KExpr<T> {
    if (value is KFpValue<T>) {
        // (abs NaN) ==> NaN
        if (value.isNaN()) {
            return value
        }

        return if (value.isNegative()) {
            // (abs x), x < 0 ==> -x
            fpNegate(value).uncheckedCast()
        } else {
            // (abs x), x >= 0 ==> x
            value
        }
    }
    return mkFpAbsExprNoSimplify(value)
}

fun <T : KFpSort> KContext.simplifyFpNegationExpr(value: KExpr<T>): KExpr<T> {
    if (value is KFpValue<T>) {
        return fpNegate(value).uncheckedCast()
    }

    // (- -x) ==> x
    if (value is KFpNegationExpr<T>) {
        return value.value
    }

    return mkFpNegationExprNoSimplify(value)
}

fun <T : KFpSort> KContext.simplifyFpAddExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    lhs: KExpr<T>,
    rhs: KExpr<T>
): KExpr<T> = evalBinaryOpOr(roundingMode, lhs, rhs, ::fpAdd) {
    mkFpAddExprNoSimplify(roundingMode, lhs, rhs)
}

// a - b ==> a + (-b)
fun <T : KFpSort> KContext.simplifyFpSubExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    lhs: KExpr<T>,
    rhs: KExpr<T>
): KExpr<T> = simplifyFpAddExpr(roundingMode, lhs, simplifyFpNegationExpr(rhs))

fun <T : KFpSort> KContext.simplifyFpMulExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    lhs: KExpr<T>,
    rhs: KExpr<T>
): KExpr<T> = evalBinaryOpOr(roundingMode, lhs, rhs, ::fpMul) {
    mkFpMulExprNoSimplify(roundingMode, lhs, rhs)
}

fun <T : KFpSort> KContext.simplifyFpDivExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    lhs: KExpr<T>,
    rhs: KExpr<T>
): KExpr<T> = evalBinaryOpOr(roundingMode, lhs, rhs, ::fpDiv) {
    mkFpDivExprNoSimplify(roundingMode, lhs, rhs)
}

fun <T : KFpSort> KContext.simplifyFpRemExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> {
    if (lhs is KFpValue<T> && rhs is KFpValue<T>) {
        val result = fpRem(lhs, rhs)
        return result.uncheckedCast()
    }
    return mkFpRemExprNoSimplify(lhs, rhs)
}

@Suppress("ComplexCondition")
fun <T : KFpSort> KContext.simplifyFpFusedMulAddExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    arg0: KExpr<T>,
    arg1: KExpr<T>,
    arg2: KExpr<T>
): KExpr<T> {
    if (roundingMode is KFpRoundingModeExpr && arg0 is KFpValue<T> && arg1 is KFpValue<T> && arg2 is KFpValue<T>) {
        val result = fpFma(roundingMode.value, arg0, arg1, arg2)
        return result.uncheckedCast()
    }
    return mkFpFusedMulAddExprNoSimplify(roundingMode, arg0, arg1, arg2)
}

fun <T : KFpSort> KContext.simplifyFpSqrtExpr(roundingMode: KExpr<KFpRoundingModeSort>, value: KExpr<T>): KExpr<T> {
    if (value is KFpValue<T> && roundingMode is KFpRoundingModeExpr) {
        val result = fpSqrt(roundingMode.value, value)
        return result.uncheckedCast()
    }
    return mkFpSqrtExprNoSimplify(roundingMode, value)
}

fun <T : KFpSort> KContext.simplifyFpRoundToIntegralExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<T>
): KExpr<T> {
    if (value is KFpValue<T> && roundingMode is KFpRoundingModeExpr) {
        val result = fpRoundToIntegral(roundingMode.value, value)
        return result.uncheckedCast()
    }
    return mkFpRoundToIntegralExprNoSimplify(roundingMode, value)
}


fun <T : KFpSort> KContext.simplifyFpFromBvExpr(
    sign: KExpr<KBv1Sort>,
    biasedExponent: KExpr<out KBvSort>,
    significand: KExpr<out KBvSort>
): KExpr<T> {
    if (sign is KBitVec1Value && biasedExponent is KBitVecValue<*> && significand is KBitVecValue<*>) {
        val exponentBits = biasedExponent.sort.sizeBits
        // +1 it required since bv doesn't contain `hidden bit`
        val significandBits = significand.sort.sizeBits + 1u
        val sort = mkFpSort(exponentBits, significandBits)

        return mkFpBiased(
            sort = sort,
            biasedExponent = biasedExponent,
            significand = significand,
            signBit = sign.value
        ).uncheckedCast()
    }
    return mkFpFromBvExprNoSimplify(sign, biasedExponent, significand)
}

fun <T : KFpSort> KContext.simplifyFpToIEEEBvExpr(arg: KExpr<T>): KExpr<KBvSort> {
    if (arg is KFpValue<T>) {
        // ensure NaN bits are always same
        val normalizedValue = if (arg.isNaN()) {
            mkFpNaN(arg.sort)
        } else {
            arg
        }
        return simplifyBvConcatExpr(
            mkBv(normalizedValue.signBit),
            simplifyBvConcatExpr(
                normalizedValue.biasedExponent,
                normalizedValue.significand
            )
        )
    }
    return mkFpToIEEEBvExprNoSimplify(arg)
}

fun <T : KFpSort> KContext.simplifyFpToFpExpr(
    sort: T,
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<out KFpSort>
): KExpr<T> {
    if (roundingMode is KFpRoundingModeExpr && value is KFpValue<*>) {
        return fpToFp(roundingMode.value, value, sort)
    }
    return mkFpToFpExprNoSimplify(sort, roundingMode, value)
}

fun <T : KFpSort> KContext.simplifyFpToBvExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<T>,
    bvSize: Int,
    isSigned: Boolean
): KExpr<KBvSort> {
    if (roundingMode is KFpRoundingModeExpr && value is KFpValue<T>) {
        val sort = mkBvSort(bvSize.toUInt())
        val result = fpBvValueOrNull(value, roundingMode.value, sort, isSigned)
        result?.let { return it }
    }
    return mkFpToBvExprNoSimplify(roundingMode, value, bvSize, isSigned)
}

fun <T : KFpSort> KContext.simplifyBvToFpExpr(
    sort: T,
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<KBvSort>,
    signed: Boolean
): KExpr<T> {
    if (roundingMode is KFpRoundingModeExpr && value is KBitVecValue<*>) {
        return fpValueFromBv(roundingMode.value, value, signed, sort)
    }

    return mkBvToFpExprNoSimplify(sort, roundingMode, value, signed)
}

fun <T : KFpSort> KContext.simplifyFpToRealExpr(arg: KExpr<T>): KExpr<KRealSort> {
    if (arg is KFpValue<T>) {
        val result = fpRealValueOrNull(arg)
        result?.let { return it }
    }
    return mkFpToRealExprNoSimplify(arg)
}

fun <T : KFpSort> KContext.simplifyRealToFpExpr(
    sort: T,
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<KRealSort>
): KExpr<T> {
    if (roundingMode is KFpRoundingModeExpr && value is KRealNumExpr) {
        return fpValueFromReal(roundingMode.value, value, sort)
    }
    return mkRealToFpExprNoSimplify(sort, roundingMode, value)
}


fun <T : KFpSort> KContext.simplifyFpEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> {
    if (lhs is KFpValue<T> && rhs is KFpValue<T>) {
        return fpEq(lhs, rhs).expr
    }
    return mkFpEqualExprNoSimplify(lhs, rhs)
}

fun <T : KFpSort> KContext.simplifyFpLessExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> = when {
    lhs is KFpValue<T> && rhs is KFpValue<T> -> fpLt(lhs, rhs).expr
    lhs is KFpValue<T> && lhs.isNaN() -> falseExpr
    rhs is KFpValue<T> && rhs.isNaN() -> falseExpr
    lhs is KFpValue<T> && lhs.isInfinity() && lhs.isPositive() -> falseExpr
    rhs is KFpValue<T> && rhs.isInfinity() && rhs.isNegative() -> falseExpr
    else -> mkFpLessExprNoSimplify(lhs, rhs)
}

fun <T : KFpSort> KContext.simplifyFpLessOrEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> = when {
    lhs is KFpValue<T> && rhs is KFpValue<T> -> fpLeq(lhs, rhs).expr
    lhs is KFpValue<T> && lhs.isNaN() -> falseExpr
    rhs is KFpValue<T> && rhs.isNaN() -> falseExpr
    else -> mkFpLessOrEqualExprNoSimplify(lhs, rhs)
}

fun <T : KFpSort> KContext.simplifyFpGreaterExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    simplifyFpLessExpr(rhs, lhs)

fun <T : KFpSort> KContext.simplifyFpGreaterOrEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    simplifyFpLessOrEqualExpr(rhs, lhs)

fun <T : KFpSort> KContext.simplifyFpMaxExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> {
    if (lhs is KFpValue<T> && lhs.isNaN()) {
        return rhs
    }

    if (rhs is KFpValue<T> && rhs.isNaN()) {
        return lhs
    }

    if (lhs is KFpValue<T> && rhs is KFpValue<T>) {
        if (!lhs.isZero() || !rhs.isZero() || lhs.signBit == rhs.signBit) {
            return fpMax(lhs, rhs).uncheckedCast()
        }
    }

    return mkFpMaxExprNoSimplify(lhs, rhs)
}

fun <T : KFpSort> KContext.simplifyFpMinExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> {
    if (lhs is KFpValue<T> && lhs.isNaN()) {
        return rhs
    }

    if (rhs is KFpValue<T> && rhs.isNaN()) {
        return lhs
    }

    if (lhs is KFpValue<T> && rhs is KFpValue<T>) {
        if (!lhs.isZero() || !rhs.isZero() || lhs.signBit == rhs.signBit) {
            return fpMin(lhs, rhs).uncheckedCast()
        }
    }

    return mkFpMinExprNoSimplify(lhs, rhs)
}


fun <T : KFpSort> KContext.simplifyFpIsInfiniteExpr(arg: KExpr<T>): KExpr<KBoolSort> =
    evalFpPredicateOr(arg, { it.isInfinity() }) { mkFpIsInfiniteExprNoSimplify(arg) }

fun <T : KFpSort> KContext.simplifyFpIsNaNExpr(arg: KExpr<T>): KExpr<KBoolSort> =
    evalFpPredicateOr(arg, { it.isNaN() }) { mkFpIsNaNExprNoSimplify(arg) }

fun <T : KFpSort> KContext.simplifyFpIsNegativeExpr(arg: KExpr<T>): KExpr<KBoolSort> =
    evalFpPredicateOr(arg, { !it.isNaN() && it.isNegative() }) { mkFpIsNegativeExprNoSimplify(arg) }

fun <T : KFpSort> KContext.simplifyFpIsNormalExpr(arg: KExpr<T>): KExpr<KBoolSort> =
    evalFpPredicateOr(arg, { it.isNormal() }) { mkFpIsNormalExprNoSimplify(arg) }

fun <T : KFpSort> KContext.simplifyFpIsPositiveExpr(arg: KExpr<T>): KExpr<KBoolSort> =
    evalFpPredicateOr(arg, { !it.isNaN() && it.isPositive() }) { mkFpIsPositiveExprNoSimplify(arg) }

fun <T : KFpSort> KContext.simplifyFpIsSubnormalExpr(arg: KExpr<T>): KExpr<KBoolSort> =
    evalFpPredicateOr(arg, { it.isSubnormal() }) { mkFpIsSubnormalExprNoSimplify(arg) }

fun <T : KFpSort> KContext.simplifyFpIsZeroExpr(arg: KExpr<T>): KExpr<KBoolSort> =
    evalFpPredicateOr(arg, { it.isZero() }) { mkFpIsZeroExprNoSimplify(arg) }

private inline fun <T : KFpSort> evalBinaryOpOr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    operation: (KFpRoundingMode, KFpValue<*>, KFpValue<*>) -> KFpValue<*>,
    default: () -> KExpr<T>
): KExpr<T> {
    if (lhs is KFpValue<T> && rhs is KFpValue<T> && roundingMode is KFpRoundingModeExpr) {
        val result = operation(roundingMode.value, lhs, rhs)
        return result.uncheckedCast()
    }
    return default()
}

private inline fun <T : KFpSort> KContext.evalFpPredicateOr(
    value: KExpr<T>,
    predicate: (KFpValue<T>) -> Boolean,
    default: () -> KExpr<KBoolSort>
): KExpr<KBoolSort> {
    if (value is KFpValue<T>) {
        return predicate(value).expr
    }
    return default()
}
