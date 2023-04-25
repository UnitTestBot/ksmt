package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KBitVec1Value
import io.ksmt.expr.KBitVecValue
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFp32Value
import io.ksmt.expr.KFp64Value
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
import io.ksmt.utils.BvUtils
import io.ksmt.utils.FpUtils
import io.ksmt.utils.FpUtils.isInfinity
import io.ksmt.utils.FpUtils.isNaN
import io.ksmt.utils.FpUtils.isNegative
import io.ksmt.utils.FpUtils.isNormal
import io.ksmt.utils.FpUtils.isPositive
import io.ksmt.utils.FpUtils.isSubnormal
import io.ksmt.utils.FpUtils.isZero
import io.ksmt.utils.uncheckedCast
import kotlin.math.IEEErem

inline fun <T : KFpSort> KContext.simplifyFpAbsExprLight(
    value: KExpr<T>,
    cont: (KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (value is KFpValue<T>) {
        // (abs NaN) ==> NaN
        if (value.isNaN()) {
            return value
        }

        return if (value.isNegative()) {
            // (abs x), x < 0 ==> -x
            FpUtils.fpNegate(value).uncheckedCast()
        } else {
            // (abs x), x >= 0 ==> x
            value
        }
    }

    return cont(value)
}

inline fun <T : KFpSort> KContext.simplifyFpNegationExprLight(
    value: KExpr<T>,
    cont: (KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (value is KFpValue<T>) {
        return FpUtils.fpNegate(value).uncheckedCast()
    }

    // (- -x) ==> x
    if (value is KFpNegationExpr<T>) {
        return value.value
    }

    return cont(value)
}

inline fun <T : KFpSort> KContext.simplifyFpAddExprLight(
    roundingMode: KExpr<KFpRoundingModeSort>,
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<KFpRoundingModeSort>, KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> = evalBinaryOpOr(roundingMode, lhs, rhs, FpUtils::fpAdd) {
    cont(roundingMode, lhs, rhs)
}

/** a - b ==> a + (-b) */
inline fun <T : KFpSort> KContext.rewriteFpSubExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteFpNegationExpr: KContext.(KExpr<T>) -> KExpr<T>,
    rewriteFpAddExpr: KContext.(KExpr<KFpRoundingModeSort>, KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> = rewriteFpAddExpr(roundingMode, lhs, rewriteFpNegationExpr(rhs))

inline fun <T : KFpSort> KContext.simplifyFpMulExprLight(
    roundingMode: KExpr<KFpRoundingModeSort>,
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<KFpRoundingModeSort>, KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> = evalBinaryOpOr(roundingMode, lhs, rhs, FpUtils::fpMul) {
    cont(roundingMode, lhs, rhs)
}

inline fun <T : KFpSort> KContext.simplifyFpDivExprLight(
    roundingMode: KExpr<KFpRoundingModeSort>,
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<KFpRoundingModeSort>, KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> = evalBinaryOpOr(roundingMode, lhs, rhs, FpUtils::fpDiv) {
    cont(roundingMode, lhs, rhs)
}

inline fun <T : KFpSort> KContext.simplifyFpRemExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (lhs is KFpValue<T> && rhs is KFpValue<T>) {
        val result = tryEvalFpRem(lhs, rhs)
        result?.let { return it.uncheckedCast() }
    }
    return cont(lhs, rhs)
}

@Suppress("ComplexCondition")
inline fun <T : KFpSort> KContext.simplifyFpFusedMulAddExprLight(
    roundingMode: KExpr<KFpRoundingModeSort>,
    arg0: KExpr<T>,
    arg1: KExpr<T>,
    arg2: KExpr<T>,
    cont: (KExpr<KFpRoundingModeSort>, KExpr<T>, KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (roundingMode is KFpRoundingModeExpr && arg0 is KFpValue<T> && arg1 is KFpValue<T> && arg2 is KFpValue<T>) {
        val result = tryEvalFpFma(roundingMode.value, arg0, arg1, arg2)
        result?.let { return it.uncheckedCast() }
    }
    return cont(roundingMode, arg0, arg1, arg2)
}

inline fun <T : KFpSort> KContext.simplifyFpSqrtExprLight(
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<T>,
    cont: (KExpr<KFpRoundingModeSort>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (value is KFpValue<T> && roundingMode is KFpRoundingModeExpr) {
        val result = FpUtils.fpSqrt(roundingMode.value, value)
        return result.uncheckedCast()
    }
    return cont(roundingMode, value)
}

inline fun <T : KFpSort> KContext.simplifyFpRoundToIntegralExprLight(
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<T>,
    cont: (KExpr<KFpRoundingModeSort>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (value is KFpValue<T> && roundingMode is KFpRoundingModeExpr) {
        val result = FpUtils.fpRoundToIntegral(roundingMode.value, value)
        return result.uncheckedCast()
    }
    return cont(roundingMode, value)
}

inline fun <T : KFpSort> KContext.simplifyFpFromBvExprLight(
    sign: KExpr<KBv1Sort>,
    biasedExponent: KExpr<out KBvSort>,
    significand: KExpr<out KBvSort>,
    cont: (KExpr<KBv1Sort>, KExpr<out KBvSort>, KExpr<out KBvSort>) -> KExpr<T>
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
    return cont(sign, biasedExponent, significand)
}

inline fun <T : KFpSort> KContext.simplifyFpToIEEEBvExprLight(
    arg: KExpr<T>,
    cont: (KExpr<T>) -> KExpr<KBvSort>
): KExpr<KBvSort> {
    if (arg is KFpValue<T>) {
        // ensure NaN bits are always same
        val normalizedValue = if (arg.isNaN()) {
            mkFpNaN(arg.sort)
        } else {
            arg
        }
        return BvUtils.concatBv(
            mkBv(normalizedValue.signBit),
            BvUtils.concatBv(
                normalizedValue.biasedExponent,
                normalizedValue.significand
            )
        ).uncheckedCast()
    }
    return cont(arg)
}

inline fun <T : KFpSort> KContext.simplifyFpToFpExprLight(
    sort: T,
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<out KFpSort>,
    cont: (T, KExpr<KFpRoundingModeSort>, KExpr<out KFpSort>) -> KExpr<T>
): KExpr<T> =
    if (roundingMode is KFpRoundingModeExpr && value is KFpValue<*>) {
        FpUtils.fpToFp(roundingMode.value, value, sort)
    } else {
        cont(sort, roundingMode, value)
    }

inline fun <T : KFpSort> KContext.simplifyFpToBvExprLight(
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<T>,
    bvSize: Int,
    isSigned: Boolean,
    cont: (KExpr<KFpRoundingModeSort>, KExpr<T>, Int, Boolean) -> KExpr<KBvSort>
): KExpr<KBvSort> {
    if (roundingMode is KFpRoundingModeExpr && value is KFpValue<T>) {
        val sort = mkBvSort(bvSize.toUInt())
        val result = FpUtils.fpBvValueOrNull(value, roundingMode.value, sort, isSigned)
        result?.let { return it }
    }
    return cont(roundingMode, value, bvSize, isSigned)
}

inline fun <T : KFpSort> KContext.simplifyBvToFpExprLight(
    sort: T,
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<KBvSort>,
    signed: Boolean,
    cont: (T, KExpr<KFpRoundingModeSort>, KExpr<KBvSort>, Boolean) -> KExpr<T>
): KExpr<T> =
    if (roundingMode is KFpRoundingModeExpr && value is KBitVecValue<*>) {
        FpUtils.fpValueFromBv(roundingMode.value, value, signed, sort)
    } else {
        cont(sort, roundingMode, value, signed)
    }

inline fun <T : KFpSort> KContext.simplifyFpToRealExprLight(
    arg: KExpr<T>,
    cont: (KExpr<T>) -> KExpr<KRealSort>
): KExpr<KRealSort> {
    if (arg is KFpValue<T>) {
        val result = FpUtils.fpRealValueOrNull(arg)
        result?.let { return it }
    }
    return cont(arg)
}

inline fun <T : KFpSort> KContext.simplifyRealToFpExprLight(
    sort: T,
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<KRealSort>,
    cont: (T, KExpr<KFpRoundingModeSort>, KExpr<KRealSort>) -> KExpr<T>
): KExpr<T> =
    if (roundingMode is KFpRoundingModeExpr && value is KRealNumExpr) {
        FpUtils.fpValueFromReal(roundingMode.value, value, sort)
    } else {
        cont(sort, roundingMode, value)
    }

inline fun <T : KFpSort> KContext.simplifyFpEqualExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> =
    if (lhs is KFpValue<T> && rhs is KFpValue<T>) {
        FpUtils.fpEq(lhs, rhs).expr
    } else {
        cont(lhs, rhs)
    }

inline fun <T : KFpSort> KContext.simplifyFpLessExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = when {
    lhs is KFpValue<T> && rhs is KFpValue<T> -> FpUtils.fpLt(lhs, rhs).expr
    lhs is KFpValue<T> && lhs.isNaN() -> falseExpr
    rhs is KFpValue<T> && rhs.isNaN() -> falseExpr
    lhs is KFpValue<T> && lhs.isInfinity() && lhs.isPositive() -> falseExpr
    rhs is KFpValue<T> && rhs.isInfinity() && rhs.isNegative() -> falseExpr
    else -> cont(lhs, rhs)
}

inline fun <T : KFpSort> KContext.simplifyFpLessOrEqualExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = when {
    lhs is KFpValue<T> && rhs is KFpValue<T> -> FpUtils.fpLeq(lhs, rhs).expr
    lhs is KFpValue<T> && lhs.isNaN() -> falseExpr
    rhs is KFpValue<T> && rhs.isNaN() -> falseExpr
    else -> cont(lhs, rhs)
}

inline fun <T : KFpSort> KContext.rewriteFpGreaterExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteFpLessExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = rewriteFpLessExpr(rhs, lhs)

inline fun <T : KFpSort> KContext.rewriteFpGreaterOrEqualExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteFpLessOrEqualExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = rewriteFpLessOrEqualExpr(rhs, lhs)

inline fun <T : KFpSort> KContext.simplifyFpMaxExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (lhs is KFpValue<T> && lhs.isNaN()) {
        return rhs
    }

    if (rhs is KFpValue<T> && rhs.isNaN()) {
        return lhs
    }

    if (lhs is KFpValue<T> && rhs is KFpValue<T>) {
        if (!lhs.isZero() || !rhs.isZero() || lhs.signBit == rhs.signBit) {
            return FpUtils.fpMax(lhs, rhs).uncheckedCast()
        }
    }

    return cont(lhs, rhs)
}

inline fun <T : KFpSort> KContext.simplifyFpMinExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (lhs is KFpValue<T> && lhs.isNaN()) {
        return rhs
    }

    if (rhs is KFpValue<T> && rhs.isNaN()) {
        return lhs
    }

    if (lhs is KFpValue<T> && rhs is KFpValue<T>) {
        if (!lhs.isZero() || !rhs.isZero() || lhs.signBit == rhs.signBit) {
            return FpUtils.fpMin(lhs, rhs).uncheckedCast()
        }
    }

    return cont(lhs, rhs)
}

inline fun <T : KFpSort> KContext.simplifyFpIsInfiniteExprLight(
    arg: KExpr<T>,
    cont: (KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = evalFpPredicateOr(arg, { it.isInfinity() }) { cont(it) }

inline fun <T : KFpSort> KContext.simplifyFpIsNaNExprLight(
    arg: KExpr<T>,
    cont: (KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = evalFpPredicateOr(arg, { it.isNaN() }) { cont(it) }

inline fun <T : KFpSort> KContext.simplifyFpIsNegativeExprLight(
    arg: KExpr<T>,
    cont: (KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = evalFpPredicateOr(arg, { !it.isNaN() && it.isNegative() }) { cont(it) }

inline fun <T : KFpSort> KContext.simplifyFpIsNormalExprLight(
    arg: KExpr<T>,
    cont: (KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = evalFpPredicateOr(arg, { it.isNormal() }) { cont(it) }

inline fun <T : KFpSort> KContext.simplifyFpIsPositiveExprLight(
    arg: KExpr<T>,
    cont: (KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = evalFpPredicateOr(arg, { !it.isNaN() && it.isPositive() }) { cont(it) }

inline fun <T : KFpSort> KContext.simplifyFpIsSubnormalExprLight(
    arg: KExpr<T>,
    cont: (KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = evalFpPredicateOr(arg, { it.isSubnormal() }) { cont(it) }

inline fun <T : KFpSort> KContext.simplifyFpIsZeroExprLight(
    arg: KExpr<T>,
    cont: (KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = evalFpPredicateOr(arg, { it.isZero() }) { cont(it) }

inline fun <T : KFpSort> KContext.simplifyEqFpLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> {
    if (lhs == rhs) return trueExpr

    if (lhs is KFpValue<T> && rhs is KFpValue<T>) {
        return FpUtils.fpStructurallyEqual(lhs, rhs).expr
    }

    return cont(lhs, rhs)
}

inline fun <T : KFpSort> KContext.evalFpPredicateOr(
    value: KExpr<T>,
    predicate: (KFpValue<T>) -> Boolean,
    cont: (KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> {
    if (value is KFpValue<T>) {
        return predicate(value).expr
    }
    return cont(value)
}

@Suppress("ForbiddenComment")
fun KContext.tryEvalFpRem(lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*>? = when {
    lhs is KFp32Value -> mkFp(lhs.value.IEEErem((rhs as KFp32Value).value), lhs.sort)
    lhs is KFp64Value -> mkFp(lhs.value.IEEErem((rhs as KFp64Value).value), lhs.sort)
    lhs.isNaN() || rhs.isNaN() -> mkFpNaN(lhs.sort)
    lhs.isInfinity() -> mkFpNaN(lhs.sort)
    rhs.isInfinity() -> lhs
    rhs.isZero() -> mkFpNaN(lhs.sort)
    lhs.isZero() -> lhs
    // todo: eval fp rem
    else -> null
}

// Eval x * y + z
@Suppress("ComplexMethod", "ForbiddenComment")
fun KContext.tryEvalFpFma(
    rm: KFpRoundingMode,
    x: KFpValue<*>,
    y: KFpValue<*>,
    z: KFpValue<*>
): KFpValue<*>? = when {
    x.isNaN() || y.isNaN() || z.isNaN() -> mkFpNaN(x.sort)

    x.isInfinity() && x.isPositive() -> when {
        y.isZero() -> mkFpNaN(x.sort)
        z.isInfinity() && (x.signBit xor y.signBit xor z.signBit) -> mkFpNaN(x.sort)
        else -> mkFpInf(y.signBit, x.sort)
    }

    y.isInfinity() && y.isPositive() -> when {
        x.isZero() -> mkFpNaN(x.sort)
        z.isInfinity() && (x.signBit xor y.signBit xor z.signBit) -> mkFpNaN(x.sort)
        else -> mkFpInf(x.signBit, x.sort)
    }

    x.isInfinity() && x.isNegative() -> when {
        y.isZero() -> mkFpNaN(x.sort)
        z.isInfinity() && (x.signBit xor y.signBit xor z.signBit) -> mkFpNaN(x.sort)
        else -> mkFpInf(!y.signBit, x.sort)
    }

    y.isInfinity() && y.isNegative() -> when {
        x.isZero() -> mkFpNaN(x.sort)
        z.isInfinity() && (x.signBit xor y.signBit xor z.signBit) -> mkFpNaN(x.sort)
        else -> mkFpInf(!x.signBit, x.sort)
    }

    z.isInfinity() -> z

    x.isZero() || y.isZero() -> if (z.isZero() && (x.signBit xor y.signBit xor z.signBit)) {
        mkFpZero(signBit = rm == KFpRoundingMode.RoundTowardNegative, sort = x.sort)
    } else {
        z
    }

    // todo: eval fp fma
    else -> null
}

inline fun <T : KFpSort> evalBinaryOpOr(
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