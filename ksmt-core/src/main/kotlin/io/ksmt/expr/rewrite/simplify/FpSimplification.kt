package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KRealSort

fun <T : KFpSort> KContext.simplifyFpAbsExpr(value: KExpr<T>): KExpr<T> =
    simplifyFpAbsExprLight(value, ::mkFpAbsExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpNegationExpr(value: KExpr<T>): KExpr<T> =
    simplifyFpNegationExprLight(value, ::mkFpNegationExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpAddExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    lhs: KExpr<T>,
    rhs: KExpr<T>
): KExpr<T> = simplifyFpAddExprLight(roundingMode, lhs, rhs, ::mkFpAddExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpSubExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    lhs: KExpr<T>,
    rhs: KExpr<T>
): KExpr<T> = rewriteFpSubExpr(roundingMode, lhs, rhs, KContext::simplifyFpNegationExpr, KContext::simplifyFpAddExpr)

fun <T : KFpSort> KContext.simplifyFpMulExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    lhs: KExpr<T>,
    rhs: KExpr<T>
): KExpr<T> = simplifyFpMulExprLight(roundingMode, lhs, rhs, ::mkFpMulExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpDivExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    lhs: KExpr<T>,
    rhs: KExpr<T>
): KExpr<T> = simplifyFpDivExprLight(roundingMode, lhs, rhs, ::mkFpDivExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpRemExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyFpRemExprLight(lhs, rhs, ::mkFpRemExprNoSimplify)

@Suppress("ComplexCondition")
fun <T : KFpSort> KContext.simplifyFpFusedMulAddExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    arg0: KExpr<T>,
    arg1: KExpr<T>,
    arg2: KExpr<T>
): KExpr<T> = simplifyFpFusedMulAddExprLight(roundingMode, arg0, arg1, arg2, ::mkFpFusedMulAddExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpSqrtExpr(roundingMode: KExpr<KFpRoundingModeSort>, value: KExpr<T>): KExpr<T> =
    simplifyFpSqrtExprLight(roundingMode, value, ::mkFpSqrtExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpRoundToIntegralExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<T>
): KExpr<T> = simplifyFpRoundToIntegralExprLight(roundingMode, value, ::mkFpRoundToIntegralExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpFromBvExpr(
    sign: KExpr<KBv1Sort>,
    biasedExponent: KExpr<out KBvSort>,
    significand: KExpr<out KBvSort>
): KExpr<T> = simplifyFpFromBvExprLight(sign, biasedExponent, significand, ::mkFpFromBvExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpToIEEEBvExpr(arg: KExpr<T>): KExpr<KBvSort> =
    simplifyFpToIEEEBvExprLight(arg, ::mkFpToIEEEBvExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpToFpExpr(
    sort: T,
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<out KFpSort>
): KExpr<T> = simplifyFpToFpExprLight(sort, roundingMode, value, ::mkFpToFpExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpToBvExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<T>,
    bvSize: Int,
    isSigned: Boolean
): KExpr<KBvSort> = simplifyFpToBvExprLight(roundingMode, value, bvSize, isSigned, ::mkFpToBvExprNoSimplify)

fun <T : KFpSort> KContext.simplifyBvToFpExpr(
    sort: T,
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<KBvSort>,
    signed: Boolean
): KExpr<T> = simplifyBvToFpExprLight(sort, roundingMode, value, signed, ::mkBvToFpExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpToRealExpr(arg: KExpr<T>): KExpr<KRealSort> =
    simplifyFpToRealExprLight(arg, ::mkFpToRealExprNoSimplify)

fun <T : KFpSort> KContext.simplifyRealToFpExpr(
    sort: T,
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<KRealSort>
): KExpr<T> = simplifyRealToFpExprLight(sort, roundingMode, value, ::mkRealToFpExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    simplifyFpEqualExprLight(lhs, rhs, ::mkFpEqualExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpLessExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    simplifyFpLessExprLight(lhs, rhs, ::mkFpLessExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpLessOrEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    simplifyFpLessOrEqualExprLight(lhs, rhs, ::mkFpLessOrEqualExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpGreaterExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    rewriteFpGreaterExpr(lhs, rhs, KContext::simplifyFpLessExpr)

fun <T : KFpSort> KContext.simplifyFpGreaterOrEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    rewriteFpGreaterOrEqualExpr(lhs, rhs, KContext::simplifyFpLessOrEqualExpr)

fun <T : KFpSort> KContext.simplifyFpMaxExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyFpMaxExprLight(lhs, rhs, ::mkFpMaxExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpMinExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyFpMinExprLight(lhs, rhs, ::mkFpMinExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpIsInfiniteExpr(arg: KExpr<T>): KExpr<KBoolSort> =
    simplifyFpIsInfiniteExprLight(arg, ::mkFpIsInfiniteExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpIsNaNExpr(arg: KExpr<T>): KExpr<KBoolSort> =
    simplifyFpIsNaNExprLight(arg, ::mkFpIsNaNExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpIsNegativeExpr(arg: KExpr<T>): KExpr<KBoolSort> =
    simplifyFpIsNegativeExprLight(arg, ::mkFpIsNegativeExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpIsNormalExpr(arg: KExpr<T>): KExpr<KBoolSort> =
    simplifyFpIsNormalExprLight(arg, ::mkFpIsNormalExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpIsPositiveExpr(arg: KExpr<T>): KExpr<KBoolSort> =
    simplifyFpIsPositiveExprLight(arg, ::mkFpIsPositiveExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpIsSubnormalExpr(arg: KExpr<T>): KExpr<KBoolSort> =
    simplifyFpIsSubnormalExprLight(arg, ::mkFpIsSubnormalExprNoSimplify)

fun <T : KFpSort> KContext.simplifyFpIsZeroExpr(arg: KExpr<T>): KExpr<KBoolSort> =
    simplifyFpIsZeroExprLight(arg, ::mkFpIsZeroExprNoSimplify)
