package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KRealSort

fun <T : KFpSort> KContext.simplifyFpAbsExpr(value: KExpr<T>): KExpr<T> = mkFpAbsExprNoSimplify(value)

fun <T : KFpSort> KContext.simplifyFpNegationExpr(value: KExpr<T>): KExpr<T> = mkFpNegationExprNoSimplify(value)

fun <T : KFpSort> KContext.simplifyFpAddExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    lhs: KExpr<T>,
    rhs: KExpr<T>
): KExpr<T> = mkFpAddExprNoSimplify(roundingMode, lhs, rhs)

fun <T : KFpSort> KContext.simplifyFpSubExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    lhs: KExpr<T>,
    rhs: KExpr<T>
): KExpr<T> = mkFpSubExprNoSimplify(roundingMode, lhs, rhs)

fun <T : KFpSort> KContext.simplifyFpMulExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    lhs: KExpr<T>,
    rhs: KExpr<T>
): KExpr<T> = mkFpMulExprNoSimplify(roundingMode, lhs, rhs)

fun <T : KFpSort> KContext.simplifyFpDivExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    lhs: KExpr<T>,
    rhs: KExpr<T>
): KExpr<T> = mkFpDivExprNoSimplify(roundingMode, lhs, rhs)

fun <T : KFpSort> KContext.simplifyFpRemExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> = mkFpRemExprNoSimplify(lhs, rhs)

fun <T : KFpSort> KContext.simplifyFpFusedMulAddExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    arg0: KExpr<T>,
    arg1: KExpr<T>,
    arg2: KExpr<T>
): KExpr<T> = mkFpFusedMulAddExprNoSimplify(roundingMode, arg0, arg1, arg2)

fun <T : KFpSort> KContext.simplifyFpSqrtExpr(roundingMode: KExpr<KFpRoundingModeSort>, value: KExpr<T>): KExpr<T> =
    mkFpSqrtExprNoSimplify(roundingMode, value)

fun <T : KFpSort> KContext.simplifyFpRoundToIntegralExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<T>
): KExpr<T> = mkFpRoundToIntegralExprNoSimplify(roundingMode, value)


fun <T : KFpSort> KContext.simplifyFpFromBvExpr(
    sign: KExpr<KBv1Sort>,
    biasedExponent: KExpr<out KBvSort>,
    significand: KExpr<out KBvSort>
): KExpr<T> = mkFpFromBvExprNoSimplify(sign, biasedExponent, significand)

fun <T : KFpSort> KContext.simplifyFpToIEEEBvExpr(arg: KExpr<T>): KExpr<KBvSort> = mkFpToIEEEBvExprNoSimplify(arg)

fun <T : KFpSort> KContext.simplifyFpToFpExpr(
    sort: T,
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<out KFpSort>
): KExpr<T> = mkFpToFpExprNoSimplify(sort, roundingMode, value)

fun <T : KFpSort> KContext.simplifyFpToBvExpr(
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<T>,
    bvSize: Int,
    isSigned: Boolean
): KExpr<KBvSort> = mkFpToBvExprNoSimplify(roundingMode, value, bvSize, isSigned)

fun <T : KFpSort> KContext.simplifyBvToFpExpr(
    sort: T,
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<KBvSort>,
    signed: Boolean
): KExpr<T> = mkBvToFpExprNoSimplify(sort, roundingMode, value, signed)

fun <T : KFpSort> KContext.simplifyFpToRealExpr(arg: KExpr<T>): KExpr<KRealSort> = mkFpToRealExprNoSimplify(arg)

fun <T : KFpSort> KContext.simplifyRealToFpExpr(
    sort: T,
    roundingMode: KExpr<KFpRoundingModeSort>,
    value: KExpr<KRealSort>
): KExpr<T> = mkRealToFpExprNoSimplify(sort, roundingMode, value)


fun <T : KFpSort> KContext.simplifyFpEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    mkFpEqualExprNoSimplify(lhs, rhs)

fun <T : KFpSort> KContext.simplifyFpGreaterExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    mkFpGreaterExprNoSimplify(lhs, rhs)

fun <T : KFpSort> KContext.simplifyFpGreaterOrEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    mkFpGreaterOrEqualExprNoSimplify(lhs, rhs)

fun <T : KFpSort> KContext.simplifyFpLessExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    mkFpLessExprNoSimplify(lhs, rhs)

fun <T : KFpSort> KContext.simplifyFpLessOrEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    mkFpLessOrEqualExprNoSimplify(lhs, rhs)

fun <T : KFpSort> KContext.simplifyFpMaxExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> = mkFpMaxExprNoSimplify(lhs, rhs)

fun <T : KFpSort> KContext.simplifyFpMinExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> = mkFpMinExprNoSimplify(lhs, rhs)


fun <T : KFpSort> KContext.simplifyFpIsInfiniteExpr(arg: KExpr<T>): KExpr<KBoolSort> = mkFpIsInfiniteExprNoSimplify(arg)

fun <T : KFpSort> KContext.simplifyFpIsNaNExpr(arg: KExpr<T>): KExpr<KBoolSort> = mkFpIsNaNExprNoSimplify(arg)

fun <T : KFpSort> KContext.simplifyFpIsNegativeExpr(arg: KExpr<T>): KExpr<KBoolSort> = mkFpIsNegativeExprNoSimplify(arg)

fun <T : KFpSort> KContext.simplifyFpIsNormalExpr(arg: KExpr<T>): KExpr<KBoolSort> = mkFpIsNormalExprNoSimplify(arg)

fun <T : KFpSort> KContext.simplifyFpIsPositiveExpr(arg: KExpr<T>): KExpr<KBoolSort> = mkFpIsPositiveExprNoSimplify(arg)

fun <T : KFpSort> KContext.simplifyFpIsSubnormalExpr(arg: KExpr<T>): KExpr<KBoolSort> =
    mkFpIsSubnormalExprNoSimplify(arg)

fun <T : KFpSort> KContext.simplifyFpIsZeroExpr(arg: KExpr<T>): KExpr<KBoolSort> = mkFpIsZeroExprNoSimplify(arg)
