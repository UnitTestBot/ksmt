package io.ksmt.expr.rewrite.simplify

import io.ksmt.expr.KBvToFpExpr
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFpAbsExpr
import io.ksmt.expr.KFpAddExpr
import io.ksmt.expr.KFpDivExpr
import io.ksmt.expr.KFpEqualExpr
import io.ksmt.expr.KFpFromBvExpr
import io.ksmt.expr.KFpFusedMulAddExpr
import io.ksmt.expr.KFpGreaterExpr
import io.ksmt.expr.KFpGreaterOrEqualExpr
import io.ksmt.expr.KFpIsInfiniteExpr
import io.ksmt.expr.KFpIsNaNExpr
import io.ksmt.expr.KFpIsNegativeExpr
import io.ksmt.expr.KFpIsNormalExpr
import io.ksmt.expr.KFpIsPositiveExpr
import io.ksmt.expr.KFpIsSubnormalExpr
import io.ksmt.expr.KFpIsZeroExpr
import io.ksmt.expr.KFpLessExpr
import io.ksmt.expr.KFpLessOrEqualExpr
import io.ksmt.expr.KFpMaxExpr
import io.ksmt.expr.KFpMinExpr
import io.ksmt.expr.KFpMulExpr
import io.ksmt.expr.KFpNegationExpr
import io.ksmt.expr.KFpRemExpr
import io.ksmt.expr.KFpRoundToIntegralExpr
import io.ksmt.expr.KFpSqrtExpr
import io.ksmt.expr.KFpSubExpr
import io.ksmt.expr.KFpToBvExpr
import io.ksmt.expr.KFpToFpExpr
import io.ksmt.expr.KFpToIEEEBvExpr
import io.ksmt.expr.KFpToRealExpr
import io.ksmt.expr.KFpValue
import io.ksmt.expr.KRealToFpExpr
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KRealSort
import io.ksmt.utils.FpUtils.fpStructurallyEqual

@Suppress("ForbiddenComment")
interface KFpExprSimplifier : KExprSimplifierBase {

    /**
     * Simplify an expression of the form (= a b).
     * For the simplification of (fp.eq a b) see [KFpEqualExpr].
     * */
    fun <T : KFpSort> simplifyEqFp(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> = with(ctx) {
        if (lhs == rhs) return trueExpr

        if (lhs is KFpValue<T> && rhs is KFpValue<T>) {
            return fpStructurallyEqual(lhs, rhs).expr
        }

        return withExpressionsOrdered(lhs, rhs, ::mkEqNoSimplify)
    }

    fun <T : KFpSort> areDefinitelyDistinctFp(lhs: KExpr<T>, rhs: KExpr<T>): Boolean {
        if (lhs is KFpValue<T> && rhs is KFpValue<T>) {
            return !fpStructurallyEqual(lhs, rhs)
        }
        return false
    }

    override fun <T : KFpSort> transform(expr: KFpAbsExpr<T>): KExpr<T> = simplifyExpr(expr, expr.value) { arg ->
        simplifyFpAbsExpr(arg)
    }

    override fun <T : KFpSort> transform(expr: KFpNegationExpr<T>): KExpr<T> = simplifyExpr(expr, expr.value) { arg ->
        simplifyFpNegationExpr(arg)
    }

    override fun <T : KFpSort> transform(expr: KFpAddExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.roundingMode, expr.arg0, expr.arg1) { rm, lhs, rhs ->
            simplifyFpAddExpr(rm, lhs, rhs)
        }

    // a - b ==> a + (-b)
    override fun <T : KFpSort> transform(expr: KFpSubExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            preprocess = {
                KFpAddExpr(this, expr.roundingMode, expr.arg0, KFpNegationExpr(this, expr.arg1))
            }
        )

    override fun <T : KFpSort> transform(expr: KFpMulExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.roundingMode, expr.arg0, expr.arg1) { rm, lhs, rhs ->
            simplifyFpMulExpr(rm, lhs, rhs)
        }

    override fun <T : KFpSort> transform(expr: KFpDivExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.roundingMode, expr.arg0, expr.arg1) { rm, lhs, rhs ->
            simplifyFpDivExpr(rm, lhs, rhs)
        }

    override fun <T : KFpSort> transform(expr: KFpFusedMulAddExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.roundingMode, expr.arg0, expr.arg1, expr.arg2) { rm, a0, a1, a2 ->
            simplifyFpFusedMulAddExpr(rm, a0, a1, a2)
        }

    override fun <T : KFpSort> transform(expr: KFpSqrtExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.roundingMode, expr.value) { rm, value ->
            simplifyFpSqrtExpr(rm, value)
        }

    override fun <T : KFpSort> transform(expr: KFpRoundToIntegralExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.roundingMode, expr.value) { rm, value ->
            simplifyFpRoundToIntegralExpr(rm, value)
        }

    override fun <T : KFpSort> transform(expr: KFpRemExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            simplifyFpRemExpr(lhs, rhs)
        }

    override fun <T : KFpSort> transform(expr: KFpMinExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            simplifyFpMinExpr(lhs, rhs)
        }

    override fun <T : KFpSort> transform(expr: KFpMaxExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            simplifyFpMaxExpr(lhs, rhs)
        }

    override fun <T : KFpSort> transform(expr: KFpLessOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            simplifyFpLessOrEqualExpr(lhs, rhs)
        }

    override fun <T : KFpSort> transform(expr: KFpLessExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            simplifyFpLessExpr(lhs, rhs)
        }

    override fun <T : KFpSort> transform(expr: KFpGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            preprocess = { KFpLessOrEqualExpr(this, expr.arg1, expr.arg0) }
        )

    override fun <T : KFpSort> transform(expr: KFpGreaterExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            preprocess = { KFpLessExpr(this, expr.arg1, expr.arg0) }
        )

    override fun <T : KFpSort> transform(expr: KFpEqualExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            simplifyFpEqualExpr(lhs, rhs)
        }

    override fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.value) { arg ->
            simplifyFpIsNormalExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.value) { arg ->
            simplifyFpIsSubnormalExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.value) { arg ->
            simplifyFpIsZeroExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.value) { arg ->
            simplifyFpIsInfiniteExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.value) { arg ->
            simplifyFpIsNaNExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.value) { arg ->
            simplifyFpIsNegativeExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.value) { arg ->
            simplifyFpIsPositiveExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpFromBvExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.sign, expr.biasedExponent, expr.significand) { sign, exp, significand ->
            simplifyFpFromBvExpr(sign, exp, significand)
        }

    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> =
        simplifyExpr(expr, expr.value) { arg ->
            simplifyFpToIEEEBvExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpToFpExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.roundingMode, expr.value) { rm, value ->
            simplifyFpToFpExpr(expr.sort, rm, value)
        }

    override fun <T : KFpSort> transform(expr: KRealToFpExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.roundingMode, expr.value) { rm, value ->
            simplifyRealToFpExpr(expr.sort, rm, value)
        }

    override fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort> =
        simplifyExpr(expr, expr.value) { arg ->
            simplifyFpToRealExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KBvToFpExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.roundingMode, expr.value) { rm, value ->
            simplifyBvToFpExpr(expr.sort, rm, value, expr.signed)
        }

    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort> =
        simplifyExpr(expr, expr.roundingMode, expr.value) { rm, value ->
            simplifyFpToBvExpr(rm, value, expr.bvSize, expr.isSigned)
        }
}
