package org.ksmt.expr.rewrite.simplify

import org.ksmt.expr.KBvToFpExpr
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFpAbsExpr
import org.ksmt.expr.KFpAddExpr
import org.ksmt.expr.KFpDivExpr
import org.ksmt.expr.KFpEqualExpr
import org.ksmt.expr.KFpFromBvExpr
import org.ksmt.expr.KFpFusedMulAddExpr
import org.ksmt.expr.KFpGreaterExpr
import org.ksmt.expr.KFpGreaterOrEqualExpr
import org.ksmt.expr.KFpIsInfiniteExpr
import org.ksmt.expr.KFpIsNaNExpr
import org.ksmt.expr.KFpIsNegativeExpr
import org.ksmt.expr.KFpIsNormalExpr
import org.ksmt.expr.KFpIsPositiveExpr
import org.ksmt.expr.KFpIsSubnormalExpr
import org.ksmt.expr.KFpIsZeroExpr
import org.ksmt.expr.KFpLessExpr
import org.ksmt.expr.KFpLessOrEqualExpr
import org.ksmt.expr.KFpMaxExpr
import org.ksmt.expr.KFpMinExpr
import org.ksmt.expr.KFpMulExpr
import org.ksmt.expr.KFpNegationExpr
import org.ksmt.expr.KFpRemExpr
import org.ksmt.expr.KFpRoundToIntegralExpr
import org.ksmt.expr.KFpSqrtExpr
import org.ksmt.expr.KFpSubExpr
import org.ksmt.expr.KFpToBvExpr
import org.ksmt.expr.KFpToFpExpr
import org.ksmt.expr.KFpToIEEEBvExpr
import org.ksmt.expr.KFpToRealExpr
import org.ksmt.expr.KFpValue
import org.ksmt.expr.KRealToFpExpr
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KRealSort
import org.ksmt.utils.FpUtils.fpStructurallyEqual

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

        return mkEqNoSimplify(lhs, rhs)
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
