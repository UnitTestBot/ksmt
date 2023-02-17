package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.expr.KApp
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
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.utils.FpUtils.fpStructurallyEqual
import org.ksmt.utils.uncheckedCast

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

    override fun <T : KFpSort> transform(expr: KFpAbsExpr<T>): KExpr<T> = simplifyApp(expr) { (arg) ->
        simplifyFpAbsExpr(arg)
    }

    override fun <T : KFpSort> transform(expr: KFpNegationExpr<T>): KExpr<T> = simplifyApp(expr) { (arg) ->
        simplifyFpNegationExpr(arg)
    }

    override fun <T : KFpSort> transform(expr: KFpAddExpr<T>): KExpr<T> = expr.simplifyFpBinaryOp { rm, lhs, rhs ->
        simplifyFpAddExpr(rm, lhs, rhs)
    }

    // a - b ==> a + (-b)
    override fun <T : KFpSort> transform(expr: KFpSubExpr<T>): KExpr<T> =
        simplifyApp(
            expr = expr,
            preprocess = {
                KFpAddExpr(this, expr.roundingMode, expr.arg0, KFpNegationExpr(this, expr.arg1))
            }
        ) {
            error("Always preprocessed")
        }

    override fun <T : KFpSort> transform(expr: KFpMulExpr<T>): KExpr<T> = expr.simplifyFpBinaryOp { rm, lhs, rhs ->
        simplifyFpMulExpr(rm, lhs, rhs)
    }

    override fun <T : KFpSort> transform(expr: KFpDivExpr<T>): KExpr<T> = expr.simplifyFpBinaryOp { rm, lhs, rhs ->
        simplifyFpDivExpr(rm, lhs, rhs)
    }

    override fun <T : KFpSort> transform(expr: KFpFusedMulAddExpr<T>): KExpr<T> =
        expr.simplifyFpTernaryOp { rm, a0, a1, a2 ->
            simplifyFpFusedMulAddExpr(rm, a0, a1, a2)
        }

    override fun <T : KFpSort> transform(expr: KFpSqrtExpr<T>): KExpr<T> =
        expr.simplifyFpUnaryOp { rm, arg ->
            simplifyFpSqrtExpr(rm, arg)
        }

    override fun <T : KFpSort> transform(expr: KFpRoundToIntegralExpr<T>): KExpr<T> =
        expr.simplifyFpUnaryOp { rm, arg ->
            simplifyFpRoundToIntegralExpr(rm, arg)
        }

    override fun <T : KFpSort> transform(expr: KFpRemExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        simplifyFpRemExpr(lhs, rhs)
    }

    override fun <T : KFpSort> transform(expr: KFpMinExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        simplifyFpMinExpr(lhs, rhs)
    }

    override fun <T : KFpSort> transform(expr: KFpMaxExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        simplifyFpMaxExpr(lhs, rhs)
    }

    override fun <T : KFpSort> transform(expr: KFpLessOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            simplifyFpLessOrEqualExpr(lhs, rhs)
        }

    override fun <T : KFpSort> transform(expr: KFpLessExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            simplifyFpLessExpr(lhs, rhs)
        }

    override fun <T : KFpSort> transform(expr: KFpGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyApp(
            expr = expr,
            preprocess = { KFpLessOrEqualExpr(this, expr.arg1, expr.arg0) }
        ) {
            error("Always preprocessed")
        }

    override fun <T : KFpSort> transform(expr: KFpGreaterExpr<T>): KExpr<KBoolSort> =
        simplifyApp(
            expr = expr,
            preprocess = { KFpLessExpr(this, expr.arg1, expr.arg0) }
        ) {
            error("Always preprocessed")
        }

    override fun <T : KFpSort> transform(expr: KFpEqualExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            simplifyFpEqualExpr(lhs, rhs)
        }

    override fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            simplifyFpIsNormalExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            simplifyFpIsSubnormalExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            simplifyFpIsZeroExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            simplifyFpIsInfiniteExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            simplifyFpIsNaNExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            simplifyFpIsNegativeExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            simplifyFpIsPositiveExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpFromBvExpr<T>): KExpr<T> =
        simplifyApp(expr) { (sign, exp, significand) ->
            simplifyFpFromBvExpr(sign.uncheckedCast(), exp, significand)
        }

    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> = simplifyApp(expr) { (arg) ->
        simplifyFpToIEEEBvExpr(arg)
    }

    override fun <T : KFpSort> transform(expr: KFpToFpExpr<T>): KExpr<T> =
        simplifyApp(expr) { (rmArg, valueArg) ->
            val rm: KExpr<KFpRoundingModeSort> = rmArg.uncheckedCast()
            val value: KExpr<KFpSort> = valueArg.uncheckedCast()

            simplifyFpToFpExpr(expr.sort, rm, value)
        }

    override fun <T : KFpSort> transform(expr: KRealToFpExpr<T>): KExpr<T> =
        simplifyApp(expr) { (rmArg, valueArg) ->
            val rm: KExpr<KFpRoundingModeSort> = rmArg.uncheckedCast()
            val value: KExpr<KRealSort> = valueArg.uncheckedCast()

            simplifyRealToFpExpr(expr.sort, rm, value)
        }

    override fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort> = simplifyApp(expr) { (arg) ->
        simplifyFpToRealExpr(arg)
    }

    override fun <T : KFpSort> transform(expr: KBvToFpExpr<T>): KExpr<T> =
        simplifyApp(expr) { (rmArg, bvValueArg) ->
            val rm: KExpr<KFpRoundingModeSort> = rmArg.uncheckedCast()
            val value: KExpr<KBvSort> = bvValueArg.uncheckedCast()

            simplifyBvToFpExpr(expr.sort, rm, value, expr.signed)
        }

    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort> =
        simplifyApp(expr) { (rmArg, valueArg) ->
            val rm: KExpr<KFpRoundingModeSort> = rmArg.uncheckedCast()
            val value: KExpr<T> = valueArg.uncheckedCast()

            simplifyFpToBvExpr(rm, value, expr.bvSize, expr.isSigned)
        }

    private inline fun <T : KFpSort> KApp<T, KSort>.simplifyFpUnaryOp(
        crossinline simplifier: KContext.(KExpr<KFpRoundingModeSort>, KExpr<T>) -> KExpr<T>
    ): KExpr<T> = simplifyApp(this) { (rm, value) ->
        simplifier(ctx, rm.uncheckedCast(), value.uncheckedCast())
    }

    private inline fun <T : KFpSort> KApp<T, KSort>.simplifyFpBinaryOp(
        crossinline simplifier: KContext.(KExpr<KFpRoundingModeSort>, KExpr<T>, KExpr<T>) -> KExpr<T>
    ): KExpr<T> = simplifyApp(this) { (rm, lhs, rhs) ->
        simplifier(ctx, rm.uncheckedCast(), lhs.uncheckedCast(), rhs.uncheckedCast())
    }

    private inline fun <T : KFpSort> KApp<T, KSort>.simplifyFpTernaryOp(
        crossinline simplifier: KContext.(KExpr<KFpRoundingModeSort>, KExpr<T>, KExpr<T>, KExpr<T>) -> KExpr<T>
    ): KExpr<T> = simplifyApp(this) { (rm, a0, a1, a2) ->
        simplifier(ctx, rm.uncheckedCast(), a0.uncheckedCast(), a1.uncheckedCast(), a2.uncheckedCast())
    }
}
