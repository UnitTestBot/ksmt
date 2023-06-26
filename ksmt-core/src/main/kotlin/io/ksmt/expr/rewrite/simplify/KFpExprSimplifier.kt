package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
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
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpRoundingModeSort
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
        simplifyEqFpLight(lhs, rhs) { lhs2, rhs2 ->
            withExpressionsOrdered(lhs2, rhs2, ::mkEqNoSimplify)
        }
    }

    fun <T : KFpSort> areDefinitelyDistinctFp(lhs: KExpr<T>, rhs: KExpr<T>): Boolean {
        if (lhs is KFpValue<T> && rhs is KFpValue<T>) {
            return !fpStructurallyEqual(lhs, rhs)
        }
        return false
    }

    fun <T : KFpSort> KContext.preprocess(expr: KFpAbsExpr<T>): KExpr<T> = expr
    fun <T : KFpSort> KContext.postRewriteFpAbsExpr(value: KExpr<T>): KExpr<T> =
        simplifyFpAbsExpr(value)

    override fun <T : KFpSort> transform(expr: KFpAbsExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteFpAbsExpr(it) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpNegationExpr<T>): KExpr<T> = expr
    fun <T : KFpSort> KContext.postRewriteFpNegationExpr(value: KExpr<T>): KExpr<T> =
        simplifyFpNegationExpr(value)

    override fun <T : KFpSort> transform(expr: KFpNegationExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteFpNegationExpr(it) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpAddExpr<T>): KExpr<T> = expr

    fun <T : KFpSort> KContext.postRewriteFpAddExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        lhs: KExpr<T>,
        rhs: KExpr<T>
    ): KExpr<T> = simplifyFpAddExpr(roundingMode, lhs, rhs)

    override fun <T : KFpSort> transform(expr: KFpAddExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.roundingMode,
            a1 = expr.arg0,
            a2 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { rm, l, r -> postRewriteFpAddExpr(rm, l, r) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpSubExpr<T>): KExpr<T> =
        rewriteFpSubExpr(
            roundingMode = expr.roundingMode,
            lhs = expr.arg0,
            rhs = expr.arg1,
            rewriteFpNegationExpr = { KFpNegationExpr(this, it) },
            rewriteFpAddExpr = { rm, l, r -> KFpAddExpr(this, rm, l, r) }
        )

    fun <T : KFpSort> KContext.postRewriteFpSubExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        lhs: KExpr<T>,
        rhs: KExpr<T>
    ): KExpr<T> = error("Always preprocessed")

    override fun <T : KFpSort> transform(expr: KFpSubExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.roundingMode,
            a1 = expr.arg0,
            a2 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { rm, l, r -> postRewriteFpSubExpr(rm, l, r) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpMulExpr<T>): KExpr<T> = expr

    fun <T : KFpSort> KContext.postRewriteFpMulExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        lhs: KExpr<T>,
        rhs: KExpr<T>
    ): KExpr<T> = simplifyFpMulExpr(roundingMode, lhs, rhs)

    override fun <T : KFpSort> transform(expr: KFpMulExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.roundingMode,
            a1 = expr.arg0,
            a2 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { rm, l, r -> postRewriteFpMulExpr(rm, l, r) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpDivExpr<T>): KExpr<T> = expr

    fun <T : KFpSort> KContext.postRewriteFpDivExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        lhs: KExpr<T>,
        rhs: KExpr<T>
    ): KExpr<T> = simplifyFpDivExpr(roundingMode, lhs, rhs)

    override fun <T : KFpSort> transform(expr: KFpDivExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.roundingMode,
            a1 = expr.arg0,
            a2 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { rm, l, r -> postRewriteFpDivExpr(rm, l, r) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpFusedMulAddExpr<T>): KExpr<T> = expr

    fun <T : KFpSort> KContext.postRewriteFpFusedMulAddExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        arg0: KExpr<T>,
        arg1: KExpr<T>,
        arg2: KExpr<T>
    ): KExpr<T> = simplifyFpFusedMulAddExpr(roundingMode, arg0, arg1, arg2)

    override fun <T : KFpSort> transform(expr: KFpFusedMulAddExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.roundingMode,
            a1 = expr.arg0,
            a2 = expr.arg1,
            a3 = expr.arg2,
            preprocess = { preprocess(it) },
            simplifier = { rm, a0, a1, a2 -> postRewriteFpFusedMulAddExpr(rm, a0, a1, a2) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpSqrtExpr<T>): KExpr<T> = expr

    fun <T : KFpSort> KContext.postRewriteFpSqrtExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<T>
    ): KExpr<T> = simplifyFpSqrtExpr(roundingMode, value)

    override fun <T : KFpSort> transform(expr: KFpSqrtExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.roundingMode,
            a1 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { rm, v -> postRewriteFpSqrtExpr(rm, v) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpRoundToIntegralExpr<T>): KExpr<T> = expr

    fun <T : KFpSort> KContext.postRewriteFpRoundToIntegralExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<T>
    ): KExpr<T> = simplifyFpRoundToIntegralExpr(roundingMode, value)

    override fun <T : KFpSort> transform(expr: KFpRoundToIntegralExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.roundingMode,
            a1 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { rm, v -> postRewriteFpRoundToIntegralExpr(rm, v) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpRemExpr<T>): KExpr<T> = expr

    fun <T : KFpSort> KContext.postRewriteFpRemExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
        simplifyFpRemExpr(lhs, rhs)

    override fun <T : KFpSort> transform(expr: KFpRemExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteFpRemExpr(l, r) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpMinExpr<T>): KExpr<T> = expr

    fun <T : KFpSort> KContext.postRewriteFpMinExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
        simplifyFpMinExpr(lhs, rhs)

    override fun <T : KFpSort> transform(expr: KFpMinExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteFpMinExpr(l, r) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpMaxExpr<T>): KExpr<T> = expr

    fun <T : KFpSort> KContext.postRewriteFpMaxExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
        simplifyFpMaxExpr(lhs, rhs)

    override fun <T : KFpSort> transform(expr: KFpMaxExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteFpMaxExpr(l, r) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpLessOrEqualExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KFpSort> KContext.postRewriteFpLessOrEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        simplifyFpLessOrEqualExpr(lhs, rhs)

    override fun <T : KFpSort> transform(expr: KFpLessOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteFpLessOrEqualExpr(l, r) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpLessExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KFpSort> KContext.postRewriteFpLessExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        simplifyFpLessExpr(lhs, rhs)

    override fun <T : KFpSort> transform(expr: KFpLessExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteFpLessExpr(l, r) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        rewriteFpGreaterOrEqualExpr(
            lhs = expr.arg0,
            rhs = expr.arg1,
            rewriteFpLessOrEqualExpr = { l, r -> KFpLessOrEqualExpr(this, l, r) }
        )

    fun <T : KFpSort> KContext.postRewriteFpGreaterOrEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        error("Always preprocessed")

    override fun <T : KFpSort> transform(expr: KFpGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteFpGreaterOrEqualExpr(l, r) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpGreaterExpr<T>): KExpr<KBoolSort> =
        rewriteFpGreaterExpr(
            lhs = expr.arg0,
            rhs = expr.arg1,
            rewriteFpLessExpr = { l, r -> KFpLessExpr(this, l, r) }
        )

    fun <T : KFpSort> KContext.postRewriteFpGreaterExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        error("Always preprocessed")

    override fun <T : KFpSort> transform(expr: KFpGreaterExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteFpGreaterExpr(l, r) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpEqualExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KFpSort> KContext.postRewriteFpEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        simplifyFpEqualExpr(lhs, rhs)

    override fun <T : KFpSort> transform(expr: KFpEqualExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteFpEqualExpr(l, r) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpIsNormalExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KFpSort> KContext.postRewriteFpIsNormalExpr(arg: KExpr<T>): KExpr<KBoolSort> =
        simplifyFpIsNormalExpr(arg)

    override fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteFpIsNormalExpr(it) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpIsSubnormalExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KFpSort> KContext.postRewriteFpIsSubnormalExpr(arg: KExpr<T>): KExpr<KBoolSort> =
        simplifyFpIsSubnormalExpr(arg)

    override fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteFpIsSubnormalExpr(it) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpIsZeroExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KFpSort> KContext.postRewriteFpIsZeroExpr(arg: KExpr<T>): KExpr<KBoolSort> =
        simplifyFpIsZeroExpr(arg)

    override fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteFpIsZeroExpr(it) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpIsInfiniteExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KFpSort> KContext.postRewriteFpIsInfiniteExpr(arg: KExpr<T>): KExpr<KBoolSort> =
        simplifyFpIsInfiniteExpr(arg)

    override fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteFpIsInfiniteExpr(it) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpIsNaNExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KFpSort> KContext.postRewriteFpIsNaNExpr(arg: KExpr<T>): KExpr<KBoolSort> =
        simplifyFpIsNaNExpr(arg)

    override fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteFpIsNaNExpr(it) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpIsNegativeExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KFpSort> KContext.postRewriteFpIsNegativeExpr(arg: KExpr<T>): KExpr<KBoolSort> =
        simplifyFpIsNegativeExpr(arg)

    override fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteFpIsNegativeExpr(it) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpIsPositiveExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KFpSort> KContext. postRewriteFpIsPositiveExpr(arg: KExpr<T>): KExpr<KBoolSort> =
        simplifyFpIsPositiveExpr(arg)

    override fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteFpIsPositiveExpr(it) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpFromBvExpr<T>): KExpr<T> = expr

    fun <T : KFpSort> KContext.postRewriteFpFromBvExpr(
        sign: KExpr<KBv1Sort>,
        biasedExponent: KExpr<out KBvSort>,
        significand: KExpr<out KBvSort>
    ): KExpr<T> = simplifyFpFromBvExpr(sign, biasedExponent, significand)

    override fun <T : KFpSort> transform(expr: KFpFromBvExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.sign,
            a1 = expr.biasedExponent,
            a2 = expr.significand,
            preprocess = { preprocess(it) },
            simplifier = { sign, exp, significant -> postRewriteFpFromBvExpr(sign, exp, significant) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> = expr

    fun <T : KFpSort> KContext.postRewriteFpToIEEEBvExpr(arg: KExpr<T>): KExpr<KBvSort> =
        simplifyFpToIEEEBvExpr(arg)

    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteFpToIEEEBvExpr(it) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpToFpExpr<T>): KExpr<T> = expr

    fun <T : KFpSort> KContext.postRewriteFpToFpExpr(
        sort: T,
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<out KFpSort>
    ): KExpr<T> = simplifyFpToFpExpr(sort, roundingMode, value)

    override fun <T : KFpSort> transform(expr: KFpToFpExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.roundingMode,
            a1 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { rm, v -> postRewriteFpToFpExpr(expr.sort, rm, v) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KRealToFpExpr<T>): KExpr<T> = expr

    fun <T : KFpSort> KContext.postRewriteRealToFpExpr(
        sort: T,
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<KRealSort>
    ): KExpr<T> = simplifyRealToFpExpr(sort, roundingMode, value)

    override fun <T : KFpSort> transform(expr: KRealToFpExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.roundingMode,
            a1 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { rm, v -> postRewriteRealToFpExpr(expr.sort, rm, v) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpToRealExpr<T>): KExpr<KRealSort> = expr

    fun <T : KFpSort> KContext.postRewriteFpToRealExpr(arg: KExpr<T>): KExpr<KRealSort> =
        simplifyFpToRealExpr(arg)

    override fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteFpToRealExpr(it) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KBvToFpExpr<T>): KExpr<T> = expr

    fun <T : KFpSort> KContext.postRewriteBvToFpExpr(
        sort: T,
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<KBvSort>,
        signed: Boolean
    ): KExpr<T> = simplifyBvToFpExpr(sort, roundingMode, value, signed)

    override fun <T : KFpSort> transform(expr: KBvToFpExpr<T>): KExpr<T> =
        simplifyExpr(
            expr,
            expr.roundingMode,
            expr.value,
            { preprocess(it) },
            { rm, v -> postRewriteBvToFpExpr(expr.sort, rm, v, expr.signed) }
        )

    fun <T : KFpSort> KContext.preprocess(expr: KFpToBvExpr<T>): KExpr<KBvSort> = expr

    fun <T : KFpSort> KContext.postRewriteFpToBvExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<T>,
        bvSize: Int,
        isSigned: Boolean
    ): KExpr<KBvSort> = simplifyFpToBvExpr(roundingMode, value, bvSize, isSigned)

    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort> =
        simplifyExpr(
            expr,
            expr.roundingMode,
            expr.value,
            { preprocess(it) },
            { rm, v -> postRewriteFpToBvExpr(rm, v, expr.bvSize, expr.isSigned) }
        )
}
