package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KBitVec1Value
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KBvToFpExpr
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFp32Value
import org.ksmt.expr.KFp64Value
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
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.expr.KFpRoundingModeExpr
import org.ksmt.expr.KFpSqrtExpr
import org.ksmt.expr.KFpSubExpr
import org.ksmt.expr.KFpToBvExpr
import org.ksmt.expr.KFpToFpExpr
import org.ksmt.expr.KFpToIEEEBvExpr
import org.ksmt.expr.KFpToRealExpr
import org.ksmt.expr.KFpValue
import org.ksmt.expr.KRealNumExpr
import org.ksmt.expr.KRealToFpExpr
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.utils.FpUtils.fpAdd
import org.ksmt.utils.FpUtils.fpBvValueOrNull
import org.ksmt.utils.FpUtils.fpDiv
import org.ksmt.utils.FpUtils.fpEq
import org.ksmt.utils.FpUtils.fpLeq
import org.ksmt.utils.FpUtils.fpLt
import org.ksmt.utils.FpUtils.fpMax
import org.ksmt.utils.FpUtils.fpMin
import org.ksmt.utils.FpUtils.fpMul
import org.ksmt.utils.FpUtils.fpNegate
import org.ksmt.utils.FpUtils.fpRealValueOrNull
import org.ksmt.utils.FpUtils.fpRoundToIntegral
import org.ksmt.utils.FpUtils.fpSqrt
import org.ksmt.utils.FpUtils.fpStructurallyEqual
import org.ksmt.utils.FpUtils.fpToFp
import org.ksmt.utils.FpUtils.fpValueFromBv
import org.ksmt.utils.FpUtils.fpValueFromReal
import org.ksmt.utils.FpUtils.isSubnormal
import org.ksmt.utils.FpUtils.isInfinity
import org.ksmt.utils.FpUtils.isNan
import org.ksmt.utils.FpUtils.isNegative
import org.ksmt.utils.FpUtils.isNormal
import org.ksmt.utils.FpUtils.isPositive
import org.ksmt.utils.FpUtils.isZero
import org.ksmt.utils.uncheckedCast
import kotlin.math.IEEErem

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

        return mkEq(lhs, rhs)
    }

    fun <T : KFpSort> areDefinitelyDistinctFp(lhs: KExpr<T>, rhs: KExpr<T>): Boolean {
        if (lhs is KFpValue<T> && rhs is KFpValue<T>) {
            return !fpStructurallyEqual(lhs, rhs)
        }
        return false
    }

    override fun <T : KFpSort> transform(expr: KFpAbsExpr<T>): KExpr<T> = simplifyApp(expr) { (arg) ->
        if (arg is KFpValue<T>) {
            // (abs NaN) ==> Nan
            if (arg.isNan()) {
                return@simplifyApp arg
            }

            if (arg.isNegative()) {
                // (abs x), x < 0 ==> -x
                return@simplifyApp fpNegate(arg).uncheckedCast()
            } else {
                // (abs x), x >= 0 ==> x
                return@simplifyApp arg
            }
        }
        mkFpAbsExpr(arg)
    }

    override fun <T : KFpSort> transform(expr: KFpNegationExpr<T>): KExpr<T> = simplifyApp(expr) { (arg) ->
        if (arg is KFpValue<T>) {
            return@simplifyApp fpNegate(arg).uncheckedCast()
        }

        // (- -x) ==> x
        if (arg is KFpNegationExpr<T>) {
            return@simplifyApp arg.value
        }

        mkFpNegationExpr(arg)
    }

    override fun <T : KFpSort> transform(expr: KFpAddExpr<T>): KExpr<T> = expr.simplifyFpBinaryOp { rm, lhs, rhs ->
        if (lhs is KFpValue<T> && rhs is KFpValue<T> && rm is KFpRoundingModeExpr) {
            val result = fpAdd(rm.value, lhs, rhs)
            return@simplifyFpBinaryOp result.uncheckedCast()
        }

        mkFpAddExpr(rm, lhs, rhs)
    }

    // a - b ==> a + (-b)
    override fun <T : KFpSort> transform(expr: KFpSubExpr<T>): KExpr<T> =
        simplifyApp(
            expr = expr,
            preprocess = { mkFpAddExpr(expr.roundingMode, expr.arg0, mkFpNegationExpr(expr.arg1)) }
        ) {
            error("Always preprocessed")
        }

    override fun <T : KFpSort> transform(expr: KFpMulExpr<T>): KExpr<T> = expr.simplifyFpBinaryOp { rm, lhs, rhs ->
        if (lhs is KFpValue<T> && rhs is KFpValue<T> && rm is KFpRoundingModeExpr) {
            val result = fpMul(rm.value, lhs, rhs)
            return@simplifyFpBinaryOp result.uncheckedCast()
        }

        mkFpMulExpr(rm, lhs, rhs)
    }

    override fun <T : KFpSort> transform(expr: KFpDivExpr<T>): KExpr<T> = expr.simplifyFpBinaryOp { rm, lhs, rhs ->
        if (lhs is KFpValue<T> && rhs is KFpValue<T> && rm is KFpRoundingModeExpr) {
            val result = fpDiv(rm.value, lhs, rhs)
            return@simplifyFpBinaryOp result.uncheckedCast()
        }

        mkFpDivExpr(rm, lhs, rhs)
    }

    @Suppress("ComplexCondition")
    override fun <T : KFpSort> transform(expr: KFpFusedMulAddExpr<T>): KExpr<T> =
        expr.simplifyFpTernaryOp { rm, a0, a1, a2 ->
            if (rm is KFpRoundingModeExpr && a0 is KFpValue<T> && a1 is KFpValue<T> && a2 is KFpValue<T>) {
                val result = tryEvalFpFma(rm.value, a0, a1, a2)
                result?.let { return@simplifyFpTernaryOp it.uncheckedCast() }
            }
            mkFpFusedMulAddExpr(rm, a0, a1, a2)
        }

    override fun <T : KFpSort> transform(expr: KFpSqrtExpr<T>): KExpr<T> =
        expr.simplifyFpUnaryOp { rm, arg ->
            if (arg is KFpValue<T> && rm is KFpRoundingModeExpr) {
                val result = fpSqrt(rm.value, arg)
                return@simplifyFpUnaryOp result.uncheckedCast()
            }
            mkFpSqrtExpr(rm, arg)
        }

    override fun <T : KFpSort> transform(expr: KFpRoundToIntegralExpr<T>): KExpr<T> =
        expr.simplifyFpUnaryOp { rm, arg ->
            if (arg is KFpValue<T> && rm is KFpRoundingModeExpr) {
                val result = fpRoundToIntegral(rm.value, arg)
                return@simplifyFpUnaryOp result.uncheckedCast()
            }
            mkFpRoundToIntegralExpr(rm, arg)
        }

    override fun <T : KFpSort> transform(expr: KFpRemExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val lhsValue = lhs as? KFpValue<T>
        val rhsValue = rhs as? KFpValue<T>

        if (lhsValue != null && rhsValue != null) {
            val result = tryEvalFpRem(lhsValue, rhsValue)
            result?.let { return@simplifyApp it.uncheckedCast() }
        }

        mkFpRemExpr(lhs, rhs)
    }

    override fun <T : KFpSort> transform(expr: KFpMinExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val lhsValue = lhs as? KFpValue<T>
        val rhsValue = rhs as? KFpValue<T>

        if (lhsValue != null && lhsValue.isNan()) {
            return@simplifyApp rhs
        }

        if (rhsValue != null && rhsValue.isNan()) {
            return@simplifyApp lhs
        }

        if (lhsValue != null && rhsValue != null) {
            if (lhsValue.isZero() && rhsValue.isZero() && lhsValue.signBit != rhsValue.signBit) {
                return@simplifyApp mkFpMinExpr(lhs, rhs)
            }

            val result = fpMin(lhsValue, rhsValue)
            return@simplifyApp result.uncheckedCast()
        }

        mkFpMinExpr(lhs, rhs)
    }

    override fun <T : KFpSort> transform(expr: KFpMaxExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val lhsValue = lhs as? KFpValue<T>
        val rhsValue = rhs as? KFpValue<T>

        if (lhsValue != null && lhsValue.isNan()) {
            return@simplifyApp rhs
        }

        if (rhsValue != null && rhsValue.isNan()) {
            return@simplifyApp lhs
        }

        if (lhsValue != null && rhsValue != null) {
            if (lhsValue.isZero() && rhsValue.isZero() && lhsValue.signBit != rhsValue.signBit) {
                return@simplifyApp mkFpMaxExpr(lhs, rhs)
            }

            val result = fpMax(lhsValue, rhsValue)
            return@simplifyApp result.uncheckedCast()
        }

        mkFpMaxExpr(lhs, rhs)
    }

    @Suppress("ComplexCondition")
    override fun <T : KFpSort> transform(expr: KFpLessOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            val lhsValue = lhs as? KFpValue<T>
            val rhsValue = rhs as? KFpValue<T>

            if (lhsValue != null && lhsValue.isNan() || rhsValue != null && rhsValue.isNan()) {
                return@simplifyApp falseExpr
            }

            if (lhsValue != null && rhsValue != null) {
                return@simplifyApp fpLeq(lhsValue, rhsValue).expr
            }

            mkFpLessOrEqualExpr(lhs, rhs)
        }

    @Suppress("ComplexCondition")
    override fun <T : KFpSort> transform(expr: KFpLessExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            val lhsValue = lhs as? KFpValue<T>
            val rhsValue = rhs as? KFpValue<T>

            if (lhsValue != null && lhsValue.isNan() || rhsValue != null && rhsValue.isNan()) {
                return@simplifyApp falseExpr
            }

            if (lhsValue != null && lhsValue.isInfinity() && lhsValue.isPositive()) {
                return@simplifyApp falseExpr
            }

            if (rhsValue != null && rhsValue.isInfinity() && rhsValue.isNegative()) {
                return@simplifyApp falseExpr
            }

            if (lhsValue != null && rhsValue != null) {
                return@simplifyApp fpLt(lhsValue, rhsValue).expr
            }

            mkFpLessExpr(lhs, rhs)
        }

    override fun <T : KFpSort> transform(expr: KFpGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyApp(
            expr = expr,
            preprocess = { mkFpLessOrEqualExpr(expr.arg1, expr.arg0) }
        ) {
            error("Always preprocessed")
        }

    override fun <T : KFpSort> transform(expr: KFpGreaterExpr<T>): KExpr<KBoolSort> =
        simplifyApp(
            expr = expr,
            preprocess = { mkFpLessExpr(expr.arg1, expr.arg0) }
        ) {
            error("Always preprocessed")
        }

    override fun <T : KFpSort> transform(expr: KFpEqualExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            val lhsValue = lhs as? KFpValue<T>
            val rhsValue = rhs as? KFpValue<T>

            if (lhsValue != null && rhsValue != null) {
                return@simplifyApp fpEq(lhsValue, rhsValue).expr
            }

            mkFpEqualExpr(lhs, rhs)
        }

    override fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            if (arg is KFpValue<T>) {
                return@simplifyApp arg.isNormal().expr
            }
            mkFpIsNormalExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            if (arg is KFpValue<T>) {
                return@simplifyApp arg.isSubnormal().expr
            }
            mkFpIsSubnormalExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            if (arg is KFpValue<T>) {
                return@simplifyApp arg.isZero().expr
            }
            mkFpIsZeroExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            if (arg is KFpValue<T>) {
                return@simplifyApp arg.isInfinity().expr
            }
            mkFpIsInfiniteExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            if (arg is KFpValue<T>) {
                return@simplifyApp arg.isNan().expr
            }
            mkFpIsNaNExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            if (arg is KFpValue<T>) {
                return@simplifyApp (!arg.isNan() && arg.isNegative()).expr
            }
            mkFpIsNegativeExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            if (arg is KFpValue<T>) {
                return@simplifyApp (!arg.isNan() && arg.isPositive()).expr
            }
            mkFpIsPositiveExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpFromBvExpr<T>): KExpr<T> =
        simplifyApp(expr) { (sign, exp, significand) ->
            if (sign is KBitVecValue<*> && exp is KBitVecValue<*> && significand is KBitVecValue<*>) {
                return@simplifyApp mkFpBiased(
                    sort = expr.sort,
                    biasedExponent = exp,
                    significand = significand,
                    signBit = (sign as KBitVec1Value).value
                )
            }
            mkFpFromBvExpr(sign.uncheckedCast(), exp, significand)
        }

    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> = simplifyApp(expr) { (arg) ->
        if (arg is KFpValue<T>) {
            if (arg.isNan()) {
                // ensure NaN bits are the same, as in KContext
                val nan = ctx.mkFpNan(arg.sort) as KFpValue<*>
                return@simplifyApp rewrite(
                    mkBvConcatExpr(
                        mkBv(nan.signBit),
                        mkBvConcatExpr(nan.biasedExponent, nan.significand)
                    )
                )
            }
            return@simplifyApp rewrite(
                mkBvConcatExpr(
                    mkBv(arg.signBit),
                    mkBvConcatExpr(arg.biasedExponent, arg.significand)
                )
            )
        }
        mkFpToIEEEBvExpr(arg)
    }

    override fun <T : KFpSort> transform(expr: KFpToFpExpr<T>): KExpr<T> =
        simplifyApp(expr) { (rmArg, valueArg) ->
            val rm: KExpr<KFpRoundingModeSort> = rmArg.uncheckedCast()
            val value: KExpr<KFpSort> = valueArg.uncheckedCast()

            if (rm is KFpRoundingModeExpr && value is KFpValue<*>) {
                return@simplifyApp fpToFp(rm.value, value, expr.sort)
            }

            mkFpToFpExpr(expr.sort, rm, value)
        }

    override fun <T : KFpSort> transform(expr: KRealToFpExpr<T>): KExpr<T> =
        simplifyApp(expr) { (rmArg, valueArg) ->
            val rm: KExpr<KFpRoundingModeSort> = rmArg.uncheckedCast()
            val value: KExpr<KRealSort> = valueArg.uncheckedCast()

            if (rm is KFpRoundingModeExpr && value is KRealNumExpr) {
                return@simplifyApp fpValueFromReal(rm.value, value, expr.sort)
            }

            mkRealToFpExpr(expr.sort, rm, value)
        }

    override fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort> = simplifyApp(expr) { (arg) ->
        if (arg is KFpValue<T>) {
            val result = fpRealValueOrNull(arg)
            if (result != null) {
                return@simplifyApp result
            }
        }
        mkFpToRealExpr(arg)
    }

    override fun <T : KFpSort> transform(expr: KBvToFpExpr<T>): KExpr<T> =
        simplifyApp(expr) { (rmArg, bvValueArg) ->
            val rm: KExpr<KFpRoundingModeSort> = rmArg.uncheckedCast()
            val value: KExpr<KBvSort> = bvValueArg.uncheckedCast()

            if (rm is KFpRoundingModeExpr && value is KBitVecValue<*>) {
                return@simplifyApp fpValueFromBv(rm.value, value, expr.signed, expr.sort)
            }

            mkBvToFpExpr(expr.sort, rm, value, expr.signed)
        }

    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort> =
        simplifyApp(expr) { (rmArg, valueArg) ->
            val rm: KExpr<KFpRoundingModeSort> = rmArg.uncheckedCast()
            val value: KExpr<T> = valueArg.uncheckedCast()

            if (rm is KFpRoundingModeExpr && value is KFpValue<T>) {
                val result = fpBvValueOrNull(value, rm.value, expr.sort, expr.isSigned)
                if (result != null) {
                    return@simplifyApp result
                }
            }

            mkFpToBvExpr(rm, value, expr.bvSize, expr.isSigned)
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

    // Eval x * y + z
    @Suppress("ComplexMethod")
    private fun tryEvalFpFma(rm: KFpRoundingMode, x: KFpValue<*>, y: KFpValue<*>, z: KFpValue<*>): KFpValue<*>? =
        with(ctx) {
            when {
                x.isNan() || y.isNan() || z.isNan() -> mkFpNan(x.sort)

                x.isInfinity() && x.isPositive() -> when {
                    y.isZero() -> mkFpNan(x.sort)
                    z.isInfinity() && (x.signBit xor y.signBit xor z.signBit) -> mkFpNan(x.sort)
                    else -> mkFpInf(y.signBit, x.sort)
                }

                y.isInfinity() && y.isPositive() -> when {
                    x.isZero() -> mkFpNan(x.sort)
                    z.isInfinity() && (x.signBit xor y.signBit xor z.signBit) -> mkFpNan(x.sort)
                    else -> mkFpInf(x.signBit, x.sort)
                }

                x.isInfinity() && x.isNegative() -> when {
                    y.isZero() -> mkFpNan(x.sort)
                    z.isInfinity() && (x.signBit xor y.signBit xor z.signBit) -> mkFpNan(x.sort)
                    else -> mkFpInf(!y.signBit, x.sort)
                }

                y.isInfinity() && y.isNegative() -> when {
                    x.isZero() -> mkFpNan(x.sort)
                    z.isInfinity() && (x.signBit xor y.signBit xor z.signBit) -> mkFpNan(x.sort)
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
        }

    private fun tryEvalFpRem(lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*>? = with(ctx) {
        when {
            lhs is KFp32Value -> mkFp(lhs.value.IEEErem((rhs as KFp32Value).value), lhs.sort)
            lhs is KFp64Value -> mkFp(lhs.value.IEEErem((rhs as KFp64Value).value), lhs.sort)
            lhs.isNan() || rhs.isNan() -> mkFpNan(lhs.sort)
            lhs.isInfinity() -> mkFpNan(lhs.sort)
            rhs.isInfinity() -> lhs
            rhs.isZero() -> mkFpNan(lhs.sort)
            lhs.isZero() -> lhs
            // todo: eval fp rem
            else -> null
        }
    }
}
