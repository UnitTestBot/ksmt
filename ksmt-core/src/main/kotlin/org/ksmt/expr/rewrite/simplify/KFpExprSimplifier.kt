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
import org.ksmt.utils.asExpr
import java.math.BigDecimal
import java.math.BigInteger
import java.math.RoundingMode
import kotlin.math.IEEErem
import kotlin.math.absoluteValue
import kotlin.math.round
import kotlin.math.sqrt

@Suppress(
    "ForbiddenComment"
)
interface KFpExprSimplifier : KExprSimplifierBase, KBvExprSimplifier {

    fun <T : KFpSort> simplifyEqFp(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> = with(ctx) {
        if (lhs == rhs) return trueExpr

        if (lhs is KFpValue<*> && rhs is KFpValue<*>) {
            // special cases
            if (lhs.isNan() && rhs.isNan()) return trueExpr
            if (lhs.isZero() && rhs.isZero() && lhs.signBit != rhs.signBit) return falseExpr

            // compare floats
            lhs.fpCompareTo(rhs)?.let { return (it == 0).expr }
        }

        return mkEq(lhs, rhs)
    }

    fun <T : KFpSort> areDefinitelyDistinctFp(lhs: KExpr<T>, rhs: KExpr<T>): Boolean {
        if (lhs is KFpValue<*> && rhs is KFpValue<*>) {
            // special cases
            if (lhs.isNan() != rhs.isNan()) return true
            if (lhs.isZero() != rhs.isZero()) return true
            if (lhs.isZero() && lhs.signBit != rhs.signBit) return true

            // compare floats
            lhs.fpCompareTo(rhs)?.let { return it != 0 }
        }
        return false
    }

    override fun <T : KFpSort> transform(expr: KFpAbsExpr<T>): KExpr<T> = simplifyApp(expr) { (arg) ->
        if (arg is KFpValue<*>) {
            // (abs NaN) ==> Nan
            if (arg.isNan()) {
                return@simplifyApp arg.asExpr(expr.sort)
            }

            if (arg.signBit) {
                // (abs x), x < 0 ==> -x
                val negated = arg.fpUnaryMinus()?.asExpr(expr.sort)
                negated?.let { return@simplifyApp it }
            } else {
                // (abs x), x >= 0 ==> x
                return@simplifyApp arg.asExpr(expr.sort)
            }
        }
        mkFpAbsExpr(arg)
    }

    override fun <T : KFpSort> transform(expr: KFpNegationExpr<T>): KExpr<T> = simplifyApp(expr) { (arg) ->
        if (arg is KFpValue<*>) {
            arg.fpUnaryMinus()?.asExpr(expr.sort)?.let { return@simplifyApp it }
        }

        // (- -x) ==> x
        if (arg is KFpNegationExpr<*>) {
            return@simplifyApp arg.value.asExpr(expr.sort)
        }

        mkFpNegationExpr(arg)
    }

    override fun <T : KFpSort> transform(expr: KFpAddExpr<T>): KExpr<T> = expr.simplifyFpBinaryOp { rm, lhs, rhs ->
        if (lhs is KFpValue<*> && rhs is KFpValue<*> && rm is KFpRoundingModeExpr) {
            val result = fpAdd(rm.value, lhs, rhs)?.asExpr(expr.sort)
            result?.let { return@simplifyFpBinaryOp it }
        }

        mkFpAddExpr(rm, lhs, rhs)
    }

    // a - b ==> a + (-b)
    @Suppress("UNCHECKED_CAST")
    override fun <T : KFpSort> transform(expr: KFpSubExpr<T>): KExpr<T> =
        simplifyApp(
            expr = expr as KApp<T, KExpr<KSort>>,
            preprocess = { mkFpAddExpr(expr.roundingMode, expr.arg0, mkFpNegationExpr(expr.arg1)) }
        ) {
            error("Always preprocessed")
        }

    override fun <T : KFpSort> transform(expr: KFpMulExpr<T>): KExpr<T> = expr.simplifyFpBinaryOp { rm, lhs, rhs ->
        if (lhs is KFpValue<*> && rhs is KFpValue<*> && rm is KFpRoundingModeExpr) {
            val result = fpMul(rm.value, lhs, rhs)?.asExpr(expr.sort)
            result?.let { return@simplifyFpBinaryOp it }
        }

        mkFpMulExpr(rm, lhs, rhs)
    }

    override fun <T : KFpSort> transform(expr: KFpDivExpr<T>): KExpr<T> = expr.simplifyFpBinaryOp { rm, lhs, rhs ->
        if (lhs is KFpValue<*> && rhs is KFpValue<*> && rm is KFpRoundingModeExpr) {
            val result = fpDiv(rm.value, lhs, rhs)?.asExpr(expr.sort)
            result?.let { return@simplifyFpBinaryOp it }
        }

        mkFpDivExpr(rm, lhs, rhs)
    }

    @Suppress("ComplexCondition")
    override fun <T : KFpSort> transform(expr: KFpFusedMulAddExpr<T>): KExpr<T> =
        expr.simplifyFpTernaryOp { rm, a0, a1, a2 ->
            if (rm is KFpRoundingModeExpr && a0 is KFpValue<*> && a1 is KFpValue<*> && a2 is KFpValue<*>) {
                val result = fpFma(rm.value, a0, a1, a2)?.asExpr(expr.sort)
                result?.let { return@simplifyFpTernaryOp it }
            }
            mkFpFusedMulAddExpr(rm, a0, a1, a2)
        }

    override fun <T : KFpSort> transform(expr: KFpSqrtExpr<T>): KExpr<T> =
        expr.simplifyFpUnaryOp { rm, arg ->
            if (arg is KFpValue<*> && rm is KFpRoundingModeExpr) {
                val result = fpSqrt(rm.value, arg)?.asExpr(expr.sort)
                result?.let { return@simplifyFpUnaryOp it }
            }
            mkFpSqrtExpr(rm, arg)
        }

    override fun <T : KFpSort> transform(expr: KFpRoundToIntegralExpr<T>): KExpr<T> =
        expr.simplifyFpUnaryOp { rm, arg ->
            if (arg is KFpValue<*> && rm is KFpRoundingModeExpr) {
                val result = fpRoundToIntegral(rm.value, arg)?.asExpr(expr.sort)
                result?.let { return@simplifyFpUnaryOp it }
            }
            mkFpRoundToIntegralExpr(rm, arg)
        }

    override fun <T : KFpSort> transform(expr: KFpRemExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val lhsValue = lhs as? KFpValue<*>
        val rhsValue = rhs as? KFpValue<*>

        if (lhsValue != null && rhsValue != null) {
            val result = fpRem(lhsValue, rhsValue)?.asExpr(expr.sort)
            result?.let { return@simplifyApp it }
        }

        mkFpRemExpr(lhs, rhs)
    }

    override fun <T : KFpSort> transform(expr: KFpMinExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val lhsValue = lhs as? KFpValue<*>
        val rhsValue = rhs as? KFpValue<*>

        if (lhsValue != null && lhsValue.isNan()) {
            return@simplifyApp rhs
        }

        if (rhsValue != null && rhsValue.isNan()) {
            return@simplifyApp lhs
        }

        if (lhsValue != null && rhsValue != null) {
            if (lhsValue.isZero() && rhsValue.isZero() && lhsValue.signBit != rhsValue.signBit) {
                return@simplifyApp mkFpMinExpr(lhs.asExpr(expr.sort), rhs.asExpr(expr.sort))
            }

            val result = fpMin(lhsValue, rhsValue)?.asExpr(expr.sort)
            result?.let { return@simplifyApp it }
        }

        mkFpMinExpr(lhs, rhs)
    }

    override fun <T : KFpSort> transform(expr: KFpMaxExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val lhsValue = lhs as? KFpValue<*>
        val rhsValue = rhs as? KFpValue<*>

        if (lhsValue != null && lhsValue.isNan()) {
            return@simplifyApp rhs
        }

        if (rhsValue != null && rhsValue.isNan()) {
            return@simplifyApp lhs
        }

        if (lhsValue != null && rhsValue != null) {
            if (lhsValue.isZero() && rhsValue.isZero() && lhsValue.signBit != rhsValue.signBit) {
                return@simplifyApp mkFpMaxExpr(lhs.asExpr(expr.sort), rhs.asExpr(expr.sort))
            }

            val result = fpMax(lhsValue, rhsValue)?.asExpr(expr.sort)
            result?.let { return@simplifyApp it }
        }

        mkFpMaxExpr(lhs, rhs)
    }

    @Suppress("ComplexCondition")
    override fun <T : KFpSort> transform(expr: KFpLessOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            val lhsValue = lhs as? KFpValue<*>
            val rhsValue = rhs as? KFpValue<*>

            if (lhsValue != null && lhsValue.isNan() || rhsValue != null && rhsValue.isNan()) {
                return@simplifyApp falseExpr
            }

            if (lhsValue != null && rhsValue != null) {
                val check = fpLe(lhsValue, rhsValue)?.expr
                check?.let { return@simplifyApp it }
            }

            mkFpLessOrEqualExpr(lhs, rhs)
        }

    @Suppress("ComplexCondition")
    override fun <T : KFpSort> transform(expr: KFpLessExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            val lhsValue = lhs as? KFpValue<*>
            val rhsValue = rhs as? KFpValue<*>

            if (lhsValue != null && lhsValue.isNan() || rhsValue != null && rhsValue.isNan()) {
                return@simplifyApp falseExpr
            }

            if (lhsValue != null && lhsValue.isInfinity() && !lhsValue.signBit) {
                return@simplifyApp falseExpr
            }

            if (rhsValue != null && rhsValue.isInfinity() && rhsValue.signBit) {
                return@simplifyApp falseExpr
            }

            if (lhsValue != null && rhsValue != null) {
                val check = fpLt(lhsValue, rhsValue)?.expr
                check?.let { return@simplifyApp it }
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
            val lhsValue = lhs as? KFpValue<*>
            val rhsValue = rhs as? KFpValue<*>

            if (lhsValue != null && rhsValue != null) {
                val check = fpEq(lhsValue, rhsValue)?.expr
                check?.let { return@simplifyApp it }
            }

            mkFpEqualExpr(lhs, rhs)
        }

    override fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            if (arg is KFpValue<*>) {
                return@simplifyApp fpIsNormal(arg).expr
            }
            mkFpIsNormalExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            if (arg is KFpValue<*>) {
                return@simplifyApp fpIsDenormal(arg).expr
            }
            mkFpIsSubnormalExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            if (arg is KFpValue<*>) {
                return@simplifyApp arg.isZero().expr
            }
            mkFpIsZeroExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            if (arg is KFpValue<*>) {
                return@simplifyApp arg.isInfinity().expr
            }
            mkFpIsInfiniteExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            if (arg is KFpValue<*>) {
                return@simplifyApp arg.isNan().expr
            }
            mkFpIsNaNExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            if (arg is KFpValue<*>) {
                return@simplifyApp (!arg.isNan() && arg.signBit).expr
            }
            mkFpIsNegativeExpr(arg)
        }

    override fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            if (arg is KFpValue<*>) {
                return@simplifyApp (!arg.isNan() && !arg.signBit).expr
            }
            mkFpIsPositiveExpr(arg)
        }

    @Suppress("UNCHECKED_CAST")
    override fun <T : KFpSort> transform(expr: KFpFromBvExpr<T>): KExpr<T> =
        simplifyApp(expr as KApp<T, KExpr<KBvSort>>) { (sign, exp, significand) ->
            if (sign is KBitVecValue<*> && exp is KBitVecValue<*> && significand is KBitVecValue<*>) {

                val unbiasedExponent = fpUnbiasExponent(exp)
                return@simplifyApp mkFpCustomSize(
                    unbiasedExponent = unbiasedExponent,
                    significand = significand,
                    signBit = (sign as KBitVec1Value).value
                )
            }
            mkFpFromBvExpr(sign.asExpr(bv1Sort), exp, significand)
        }

    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> = simplifyApp(expr) { (arg) ->
        if (arg is KFpValue<*>) {
            if (arg.isNan()) {
                // ensure NaN bits are the same, as in KContext
                val nan = ctx.mkFpNan(arg.sort) as KFpValue<*>
                return@simplifyApp mkBvConcatExpr(
                    mkBv(nan.signBit),
                    mkBvConcatExpr(nan.biasedExponent, nan.significand)
                ).also { rewrite(it) }
            }
            return@simplifyApp mkBvConcatExpr(
                mkBv(arg.signBit),
                mkBvConcatExpr(arg.biasedExponent, arg.significand)
            ).also { rewrite(it) }
        }
        mkFpToIEEEBvExpr(arg)
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : KFpSort> transform(expr: KFpToFpExpr<T>): KExpr<T> =
        simplifyApp(expr as KApp<T, KExpr<KSort>>) { (rmArg, valueArg) ->
            val rm = rmArg.asExpr(mkFpRoundingModeSort())
            val value = valueArg.asExpr(expr.value.sort)

            if (rm is KFpRoundingModeExpr && value is KFpValue<*>) {
                // todo: evaluate
                return@simplifyApp mkFpToFpExpr(expr.sort, rm, value)
            }

            mkFpToFpExpr(expr.sort, rm, value)
        }

    @Suppress("UNCHECKED_CAST")
    override fun <T : KFpSort> transform(expr: KRealToFpExpr<T>): KExpr<T> =
        simplifyApp(expr as KApp<T, KExpr<KSort>>) { (rmArg, valueArg) ->
            val rm = rmArg.asExpr(mkFpRoundingModeSort())
            val value = valueArg.asExpr(expr.value.sort)

            if (rm is KFpRoundingModeExpr && value is KRealNumExpr) {
                // todo: evaluate
                return@simplifyApp mkRealToFpExpr(expr.sort, rm, value)
            }

            mkRealToFpExpr(expr.sort, rm, value)
        }

    override fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort> = simplifyApp(expr) { (arg) ->
        if (arg is KFpValue<*>) {
            if (!arg.isNan() && !arg.isInfinity()) {
                val decimalValue = fpDecimalValue(arg)
                if (decimalValue != null) {
                    val decimalPower = decimalValue.scale()
                    var numerator = decimalValue.unscaledValue()
                    var denominator = BigInteger.ONE
                    if (decimalPower >= 0) {
                        numerator *= BigInteger.TEN.pow(decimalPower)
                    } else {
                        denominator = BigInteger.TEN.pow(decimalPower.absoluteValue)
                    }
                    return@simplifyApp mkRealNum(numerator.expr, denominator.expr)
                }
            }
        }
        mkFpToRealExpr(arg)
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : KFpSort> transform(expr: KBvToFpExpr<T>): KExpr<T> =
        simplifyApp(expr as KApp<T, KExpr<KSort>>) { (rmArg, bvValueArg) ->
            val rm = rmArg.asExpr(mkFpRoundingModeSort())
            val value = bvValueArg.asExpr(expr.value.sort)

            if (rm is KFpRoundingModeExpr && value is KBitVecValue<*>) {
                // todo: evaluate
                return@simplifyApp mkBvToFpExpr(expr.sort, rm, value, expr.signed)
            }

            mkBvToFpExpr(expr.sort, rm, value, expr.signed)
        }

    @Suppress("UNCHECKED_CAST")
    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort> =
        simplifyApp(expr as KApp<KBvSort, KExpr<KSort>>) { (rmArg, valueArg) ->
            val rm = rmArg.asExpr(mkFpRoundingModeSort())
            val value = valueArg.asExpr(expr.value.sort)

            if (rm is KFpRoundingModeExpr && value is KFpValue<*>) {
                if (!value.isNan() && !value.isInfinity()) {
                    val decimalValue = fpDecimalValue(value)
                    if (decimalValue != null) {
                        val lowLimit = if (expr.isSigned) {
                            -BigDecimal.valueOf(2).pow(expr.bvSize - 1)
                        } else {
                            BigDecimal.ZERO
                        }
                        val upperLimit = if (expr.isSigned) {
                            BigDecimal.valueOf(2).pow(expr.bvSize - 1) - BigDecimal.valueOf(1)
                        } else {
                            BigDecimal.valueOf(2).pow(expr.bvSize) - BigDecimal.valueOf(1)
                        }
                        if (decimalValue >= lowLimit && decimalValue <= upperLimit) {
                            val intValue = decimalValue.unscaledValue(rm.value).toBigInteger()
                            return@simplifyApp mkBvFromBigInteger(intValue, expr.bvSize.toUInt())
                        }
                    }
                }
            }

            mkFpToBvExpr(rm, value, expr.bvSize, expr.isSigned)
        }

    @Suppress("UNCHECKED_CAST")
    private inline fun <T : KFpSort> KExpr<T>.simplifyFpUnaryOp(
        crossinline simplifier: KContext.(KExpr<KFpRoundingModeSort>, KExpr<T>) -> KExpr<T>
    ): KExpr<T> = simplifyApp(this as KApp<T, KExpr<KSort>>) { (rm, value) ->
        simplifier(ctx, rm.asExpr(mkFpRoundingModeSort()), value.asExpr(sort))
    }

    @Suppress("UNCHECKED_CAST")
    private inline fun <T : KFpSort> KExpr<T>.simplifyFpBinaryOp(
        crossinline simplifier: KContext.(KExpr<KFpRoundingModeSort>, KExpr<T>, KExpr<T>) -> KExpr<T>
    ): KExpr<T> = simplifyApp(this as KApp<T, KExpr<KSort>>) { (rm, lhs, rhs) ->
        simplifier(ctx, rm.asExpr(mkFpRoundingModeSort()), lhs.asExpr(sort), rhs.asExpr(sort))
    }

    @Suppress("UNCHECKED_CAST")
    private inline fun <T : KFpSort> KExpr<T>.simplifyFpTernaryOp(
        crossinline simplifier: KContext.(KExpr<KFpRoundingModeSort>, KExpr<T>, KExpr<T>, KExpr<T>) -> KExpr<T>
    ): KExpr<T> = simplifyApp(this as KApp<T, KExpr<KSort>>) { (rm, a0, a1, a2) ->
        simplifier(ctx, rm.asExpr(mkFpRoundingModeSort()), a0.asExpr(sort), a1.asExpr(sort), a2.asExpr(sort))
    }

    private fun KFpValue<*>.isNan(): Boolean = when (this) {
        is KFp32Value -> value.isNaN()
        is KFp64Value -> value.isNaN()
        else -> this == ctx.mkFpNan(sort)
    }

    @Suppress("MagicNumber")
    private fun KFpValue<*>.isZero(): Boolean = when (this) {
        is KFp32Value -> value == 0.0f || value == -0.0f
        is KFp64Value -> value == 0.0 || value == -0.0
        else -> this == ctx.mkFpZero(signBit, sort)
    }

    private fun KFpValue<*>.isInfinity(): Boolean = when (this) {
        is KFp32Value -> value.isInfinite()
        is KFp64Value -> value.isInfinite()
        else -> this == ctx.mkFpInf(signBit, sort)
    }

    private fun KFpValue<*>.fpCompareTo(other: KFpValue<*>): Int? = when (this) {
        is KFp32Value -> value.compareTo((other as KFp32Value).value)
        is KFp64Value -> value.compareTo((other as KFp64Value).value)
        else -> null
    }

    private fun KFpValue<*>.fpUnaryMinus(): KFpValue<*>? = with(ctx) {
        when (this@fpUnaryMinus) {
            is KFp32Value -> mkFp(-value, sort) as KFpValue<*>
            is KFp64Value -> mkFp(-value, sort) as KFpValue<*>
            else -> null
        }
    }

    private fun fpAdd(rm: KFpRoundingMode, lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*>? = with(ctx) {
        if (rm != KFpRoundingMode.RoundNearestTiesToEven) {
            // todo: RNE is JVM default. Support others.
            return null
        }
        when (lhs) {
            is KFp32Value -> mkFp(lhs.value + (rhs as KFp32Value).value, lhs.sort) as KFpValue<*>
            is KFp64Value -> mkFp(lhs.value + (rhs as KFp64Value).value, lhs.sort) as KFpValue<*>
            else -> null
        }
    }

    private fun fpMul(rm: KFpRoundingMode, lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*>? = with(ctx) {
        if (rm != KFpRoundingMode.RoundNearestTiesToEven) {
            // todo: RNE is JVM default. Support others.
            return null
        }
        when (lhs) {
            is KFp32Value -> mkFp(lhs.value * (rhs as KFp32Value).value, lhs.sort) as KFpValue<*>
            is KFp64Value -> mkFp(lhs.value * (rhs as KFp64Value).value, lhs.sort) as KFpValue<*>
            else -> null
        }
    }

    private fun fpDiv(rm: KFpRoundingMode, lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*>? = with(ctx) {
        if (rm != KFpRoundingMode.RoundNearestTiesToEven) {
            // todo: RNE is JVM default. Support others.
            return null
        }
        when (lhs) {
            is KFp32Value -> mkFp(lhs.value / (rhs as KFp32Value).value, lhs.sort) as KFpValue<*>
            is KFp64Value -> mkFp(lhs.value / (rhs as KFp64Value).value, lhs.sort) as KFpValue<*>
            else -> null
        }
    }

    private fun fpFma(rm: KFpRoundingMode, a: KFpValue<*>, b: KFpValue<*>, c: KFpValue<*>): KFpValue<*>? = with(ctx) {
        if (rm != KFpRoundingMode.RoundNearestTiesToEven) {
            // todo: RNE is JVM default. Support others.
            return null
        }

        when (a) {
            is KFp32Value ->
                mkFp(Math.fma(a.value, (b as KFp32Value).value, (c as KFp32Value).value), a.sort) as KFpValue<*>

            is KFp64Value ->
                mkFp(Math.fma(a.value, (b as KFp64Value).value, (c as KFp64Value).value), a.sort) as KFpValue<*>

            else -> null
        }
    }

    private fun fpSqrt(rm: KFpRoundingMode, arg: KFpValue<*>): KFpValue<*>? = with(ctx) {
        if (rm != KFpRoundingMode.RoundNearestTiesToEven) {
            // todo: RNE is JVM default. Support others.
            return null
        }
        when (arg) {
            is KFp32Value -> mkFp(sqrt(arg.value), arg.sort) as KFpValue<*>
            is KFp64Value -> mkFp(sqrt(arg.value), arg.sort) as KFpValue<*>
            else -> null
        }
    }

    private fun fpRoundToIntegral(rm: KFpRoundingMode, arg: KFpValue<*>): KFpValue<*>? = with(ctx) {
        if (rm != KFpRoundingMode.RoundTowardPositive) {
            // todo: JVM Math rounds toward positive.
            return null
        }
        when (arg) {
            is KFp32Value -> mkFp(round(arg.value), arg.sort) as KFpValue<*>
            is KFp64Value -> mkFp(round(arg.value), arg.sort) as KFpValue<*>
            else -> null
        }
    }

    private fun fpRem(lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*>? = with(ctx) {
        when (lhs) {
            is KFp32Value -> mkFp(lhs.value.IEEErem((rhs as KFp32Value).value), lhs.sort) as KFpValue<*>
            is KFp64Value -> mkFp(lhs.value.IEEErem((rhs as KFp64Value).value), lhs.sort) as KFpValue<*>
            else -> null
        }
    }

    private fun fpMin(lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*>? = with(ctx) {
        when (lhs) {
            is KFp32Value -> mkFp(minOf(lhs.value, (rhs as KFp32Value).value), lhs.sort) as KFpValue<*>
            is KFp64Value -> mkFp(minOf(lhs.value, (rhs as KFp64Value).value), lhs.sort) as KFpValue<*>
            else -> null
        }
    }

    private fun fpMax(lhs: KFpValue<*>, rhs: KFpValue<*>): KFpValue<*>? = with(ctx) {
        when (lhs) {
            is KFp32Value -> mkFp(maxOf(lhs.value, (rhs as KFp32Value).value), lhs.sort) as KFpValue<*>
            is KFp64Value -> mkFp(maxOf(lhs.value, (rhs as KFp64Value).value), lhs.sort) as KFpValue<*>
            else -> null
        }
    }

    private fun fpIsNormal(arg: KFpValue<*>): Boolean {
        val topExp = (ctx.mkFpInf(signBit = false, arg.sort) as KFpValue<*>).biasedExponent
        return !(arg.isZero() || fpIsDenormal(arg) || arg.biasedExponent == topExp)
    }

    private fun fpIsDenormal(arg: KFpValue<*>): Boolean {
        val botExp = (ctx.mkFpZero(signBit = false, arg.sort) as KFpValue<*>).biasedExponent
        return !arg.isZero() && arg.biasedExponent == botExp
    }

    private fun fpEq(lhs: KFpValue<*>, rhs: KFpValue<*>): Boolean? = when (lhs) {
        is KFp32Value -> lhs.value == (rhs as KFp32Value).value
        is KFp64Value -> lhs.value == (rhs as KFp64Value).value
        else -> null
    }

    private fun fpLt(lhs: KFpValue<*>, rhs: KFpValue<*>): Boolean? = when (lhs) {
        is KFp32Value -> lhs.value < (rhs as KFp32Value).value
        is KFp64Value -> lhs.value < (rhs as KFp64Value).value
        else -> null
    }

    private fun fpLe(lhs: KFpValue<*>, rhs: KFpValue<*>): Boolean? = when (lhs) {
        is KFp32Value -> lhs.value <= (rhs as KFp32Value).value
        is KFp64Value -> lhs.value <= (rhs as KFp64Value).value
        else -> null
    }

    private fun fpDecimalValue(value: KFpValue<*>): BigDecimal? = when (value) {
        is KFp32Value -> BigDecimal.valueOf(value.value.toDouble())
        is KFp64Value -> BigDecimal.valueOf(value.value)
        else -> null
    }

    private fun fpUnbiasExponent(value: KBitVecValue<*>): KBitVecValue<*> = with(this as KBvExprSimplifier) {
        value - maxValueSigned(value.sort.sizeBits)
    }

    private fun BigDecimal.unscaledValue(rm: KFpRoundingMode): BigDecimal {
        val decimalRm = when (rm) {
            KFpRoundingMode.RoundNearestTiesToEven -> RoundingMode.HALF_EVEN
            KFpRoundingMode.RoundNearestTiesToAway -> RoundingMode.HALF_UP
            KFpRoundingMode.RoundTowardPositive -> RoundingMode.CEILING
            KFpRoundingMode.RoundTowardNegative -> RoundingMode.FLOOR
            KFpRoundingMode.RoundTowardZero -> RoundingMode.DOWN
        }
        return setScale(0, decimalRm)
    }
}
