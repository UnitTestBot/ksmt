package org.ksmt.expr.rewrite.simplify

import org.ksmt.expr.KAddArithExpr
import org.ksmt.expr.KDivArithExpr
import org.ksmt.expr.KExpr
import org.ksmt.expr.KGeArithExpr
import org.ksmt.expr.KGtArithExpr
import org.ksmt.expr.KInt32NumExpr
import org.ksmt.expr.KInt64NumExpr
import org.ksmt.expr.KIntBigNumExpr
import org.ksmt.expr.KIntNumExpr
import org.ksmt.expr.KIsIntRealExpr
import org.ksmt.expr.KLeArithExpr
import org.ksmt.expr.KLtArithExpr
import org.ksmt.expr.KModIntExpr
import org.ksmt.expr.KMulArithExpr
import org.ksmt.expr.KPowerArithExpr
import org.ksmt.expr.KRealNumExpr
import org.ksmt.expr.KRemIntExpr
import org.ksmt.expr.KSubArithExpr
import org.ksmt.expr.KToIntRealExpr
import org.ksmt.expr.KToRealIntExpr
import org.ksmt.expr.KUnaryMinusArithExpr
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.utils.uncheckedCast
import java.math.BigInteger

interface KArithExprSimplifier : KExprSimplifierBase {

    fun simplifyEqInt(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KExpr<KBoolSort> = with(ctx) {
        if (lhs == rhs) return trueExpr

        if (lhs is KIntNumExpr && rhs is KIntNumExpr) {
            return (lhs.compareTo(rhs) == 0).expr
        }

        return mkEq(lhs, rhs)
    }

    fun areDefinitelyDistinctInt(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): Boolean {
        if (lhs is KIntNumExpr && rhs is KIntNumExpr) {
            return lhs.compareTo(rhs) != 0
        }
        return false
    }

    fun simplifyEqReal(lhs: KExpr<KRealSort>, rhs: KExpr<KRealSort>): KExpr<KBoolSort> = with(ctx) {
        if (lhs == rhs) return trueExpr

        if (lhs is KRealNumExpr && rhs is KRealNumExpr) {
            return (lhs.compareTo(rhs) == 0).expr
        }

        return mkEq(lhs, rhs)
    }

    fun areDefinitelyDistinctReal(lhs: KExpr<KRealSort>, rhs: KExpr<KRealSort>): Boolean {
        if (lhs is KRealNumExpr && rhs is KRealNumExpr) {
            return lhs.compareTo(rhs) != 0
        }
        return false
    }

    override fun <T : KArithSort<T>> transform(expr: KLtArithExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            if (lhs is KIntNumExpr && rhs is KIntNumExpr) {
                return@simplifyApp (lhs < rhs).expr
            }
            if (lhs is KRealNumExpr && rhs is KRealNumExpr) {
                return@simplifyApp (lhs < rhs).expr
            }
            mkArithLt(lhs, rhs)
        }

    override fun <T : KArithSort<T>> transform(expr: KLeArithExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            if (lhs is KIntNumExpr && rhs is KIntNumExpr) {
                return@simplifyApp (lhs <= rhs).expr
            }
            if (lhs is KRealNumExpr && rhs is KRealNumExpr) {
                return@simplifyApp (lhs <= rhs).expr
            }
            mkArithLe(lhs, rhs)
        }

    override fun <T : KArithSort<T>> transform(expr: KGtArithExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            if (lhs is KIntNumExpr && rhs is KIntNumExpr) {
                return@simplifyApp (lhs > rhs).expr
            }
            if (lhs is KRealNumExpr && rhs is KRealNumExpr) {
                return@simplifyApp (lhs > rhs).expr
            }
            mkArithGt(lhs, rhs)
        }

    override fun <T : KArithSort<T>> transform(expr: KGeArithExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            if (lhs is KIntNumExpr && rhs is KIntNumExpr) {
                return@simplifyApp (lhs >= rhs).expr
            }
            if (lhs is KRealNumExpr && rhs is KRealNumExpr) {
                return@simplifyApp (lhs >= rhs).expr
            }
            mkArithGe(lhs, rhs)
        }

    override fun <T : KArithSort<T>> transform(expr: KAddArithExpr<T>): KExpr<T> = simplifyApp(expr) { args ->
        val (numerals, nonNumerals) = args.partition { it is KIntNumExpr || it is KRealNumExpr }
        val simplifiedArgs = nonNumerals.toMutableList()
        if (numerals.isNotEmpty()) {
            simplifiedArgs += sumNumerals(numerals, expr.sort)
        }
        when (simplifiedArgs.size) {
            0 -> numericValue(BigInteger.ZERO, sort = expr.sort)
            1 -> simplifiedArgs.single()
            else -> mkArithAdd(simplifiedArgs)
        }
    }

    override fun <T : KArithSort<T>> transform(expr: KMulArithExpr<T>): KExpr<T> = simplifyApp(expr) { args ->
        val (numerals, nonNumerals) = args.partition { it is KIntNumExpr || it is KRealNumExpr }
        val simplifiedArgs = nonNumerals.toMutableList()
        if (numerals.isNotEmpty()) {
            simplifiedArgs += timesNumerals(numerals, expr.sort)
        }
        when (simplifiedArgs.size) {
            0 -> numericValue(BigInteger.ONE, sort = expr.sort)
            1 -> simplifiedArgs.single()
            else -> mkArithMul(simplifiedArgs)
        }
    }

    override fun <T : KArithSort<T>> transform(expr: KSubArithExpr<T>): KExpr<T> =
        simplifyApp(
            expr = expr,
            preprocess = {
                val args = expr.args
                if (args.size == 1) {
                    args.single()
                } else {
                    val simplifiedArgs = arrayListOf(args.first())
                    for (arg in args.drop(1)) {
                        simplifiedArgs += mkArithUnaryMinus(arg)
                    }
                    mkArithAdd(simplifiedArgs)
                }
            }
        ) {
            error("Always preprocessed")
        }

    override fun <T : KArithSort<T>> transform(expr: KUnaryMinusArithExpr<T>): KExpr<T> = simplifyApp(expr) { (arg) ->
        if (arg is KIntNumExpr) {
            return@simplifyApp mkIntNum(-arg.value).uncheckedCast()
        }
        if (arg is KRealNumExpr) {
            return@simplifyApp mkRealNum(
                mkIntNum(-arg.numerator.value), arg.denominator
            ).uncheckedCast()
        }
        mkArithUnaryMinus(arg)
    }

    override fun <T : KArithSort<T>> transform(expr: KDivArithExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        when (expr.sort) {
            intSort -> rewrite(simplifyIntegerDiv(lhs.uncheckedCast(), rhs.uncheckedCast()) as KExpr<T>)
            realSort -> rewrite(simplifyRealDiv(lhs.uncheckedCast(), rhs.uncheckedCast()) as KExpr<T>)
            else -> mkArithDiv(lhs, rhs)
        }
    }

    private fun simplifyIntegerDiv(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KExpr<KIntSort> = with(ctx) {
        if (rhs is KIntNumExpr) {
            val rValue = rhs.value
            when (rValue) {
                BigInteger.ZERO -> return mkArithDiv(lhs, rhs)
                BigInteger.ONE -> return lhs
                -BigInteger.ONE -> return -lhs
            }
            if (lhs is KIntNumExpr) {
                return mkIntNum(lhs.value / rValue)
            }
        }
        if (lhs == rhs) {
            return mkIte(lhs eq 0.expr, mkArithDiv(0.expr, 0.expr), 1.expr)
        }

        return mkArithDiv(lhs, rhs)
    }

    private fun simplifyRealDiv(lhs: KExpr<KRealSort>, rhs: KExpr<KRealSort>): KExpr<KRealSort> = with(ctx) {
        if (rhs is KRealNumExpr) {
            val coefficient = mkRealNum(rhs.denominator, rhs.numerator)
            return mkArithMul(lhs, coefficient)
        }
        return mkArithDiv(lhs, rhs)
    }

    @Suppress("ForbiddenComment")
    override fun <T : KArithSort<T>> transform(expr: KPowerArithExpr<T>): KExpr<T> =
        simplifyApp(expr) { (base, power) ->
            // todo: evaluate
            mkArithPower(base, power)
        }

    override fun transform(expr: KModIntExpr): KExpr<KIntSort> = simplifyApp(expr) { (lhs, rhs) ->
        if (rhs is KIntNumExpr) {
            val rValue = rhs.value
            when (rValue) {
                BigInteger.ZERO -> return@simplifyApp mkIntMod(lhs, rhs)
                BigInteger.ONE, -BigInteger.ONE -> return@simplifyApp 0.expr
            }
            if (lhs is KIntNumExpr) {
                return@simplifyApp (lhs.value.mod(rValue)).expr
            }
        }
        mkIntMod(lhs, rhs)
    }

    override fun transform(expr: KRemIntExpr): KExpr<KIntSort> = simplifyApp(expr) { (lhs, rhs) ->
        if (rhs is KIntNumExpr) {
            val rValue = rhs.value
            when (rValue) {
                BigInteger.ZERO -> return@simplifyApp mkIntMod(lhs, rhs)
                BigInteger.ONE, -BigInteger.ONE -> return@simplifyApp 0.expr
            }
            if (lhs is KIntNumExpr) {
                return@simplifyApp (lhs.value.rem(rValue)).expr
            }
        }
        mkIntRem(lhs, rhs)
    }

    override fun transform(expr: KToIntRealExpr): KExpr<KIntSort> = simplifyApp(expr) { (arg) ->
        if (arg is KRealNumExpr) {
            val numer = arg.numerator.value
            val denom = arg.denominator.value
            return@simplifyApp (numer / denom).expr
        }
        // (real2int (int2real x)) ==> x
        if (arg is KToRealIntExpr) {
            return@simplifyApp arg.arg
        }
        mkRealToInt(arg)
    }

    override fun transform(expr: KIsIntRealExpr): KExpr<KBoolSort> = simplifyApp(expr) { (arg) ->
        if (arg is KRealNumExpr) {
            return@simplifyApp (arg.denominator.value == BigInteger.ONE).expr
        }
        // (isInt (int2real x)) ==> true
        if (arg is KToRealIntExpr) {
            return@simplifyApp trueExpr
        }
        mkRealIsInt(arg)
    }

    override fun transform(expr: KToRealIntExpr): KExpr<KRealSort> = simplifyApp(expr) { (arg) ->
        if (arg is KIntNumExpr) {
            return@simplifyApp mkRealNum(arg)
        }
        mkIntToReal(arg)
    }

    private fun <T : KArithSort<T>> sumNumerals(
        numerals: List<KExpr<T>>, sort: T
    ): KExpr<T> {
        var numerator = BigInteger.ZERO
        var denominator = BigInteger.ONE
        for (numeral in numerals) {
            if (numeral is KIntNumExpr) {
                val n = numeral.value
                numerator += (n * denominator)
            }
            if (numeral is KRealNumExpr) {
                val n = numeral.numerator.value
                val d = numeral.denominator.value
                numerator = (numerator * d) + (n * denominator)
                denominator *= d
            }
        }
        return numericValue(numerator, denominator, sort)
    }

    private fun <T : KArithSort<T>> timesNumerals(
        numerals: List<KExpr<T>>, sort: T
    ): KExpr<T> {
        var numerator = BigInteger.ONE
        var denominator = BigInteger.ONE
        for (numeral in numerals) {
            if (numeral is KIntNumExpr) {
                numerator *= numeral.value
            }
            if (numeral is KRealNumExpr) {
                numerator *= numeral.numerator.value
                denominator *= numeral.denominator.value
            }
        }
        return numericValue(numerator, denominator, sort)
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T : KArithSort<T>> numericValue(
        numerator: BigInteger, denominator: BigInteger = BigInteger.ONE, sort: T
    ): KExpr<T> = with(ctx) {
        when (sort) {
            intSort -> mkIntNum(numerator) as KExpr<T>
            realSort -> mkRealNum(mkIntNum(numerator), mkIntNum(denominator)) as KExpr<T>
            else -> error("Unexpected arith sort: $sort")
        }
    }

    private operator fun KIntNumExpr.compareTo(other: KIntNumExpr): Int = when {
        this is KInt32NumExpr && other is KInt32NumExpr -> this.value.compareTo(other.value)
        this is KInt32NumExpr && other is KInt64NumExpr -> this.value.compareTo(other.value)
        this is KInt64NumExpr && other is KInt64NumExpr -> this.value.compareTo(other.value)
        this is KInt64NumExpr && other is KInt32NumExpr -> this.value.compareTo(other.value)
        else -> value.compareTo(other.value)
    }

    private operator fun KRealNumExpr.compareTo(other: KRealNumExpr): Int {
        val na = this.numerator.value
        val nb = other.numerator.value

        val naSign = na.signum()
        val nbSign = nb.signum()
        return when {
            naSign == 0 -> when {
                nbSign == 0 -> 0
                nbSign > 0 -> -1
                else -> 1
            }

            naSign < 0 && nbSign >= 0 -> -1
            naSign > 0 && nbSign <= 0 -> 1
            else -> {
                // (naSign < 0 && nbSign < 0) || (naSign > 0 && nbSign > 0) ||
                val da = this.denominator.value
                val db = other.denominator.value

                val aNormalized = na * db
                val bNormalized = nb * da
                aNormalized.compareTo(bNormalized)
            }
        }
    }

    private val KIntNumExpr.value
        get() = when (this) {
            is KInt32NumExpr -> value.toBigInteger()
            is KInt64NumExpr -> value.toBigInteger()
            is KIntBigNumExpr -> value
            else -> decl.value.toBigInteger()
        }
}
