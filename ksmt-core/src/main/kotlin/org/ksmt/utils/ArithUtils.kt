package org.ksmt.utils

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KInt32NumExpr
import org.ksmt.expr.KInt64NumExpr
import org.ksmt.expr.KIntBigNumExpr
import org.ksmt.expr.KIntNumExpr
import org.ksmt.expr.KRealNumExpr
import org.ksmt.sort.KArithSort
import java.math.BigInteger

object ArithUtils {

    fun <T : KArithSort> KContext.numericValue(realValue: RealValue, sort: T): KExpr<T> = when (sort) {
        intSort -> {
            check(realValue.denominator == BigInteger.ONE) { "Integer with denominator ${realValue.denominator}" }
            mkIntNum(realValue.numerator).uncheckedCast()
        }
        realSort -> mkRealNum(mkIntNum(realValue.numerator), mkIntNum(realValue.denominator)).uncheckedCast()
        else -> error("Unexpected arith sort: $sort")
    }

    operator fun KIntNumExpr.compareTo(other: KIntNumExpr): Int = when {
        this is KInt32NumExpr && other is KInt32NumExpr -> this.value.compareTo(other.value)
        this is KInt32NumExpr && other is KInt64NumExpr -> this.value.compareTo(other.value)
        this is KInt64NumExpr && other is KInt64NumExpr -> this.value.compareTo(other.value)
        this is KInt64NumExpr && other is KInt32NumExpr -> this.value.compareTo(other.value)
        else -> bigIntegerValue.compareTo(other.bigIntegerValue)
    }

    fun KIntNumExpr.toRealValue(): RealValue =
        RealValue.create(bigIntegerValue)

    fun KRealNumExpr.toRealValue(): RealValue =
        RealValue.create(numerator.bigIntegerValue, denominator.bigIntegerValue)

    val KIntNumExpr.bigIntegerValue: BigInteger
        get() = when (this) {
            is KInt32NumExpr -> value.toBigInteger()
            is KInt64NumExpr -> value.toBigInteger()
            is KIntBigNumExpr -> value
            else -> decl.value.toBigInteger()
        }

    class RealValue private constructor(
        numerator: BigInteger,
        denominator: BigInteger
    ) : Comparable<RealValue> {
        val numerator: BigInteger
        val denominator: BigInteger

        init {
            if (denominator.signum() < 0) {
                this.numerator = -numerator
                this.denominator = -denominator
            } else {
                this.numerator = numerator
                this.denominator = denominator
            }
        }

        fun isNegative(): Boolean = numerator.signum() < 0

        fun isZero(): Boolean = numerator == BigInteger.ZERO

        fun abs() = RealValue(numerator.abs(), denominator)

        fun inverse() = RealValue(denominator, numerator)

        fun floor(): BigInteger {
            if (denominator == BigInteger.ONE) {
                return numerator
            }

            var result = numerator.divide(denominator)
            if (isNegative()) {
                result--
            }
            return result
        }

        fun div(value: BigInteger) = RealValue(numerator, denominator * value).normalize()

        fun mul(value: BigInteger) = RealValue(numerator * value, denominator).normalize()

        fun add(other: RealValue): RealValue {
            val resultNumerator = numerator * other.denominator + other.numerator * denominator
            val resultDenominator = denominator * other.denominator
            return RealValue(resultNumerator, resultDenominator).normalize()
        }

        fun sub(other: RealValue): RealValue {
            val resultNumerator = numerator * other.denominator - other.numerator * denominator
            val resultDenominator = denominator * other.denominator
            return RealValue(resultNumerator, resultDenominator).normalize()
        }

        fun mul(other: RealValue): RealValue =
            RealValue(numerator * other.numerator, denominator * other.denominator).normalize()

        fun div(other: RealValue): RealValue =
            RealValue(numerator * other.denominator, denominator * other.numerator).normalize()

        override fun compareTo(other: RealValue): Int = when {
            this.eq(other) -> 0
            this.lt(other) -> -1
            else -> 1
        }

        override fun equals(other: Any?): Boolean = this === other || other is RealValue && eq(other)

        private fun eq(other: RealValue): Boolean =
            numerator == other.numerator && denominator == other.denominator

        private fun lt(b: RealValue): Boolean = when {
            numerator.signum() < 0 && b.numerator.signum() >= 0 -> true
            numerator.signum() > 0 && b.numerator.signum() <= 0 -> false
            numerator.signum() == 0 -> b.numerator.signum() > 0
            else -> numerator * b.denominator < b.numerator * denominator
        }

        private fun normalize(): RealValue {
            if (denominator == BigInteger.ONE || numerator == BigInteger.ONE) {
                return this
            }

            val gcd = numerator.gcd(denominator)
            if (gcd == BigInteger.ONE) {
                return this
            }

            val normalizedNumerator = numerator.divide(gcd)
            val normalizedDenominator = denominator.divide(gcd)
            return RealValue(normalizedNumerator, normalizedDenominator)
        }

        companion object {
            fun create(numerator: BigInteger, denominator: BigInteger = BigInteger.ONE) =
                RealValue(numerator, denominator).normalize()

            val zero by lazy { create(BigInteger.ZERO) }
            val one by lazy { create(BigInteger.ONE) }
            val two by lazy { create(2.toBigInteger()) }
            val minusOne by lazy { create(-BigInteger.ONE) }
        }
    }

}
