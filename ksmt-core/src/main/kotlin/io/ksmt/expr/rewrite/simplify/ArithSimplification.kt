package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KAddArithExpr
import io.ksmt.expr.KExpr
import io.ksmt.expr.KIntNumExpr
import io.ksmt.expr.KMulArithExpr
import io.ksmt.expr.KRealNumExpr
import io.ksmt.expr.KToRealIntExpr
import io.ksmt.expr.KUnaryMinusArithExpr
import io.ksmt.sort.KArithSort
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import io.ksmt.utils.ArithUtils.RealValue
import io.ksmt.utils.ArithUtils.bigIntegerValue
import io.ksmt.utils.ArithUtils.compareTo
import io.ksmt.utils.ArithUtils.numericValue
import io.ksmt.utils.ArithUtils.toRealValue
import io.ksmt.utils.uncheckedCast
import java.math.BigInteger
import kotlin.math.absoluteValue

fun <T : KArithSort> KContext.simplifyArithUnaryMinus(arg: KExpr<T>): KExpr<T> {
    if (arg is KIntNumExpr) {
        return mkIntNum(-arg.bigIntegerValue).uncheckedCast()
    }

    if (arg is KRealNumExpr) {
        return mkRealNum(
            mkIntNum(-arg.numerator.bigIntegerValue),
            arg.denominator
        ).uncheckedCast()
    }

    if (arg is KUnaryMinusArithExpr<T>) {
        return arg.arg
    }

    return mkArithUnaryMinusNoSimplify(arg)
}

fun <T : KArithSort> KContext.simplifyArithAdd(args: List<KExpr<T>>): KExpr<T> {
    require(args.isNotEmpty()) {
        "Arith add requires at least a single argument"
    }

    val simplifiedArgs = ArrayList<KExpr<T>>(args.size)
    var constantTerm = RealValue.zero

    for (arg in args) {
        // flat one level
        if (arg is KAddArithExpr<T>) {
            for (flatArg in arg.args) {
                constantTerm = addArithTerm(constantTerm, flatArg, simplifiedArgs)
            }
        } else {
            constantTerm = addArithTerm(constantTerm, arg, simplifiedArgs)
        }
    }

    if (simplifiedArgs.isEmpty()) {
        return numericValue(constantTerm, args.first().sort)
    }

    if (!constantTerm.isZero()) {
        // prefer constant to be the first argument
        val firstArg = simplifiedArgs.first()
        simplifiedArgs[0] = numericValue(constantTerm, firstArg.sort)
        simplifiedArgs.add(firstArg)
    }

    if (simplifiedArgs.size == 1) {
        return simplifiedArgs.single()
    }

    return mkArithAddNoSimplify(simplifiedArgs)
}

fun <T : KArithSort> KContext.simplifyArithSub(args: List<KExpr<T>>): KExpr<T> =
    if (args.size == 1) {
        args.single()
    } else {
        require(args.isNotEmpty()) {
            "Arith sub requires at least a single argument"
        }

        val simplifiedArgs = arrayListOf(args.first())
        for (arg in args.drop(1)) {
            simplifiedArgs += simplifyArithUnaryMinus(arg)
        }
        simplifyArithAdd(simplifiedArgs)
    }

fun <T : KArithSort> KContext.simplifyArithMul(args: List<KExpr<T>>): KExpr<T> {
    require(args.isNotEmpty()) {
        "Arith mul requires at least a single argument"
    }

    val simplifiedArgs = ArrayList<KExpr<T>>(args.size)
    var constantTerm = RealValue.one

    for (arg in args) {
        // flat one level
        if (arg is KMulArithExpr<T>) {
            for (flatArg in arg.args) {
                constantTerm = mulArithTerm(constantTerm, flatArg, simplifiedArgs)
            }
        } else {
            constantTerm = mulArithTerm(constantTerm, arg, simplifiedArgs)
        }
    }

    if (simplifiedArgs.isEmpty() || constantTerm.isZero()) {
        return numericValue(constantTerm, args.first().sort)
    }

    if (constantTerm != RealValue.one) {
        // prefer constant to be the first argument
        val firstArg = simplifiedArgs.first()
        simplifiedArgs[0] = numericValue(constantTerm, firstArg.sort)
        simplifiedArgs.add(firstArg)
    }

    if (simplifiedArgs.size == 1) {
        return simplifiedArgs.single()
    }

    return mkArithMulNoSimplify(simplifiedArgs)
}

fun <T : KArithSort> KContext.simplifyArithDiv(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> {
    val rValue = rhs.toRealValue()

    if (rValue != null && !rValue.isZero()) {
        if (rValue == RealValue.one) {
            return lhs
        }

        if (rValue == RealValue.minusOne) {
            return simplifyArithUnaryMinus(lhs)
        }

        if (lhs is KIntNumExpr) {
            return mkIntNum(lhs.bigIntegerValue / rValue.numerator).uncheckedCast()
        }

        if (lhs is KRealNumExpr) {
            val value = lhs.toRealValue().div(rValue)
            return numericValue(value, lhs.sort).uncheckedCast()
        }
    }
    return mkArithDivNoSimplify(lhs, rhs)
}

fun <T : KArithSort> KContext.simplifyArithPower(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> {
    val lValue = lhs.toRealValue()
    val rValue = rhs.toRealValue()

    if (lValue != null && rValue != null) {
        tryEvalArithPower(lValue, rValue)?.let { return castRealValue(it, lhs.sort) }
    }

    if (lValue == RealValue.one) {
        return lhs
    }

    if (rValue == RealValue.one) {
        return lhs
    }

    return mkArithPowerNoSimplify(lhs, rhs)
}

private fun tryEvalArithPower(base: RealValue, power: RealValue): RealValue? = when {
    base.isZero() && power.isZero() -> null
    power.isZero() -> RealValue.one
    base.isZero() -> RealValue.zero
    base == RealValue.one -> RealValue.one
    power == RealValue.one -> base
    power == RealValue.minusOne -> base.inverse()
    else -> power.smallIntValue()?.let { powerIntValue ->
        if (powerIntValue >= 0) {
            base.pow(powerIntValue)
        } else {
            base.inverse().pow(powerIntValue.absoluteValue)
        }
    }
}

private fun <T : KArithSort> KContext.castRealValue(value: RealValue, sort: T): KExpr<T> = when (sort) {
    realSort -> numericValue(value, sort)
    intSort -> mkIntNum(value.numerator / value.denominator).uncheckedCast()
    else -> error("Unexpected arith sort: $sort")
}

fun <T : KArithSort> KContext.simplifyArithLe(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> {
    if (lhs is KIntNumExpr && rhs is KIntNumExpr) {
        return (lhs <= rhs).expr
    }
    if (lhs is KRealNumExpr && rhs is KRealNumExpr) {
        return (lhs.toRealValue() <= rhs.toRealValue()).expr
    }
    return mkArithLeNoSimplify(lhs, rhs)
}

fun <T : KArithSort> KContext.simplifyArithLt(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> {
    if (lhs is KIntNumExpr && rhs is KIntNumExpr) {
        return (lhs < rhs).expr
    }
    if (lhs is KRealNumExpr && rhs is KRealNumExpr) {
        return (lhs.toRealValue() < rhs.toRealValue()).expr
    }
    return mkArithLtNoSimplify(lhs, rhs)
}

fun <T : KArithSort> KContext.simplifyArithGe(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    simplifyArithLe(rhs, lhs)

fun <T : KArithSort> KContext.simplifyArithGt(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    simplifyArithLt(rhs, lhs)


fun KContext.simplifyIntMod(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KExpr<KIntSort> {
    if (rhs is KIntNumExpr) {
        val rValue = rhs.bigIntegerValue

        if (rValue == BigInteger.ONE || rValue == -BigInteger.ONE) {
            return mkIntNum(0)
        }

        if (rValue != BigInteger.ZERO && lhs is KIntNumExpr) {
            return mkIntNum(evalIntMod(lhs.bigIntegerValue, rValue))
        }
    }
    return mkIntModNoSimplify(lhs, rhs)
}

/**
 * Eval integer mod wrt Int theory rules.
 * */
private fun evalIntMod(a: BigInteger, b: BigInteger): BigInteger {
    val remainder = a.rem(b)
    if (remainder >= BigInteger.ZERO) return remainder
    return if (b >= BigInteger.ZERO) remainder + b else remainder - b
}

fun KContext.simplifyIntRem(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KExpr<KIntSort> {
    if (rhs is KIntNumExpr) {
        val rValue = rhs.bigIntegerValue

        if (rValue == BigInteger.ONE || rValue == -BigInteger.ONE) {
            return mkIntNum(0)
        }

        if (rValue != BigInteger.ZERO && lhs is KIntNumExpr) {
            return mkIntNum(evalIntRem(lhs.bigIntegerValue, rValue))
        }
    }
    return mkIntRemNoSimplify(lhs, rhs)
}

/**
 * Eval integer rem wrt Int theory rules.
 * */
private fun evalIntRem(a: BigInteger, b: BigInteger): BigInteger {
    val mod = evalIntMod(a, b)
    return if (b >= BigInteger.ZERO) mod else -mod
}

fun KContext.simplifyIntToReal(arg: KExpr<KIntSort>): KExpr<KRealSort> {
    if (arg is KIntNumExpr) {
        return mkRealNum(arg)
    }
    return mkIntToRealNoSimplify(arg)
}

fun KContext.simplifyRealIsInt(arg: KExpr<KRealSort>): KExpr<KBoolSort> {
    if (arg is KRealNumExpr) {
        return (arg.toRealValue().denominator == BigInteger.ONE).expr
    }

    // (isInt (int2real x)) ==> true
    if (arg is KToRealIntExpr) {
        return trueExpr
    }

    return mkRealIsIntNoSimplify(arg)
}

fun KContext.simplifyRealToInt(arg: KExpr<KRealSort>): KExpr<KIntSort> {
    if (arg is KRealNumExpr) {
        val realValue = arg.toRealValue()
        return mkIntNum(realValue.numerator / realValue.denominator)
    }

    // (real2int (int2real x)) ==> x
    if (arg is KToRealIntExpr) {
        return arg.arg
    }

    return mkRealToIntNoSimplify(arg)
}


private fun <T : KArithSort> addArithTerm(value: RealValue, term: KExpr<T>, terms: MutableList<KExpr<T>>): RealValue {
    if (term is KIntNumExpr) {
        return value.add(term.toRealValue())
    }

    if (term is KRealNumExpr) {
        return value.add(term.toRealValue())
    }

    terms += term
    return value
}

private fun <T : KArithSort> mulArithTerm(value: RealValue, term: KExpr<T>, terms: MutableList<KExpr<T>>): RealValue {
    if (term is KIntNumExpr) {
        return value.mul(term.toRealValue())
    }

    if (term is KRealNumExpr) {
        return value.mul(term.toRealValue())
    }

    terms += term
    return value
}

private fun <T : KArithSort> KExpr<T>.toRealValue(): RealValue? = when (this) {
    is KIntNumExpr -> toRealValue()
    is KRealNumExpr -> toRealValue()
    else -> null
}
