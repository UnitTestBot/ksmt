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
import io.ksmt.utils.ArithUtils.bigIntegerValue
import io.ksmt.utils.ArithUtils.compareTo
import io.ksmt.utils.ArithUtils.numericValue
import io.ksmt.utils.ArithUtils.toRealValue
import io.ksmt.utils.ArithUtils.RealValue
import io.ksmt.utils.uncheckedCast
import java.math.BigInteger
import kotlin.math.absoluteValue

inline fun <T : KArithSort> KContext.simplifyArithUnaryMinusLight(
    arg: KExpr<T>,
    cont: (KExpr<T>) -> KExpr<T>
): KExpr<T> {
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

    return cont(arg)
}

inline fun <T : KArithSort> KContext.simplifyArithAddLight(
    args: List<KExpr<T>>,
    cont: (List<KExpr<T>>) -> KExpr<T>
): KExpr<T> {
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

    return cont(simplifiedArgs)
}

/** (- a b) ==> (+ a -b) */
inline fun <T : KArithSort> KContext.rewriteArithSub(
    args: List<KExpr<T>>,
    rewriteArithAdd: KContext.(List<KExpr<T>>) -> KExpr<T>,
    rewriteArithUnaryMinus: KContext.(KExpr<T>) -> KExpr<T>
): KExpr<T> = if (args.size == 1) {
    args.single()
} else {
    require(args.isNotEmpty()) {
        "Arith sub requires at least a single argument"
    }

    val simplifiedArgs = arrayListOf(args.first())
    for (arg in args.drop(1)) {
        simplifiedArgs += rewriteArithUnaryMinus(arg)
    }
    rewriteArithAdd(simplifiedArgs)
}

inline fun <T : KArithSort> KContext.simplifyArithMulLight(
    args: List<KExpr<T>>,
    cont: (List<KExpr<T>>) -> KExpr<T>
): KExpr<T> {
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

    return cont(simplifiedArgs)
}

inline fun <T : KArithSort> KContext.simplifyArithLeLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> {
    if (lhs is KIntNumExpr && rhs is KIntNumExpr) {
        return (lhs <= rhs).expr
    }
    if (lhs is KRealNumExpr && rhs is KRealNumExpr) {
        return (lhs.toRealValue() <= rhs.toRealValue()).expr
    }
    return cont(lhs, rhs)
}

inline fun <T : KArithSort> KContext.simplifyArithDivLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteArithUnaryMinus: KContext.(KExpr<T>) -> KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    val rValue = rhs.toRealValue()

    if (rValue != null && !rValue.isZero()) {
        if (rValue == RealValue.one) {
            return lhs
        }

        if (rValue == RealValue.minusOne) {
            return rewriteArithUnaryMinus(lhs)
        }

        if (lhs is KIntNumExpr) {
            return mkIntNum(evalIntDiv(lhs.bigIntegerValue, rValue.numerator)).uncheckedCast()
        }

        if (lhs is KRealNumExpr) {
            val value = lhs.toRealValue().div(rValue)
            return numericValue(value, lhs.sort).uncheckedCast()
        }
    }
    return cont(lhs, rhs)
}

inline fun <T : KArithSort> KContext.simplifyArithPowerLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
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

    return cont(lhs, rhs)
}

inline fun <T : KArithSort> KContext.simplifyArithLtLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> {
    if (lhs is KIntNumExpr && rhs is KIntNumExpr) {
        return (lhs < rhs).expr
    }
    if (lhs is KRealNumExpr && rhs is KRealNumExpr) {
        return (lhs.toRealValue() < rhs.toRealValue()).expr
    }
    return cont(lhs, rhs)
}

/** (>= a b) ==> (<= b a) */
inline fun <T : KArithSort> KContext.rewriteArithGe(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteArithLe: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = rewriteArithLe(rhs, lhs)

/** (> a b) ==> (< b a) */
inline fun <T : KArithSort> KContext.rewriteArithGt(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteArithLt: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = rewriteArithLt(rhs, lhs)

inline fun KContext.simplifyIntModLight(
    lhs: KExpr<KIntSort>,
    rhs: KExpr<KIntSort>,
    cont: (KExpr<KIntSort>, KExpr<KIntSort>) -> KExpr<KIntSort>
): KExpr<KIntSort> {
    if (rhs is KIntNumExpr) {
        val rValue = rhs.bigIntegerValue

        if (rValue == BigInteger.ONE || rValue == -BigInteger.ONE) {
            return mkIntNum(0)
        }

        if (rValue != BigInteger.ZERO && lhs is KIntNumExpr) {
            return mkIntNum(evalIntMod(lhs.bigIntegerValue, rValue))
        }
    }
    return cont(lhs, rhs)
}

inline fun KContext.simplifyIntRemLight(
    lhs: KExpr<KIntSort>,
    rhs: KExpr<KIntSort>,
    cont: (KExpr<KIntSort>, KExpr<KIntSort>) -> KExpr<KIntSort>
): KExpr<KIntSort> {
    if (rhs is KIntNumExpr) {
        val rValue = rhs.bigIntegerValue

        if (rValue == BigInteger.ONE || rValue == -BigInteger.ONE) {
            return mkIntNum(0)
        }

        if (rValue != BigInteger.ZERO && lhs is KIntNumExpr) {
            return mkIntNum(evalIntRem(lhs.bigIntegerValue, rValue))
        }
    }
    return cont(lhs, rhs)
}

inline fun KContext.simplifyIntToRealLight(
    arg: KExpr<KIntSort>,
    cont: (KExpr<KIntSort>) -> KExpr<KRealSort>
): KExpr<KRealSort> =
    if (arg is KIntNumExpr) {
        mkRealNum(arg)
    } else {
        cont(arg)
    }

inline fun KContext.simplifyRealIsIntLight(
    arg: KExpr<KRealSort>,
    cont: (KExpr<KRealSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> {
    if (arg is KRealNumExpr) {
        return (arg.toRealValue().denominator == BigInteger.ONE).expr
    }

    // (isInt (int2real x)) ==> true
    if (arg is KToRealIntExpr) {
        return trueExpr
    }

    return cont(arg)
}

inline fun KContext.simplifyRealToIntLight(
    arg: KExpr<KRealSort>,
    cont: (KExpr<KRealSort>) -> KExpr<KIntSort>
): KExpr<KIntSort> {
    if (arg is KRealNumExpr) {
        val realValue = arg.toRealValue()
        return mkIntNum(evalIntDiv(realValue.numerator, realValue.denominator))
    }

    // (real2int (int2real x)) ==> x
    if (arg is KToRealIntExpr) {
        return arg.arg
    }

    return cont(arg)
}

inline fun KContext.simplifyEqIntLight(
    lhs: KExpr<KIntSort>,
    rhs: KExpr<KIntSort>,
    cont: (KExpr<KIntSort>, KExpr<KIntSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> {
    if (lhs == rhs) return trueExpr

    if (lhs is KIntNumExpr && rhs is KIntNumExpr) {
        return (lhs.compareTo(rhs) == 0).expr
    }

    return cont(lhs, rhs)
}

inline fun KContext.simplifyEqRealLight(
    lhs: KExpr<KRealSort>,
    rhs: KExpr<KRealSort>,
    cont: (KExpr<KRealSort>, KExpr<KRealSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> {
    if (lhs == rhs) return trueExpr

    if (lhs is KRealNumExpr && rhs is KRealNumExpr) {
        return (lhs.toRealValue().compareTo(rhs.toRealValue()) == 0).expr
    }

    return cont(lhs, rhs)
}

fun <T : KArithSort> addArithTerm(
    value: RealValue,
    term: KExpr<T>,
    terms: MutableList<KExpr<T>>
): RealValue {
    if (term is KIntNumExpr) {
        return value.add(term.toRealValue())
    }

    if (term is KRealNumExpr) {
        return value.add(term.toRealValue())
    }

    terms += term
    return value
}

fun <T : KArithSort> mulArithTerm(
    value: RealValue,
    term: KExpr<T>,
    terms: MutableList<KExpr<T>>
): RealValue {
    if (term is KIntNumExpr) {
        return value.mul(term.toRealValue())
    }

    if (term is KRealNumExpr) {
        return value.mul(term.toRealValue())
    }

    terms += term
    return value
}

fun tryEvalArithPower(base: RealValue, power: RealValue): RealValue? = when {
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

fun <T : KArithSort> KContext.castRealValue(value: RealValue, sort: T): KExpr<T> = when (sort) {
    realSort -> numericValue(value, sort)
    intSort -> mkIntNum(value.numerator / value.denominator).uncheckedCast()
    else -> error("Unexpected arith sort: $sort")
}

fun <T : KArithSort> KExpr<T>.toRealValue(): RealValue? = when (this) {
    is KIntNumExpr -> toRealValue()
    is KRealNumExpr -> toRealValue()
    else -> null
}

/**
 * Eval integer div wrt Int theory rules.
 * */
fun evalIntDiv(a: BigInteger, b: BigInteger): BigInteger {
    if (a >= BigInteger.ZERO) {
        return a / b
    }
    val (divisionRes, rem) = a.divideAndRemainder(b)
    if (rem == BigInteger.ZERO) {
        return divisionRes
    }

    // round toward zero
    return if (b >= BigInteger.ZERO) {
        divisionRes - BigInteger.ONE
    } else {
        divisionRes + BigInteger.ONE
    }
}

/**
 * Eval integer mod wrt Int theory rules.
 * */
fun evalIntMod(a: BigInteger, b: BigInteger): BigInteger {
    val remainder = a.rem(b)
    if (remainder >= BigInteger.ZERO) return remainder
    return if (b >= BigInteger.ZERO) remainder + b else remainder - b
}

/**
 * Eval integer rem wrt Int theory rules.
 * */
fun evalIntRem(a: BigInteger, b: BigInteger): BigInteger {
    val mod = evalIntMod(a, b)
    return if (b >= BigInteger.ZERO) mod else -mod
}
