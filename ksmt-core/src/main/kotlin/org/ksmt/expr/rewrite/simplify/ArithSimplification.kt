package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.expr.KAddArithExpr
import org.ksmt.expr.KExpr
import org.ksmt.expr.KIntNumExpr
import org.ksmt.expr.KMulArithExpr
import org.ksmt.expr.KRealNumExpr
import org.ksmt.expr.KToRealIntExpr
import org.ksmt.expr.KUnaryMinusArithExpr
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.utils.ArithUtils.RealValue
import org.ksmt.utils.ArithUtils.bigIntegerValue
import org.ksmt.utils.ArithUtils.compareTo
import org.ksmt.utils.ArithUtils.modWithNegativeNumbers
import org.ksmt.utils.ArithUtils.numericValue
import org.ksmt.utils.ArithUtils.toRealValue
import org.ksmt.utils.uncheckedCast
import java.math.BigInteger

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
    val rValue = when (rhs) {
        is KIntNumExpr -> rhs.toRealValue()
        is KRealNumExpr -> rhs.toRealValue()
        else -> null
    }

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

@Suppress("ForbiddenComment")
fun <T : KArithSort> KContext.simplifyArithPower(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> {
    // todo: evaluate arith power
    return mkArithPowerNoSimplify(lhs, rhs)
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
            return mkIntNum(modWithNegativeNumbers(lhs.bigIntegerValue, rValue))
        }
    }
    return mkIntModNoSimplify(lhs, rhs)
}

fun KContext.simplifyIntRem(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KExpr<KIntSort> {
    if (rhs is KIntNumExpr) {
        val rValue = rhs.bigIntegerValue

        if (rValue == BigInteger.ONE || rValue == -BigInteger.ONE) {
            return mkIntNum(0)
        }

        if (rValue != BigInteger.ZERO && lhs is KIntNumExpr) {
            return mkIntNum(lhs.bigIntegerValue.rem(rValue))
        }
    }
    return mkIntRemNoSimplify(lhs, rhs)
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
