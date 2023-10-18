package io.ksmt.utils

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.*
import io.ksmt.utils.BvUtils.bigIntValue
import io.ksmt.utils.BvUtils.isBvOne
import java.math.BigInteger

object Permutations {
    fun <T> getPermutations(set: Set<T>): Set<List<T>> {
        fun <T> allPermutations(list: List<T>): Set<List<T>> {
            val result: MutableSet<List<T>> = mutableSetOf()
            if (list.isEmpty())
                result.add(list)
            for (item in list) {
                allPermutations(list - item).forEach { tail ->
                    result.add(tail + item)
                }
            }
            return result
        }

        if (set.isEmpty()) return setOf(emptyList())
        return allPermutations(set.toList())
    }
}

fun KBitVecValue<*>.gcd(ctx: KContext, bvSize: UInt, number: BigInteger):
        KBitVecValue<*> {
    val bvValue = this.bigIntValue()
    val bigIntegerGCD = bvValue.gcd(number)
    return ctx.mkBv(bigIntegerGCD, bvSize)
}

fun sameDecl(expr: KExpr<*>, bound: KDecl<*>): Boolean =
    expr is KConst<*> && expr.decl == bound

fun occursInExponentialExpression(bound: KDecl<*>, assertion: KExpr<*>): Boolean =
    assertion is KBvShiftLeftExpr<*> && sameDecl(assertion.shift, bound)

fun hasLinearCoefficient(bound: KDecl<*>, assertion: KExpr<*>): Pair<Boolean, KBitVecValue<*>?>
{
    val arg0: KExpr<*>
    val arg1: KExpr<*>
    when (assertion) {
        is KBvMulExpr<*> -> {
            arg0 = assertion.arg0
            arg1 = assertion.arg1
        }

        is KBvMulNoOverflowExpr<*> -> {
            arg0 = assertion.arg0
            arg1 = assertion.arg1
        }

        else -> return false to null
    }
    val argPairs = arrayOf((arg0 to arg1), (arg1 to arg0))
    for ((arg, coef) in argPairs)
        if (sameDecl(arg, bound))
            if (coef is KBitVecValue<*> && !coef.isBvOne())
                return true to coef
    return false to null
}
