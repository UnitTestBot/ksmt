package io.ksmt.utils

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.*
import io.ksmt.expr.rewrite.KExprCollector
import io.ksmt.expr.rewrite.simplify.BvConstants
import io.ksmt.sort.KBvSort
import io.ksmt.utils.BvUtils.bigIntValue
import io.ksmt.utils.BvUtils.bvOne
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

fun hasLinearCoefficient(ctx: KContext, bound: KDecl<*>, assertion: KExpr<*>): Pair<Boolean, KBitVecValue<*>?>
{
    val mulTerms = KExprCollector.collectDeclarations(assertion) {
            arg -> arg is KBvMulExpr || arg is KBvMulNoOverflowExpr<*>}
    var mulTerm: KExpr<*>? = null
    for (curTerm in mulTerms) {
        val mulTerms = KExprCollector.collectDeclarations(curTerm) { arg -> sameDecl(arg, bound) }
        if (mulTerms.isNotEmpty()) {
            mulTerm = curTerm
            break
        }
    }
    if (mulTerm == null) {
        val linearTerms = KExprCollector.collectDeclarations(assertion) { arg -> sameDecl(arg, bound) }
        return if (linearTerms.isNotEmpty())
            true to ctx.bvOne<KBvSort>(BvConstants.bvSize)
        else
            false to null
    }

    val arg0: KExpr<*>
    val arg1: KExpr<*>
    when (mulTerm) {
        is KBvMulExpr<*> -> {
            arg0 = mulTerm.arg0
            arg1 = mulTerm.arg1
        }

        is KBvMulNoOverflowExpr<*> -> {
            arg0 = mulTerm.arg0
            arg1 = mulTerm.arg1
        }

        else -> return false to null
    }
    val argPairs = arrayOf((arg0 to arg1), (arg1 to arg0))
    for ((arg, coef) in argPairs)
        if (sameDecl(arg, bound))
            if (coef is KBitVecValue<*>)
                return true to coef
    return false to null
}
