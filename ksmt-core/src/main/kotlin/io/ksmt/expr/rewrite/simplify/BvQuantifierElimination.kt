package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.*
import io.ksmt.sort.*
import io.ksmt.expr.rewrite.KExprCollector
import io.ksmt.expr.rewrite.KQuantifierSubstitutor
import io.ksmt.utils.BvUtils.bvMaxValueSigned
import io.ksmt.utils.BvUtils.bvMaxValueUnsigned
import io.ksmt.utils.BvUtils.bvMinValueSigned
import io.ksmt.utils.BvUtils.bvZero
import io.ksmt.utils.BvUtils.isBvOne
import io.ksmt.utils.Permutations
import io.ksmt.utils.uncheckedCast

object BvConstants {
    var bvSize = 0u
    var bvMaxValueSigned: KBitVecValue<*>? = null
    var bvMinValueSigned: KBitVecValue<*>? = null
    var bvMaxValueUnsigned: KBitVecValue<*>? = null
    var bvZero: KBitVecValue<*>? = null

    fun init(ctx: KContext, expr: KDecl<*>) {
        if (expr.sort is KBvSort) {
            bvSize = expr.sort.sizeBits
            bvMaxValueSigned = ctx.bvMaxValueSigned(bvSize)
            bvMinValueSigned = ctx.bvMinValueSigned(bvSize)
            bvMaxValueUnsigned = ctx.bvMaxValueUnsigned(bvSize)
            bvZero = ctx.bvZero(bvSize)
        } else
            assert(false) { "Unexpected theory." }
    }
}

fun quantifierElimination(ctx: KContext, assertions: List<KExpr<KBoolSort>>): List<KExpr<KBoolSort>> =
    with(ctx) {
        val qfAssertions = arrayListOf<KExpr<KBoolSort>>()

        for (assertion in assertions) {
            val quantifierExpressions = KExprCollector.collectDeclarations(assertion) { it is KQuantifier }
            if (quantifierExpressions.isEmpty())
                qfAssertions.add(assertion)
            for (curExpr in quantifierExpressions) {
                val qExpr = curExpr as KQuantifier

                BvConstants.init(ctx, qExpr.bounds[0])

                val (quantifierExpr, isUniversal) = qePreprocess(ctx, qExpr)
                var qfExpr = qeProcess(ctx, quantifierExpr)
                if (isUniversal)
                    qfExpr = mkNot(qfExpr)

                // println(curExpr.body)
                val substitutor = KQuantifierSubstitutor(this).apply {
                    substitute(qExpr, qfExpr)
                }
                val qfAssertion = substitutor.apply(assertion)
                qfAssertions.add(qfAssertion)
            }
        }
        return qfAssertions
    }

fun qePreprocess(ctx: KContext, assertion: KQuantifier):
        Pair<KExistentialQuantifier, Boolean> =
    with (ctx) {
        val isUniversal = (assertion is KUniversalQuantifier)
        var body = assertion.body
        val bounds = assertion.bounds
        if (isUniversal)
            body = mkNot(body)
        body = simplifyNot(body)
        // TODO toRequiredForm(body, bounds) p.5, 1st and 2nd steps
        val quantifierAssertion = mkExistentialQuantifier(body, bounds)
        return quantifierAssertion to isUniversal
    }

fun qeProcess(ctx: KContext, assertion: KExistentialQuantifier): KExpr<KBoolSort> {
    var qfAssertion: KExpr<KBoolSort> = assertion
    val bounds = assertion.bounds
    for (curBound in bounds.reversed()) {
        val expExpressions = KExprCollector.collectDeclarations(assertion) { expr ->
            occursInExponentialExpression(curBound, expr) }
        if (expExpressions.isEmpty())
            qfAssertion = linearQE(ctx, assertion, curBound)
        else
            TODO()
    }

    return qfAssertion
}

fun linearQE(ctx: KContext, assertion: KExistentialQuantifier, bound: KDecl<*>):
        KExpr<KBoolSort>
{
    val coefficientExpressions = KExprCollector.collectDeclarations(assertion) { expr ->
        hasLinearCoefficient(bound, expr) }
    if (coefficientExpressions.isNotEmpty())
        TODO()
    val body = assertion.body

    if (body is KAndExpr) {
        val lessSet = mutableSetOf<KExpr<*>>()
        val greaterSet = mutableSetOf<KExpr<*>>()
        for (expr in body.args) {
            if (isBvNonStrictGreater(expr))
                greaterSet.add(expr)
            if (isBvNonStrictLess(expr))
                lessSet.add(expr)
        }
        val lessPermutations = Permutations.getPermutations(lessSet)
        val greaterPermutations = Permutations.getPermutations(lessSet)
        TODO()
    }

    else if (isBvNonStrictGreater(body) || isBvNonStrictLess(body))
        return KTrue(ctx)

    else
        assert(false) { "Expression is not in required form." }

    return KFalse(ctx)
}

fun sameDecl(expr: KExpr<*>, bound: KDecl<*>): Boolean =
    expr is KConst<*> && expr.decl == bound

fun occursInExponentialExpression(bound: KDecl<*>, assertion: KExpr<*>): Boolean =
    assertion is KBvShiftLeftExpr<*> && sameDecl(assertion.shift, bound)

fun hasLinearCoefficient(bound: KDecl<*>, assertion: KExpr<*>): Boolean
{
    if (assertion is KBvMulExpr<*>) {
        val argPairs = arrayOf((assertion.arg0 to assertion.arg1),
            (assertion.arg1 to assertion.arg0))
        for ((arg, coef) in argPairs)
        if (sameDecl(arg, bound))
            if (coef is KBitVecValue<*> && !coef.isBvOne())
                return true
    }
    return false
}

fun isBvNonStrictGreater(expr: KExpr<*>): Boolean =
    (expr is KBvUnsignedGreaterOrEqualExpr<*>) || (expr is KBvSignedGreaterOrEqualExpr<*>)

fun isBvNonStrictLess(expr: KExpr<*>): Boolean =
    (expr is KBvUnsignedLessOrEqualExpr<*>) || (expr is KBvSignedLessOrEqualExpr<*>)