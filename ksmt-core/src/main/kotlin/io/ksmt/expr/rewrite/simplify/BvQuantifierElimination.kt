package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.*
import io.ksmt.expr.rewrite.KBvQETransformer
import io.ksmt.sort.*
import io.ksmt.expr.rewrite.KExprCollector
import io.ksmt.expr.rewrite.KQuantifierSubstitutor
import io.ksmt.utils.BvUtils.bvMaxValueUnsigned
import io.ksmt.utils.BvUtils.bvZero
import io.ksmt.utils.BvUtils.bvOne
import io.ksmt.utils.BvUtils.isBvOne
import io.ksmt.utils.Permutations
import io.ksmt.utils.uncheckedCast

object BvConstants {
    var bvSize = 0u
    var bvMaxValueUnsigned: KBitVecValue<*>? = null
    var bvZero: KBitVecValue<*>? = null
    var bvOne: KBitVecValue<*>? = null

    fun init(ctx: KContext, expr: KDecl<*>) = with(ctx)
    {
        if (expr.sort is KBvSort) {
            bvSize = expr.sort.sizeBits
            bvMaxValueUnsigned = bvMaxValueUnsigned<KBvSort>(bvSize)
            bvZero = bvZero<KBvSort>(bvSize)
            bvOne = bvOne<KBvSort>(bvSize)
        } else
            assert(false) { "Unexpected theory." }
    }
}

fun quantifierElimination(ctx: KContext, assertions: List<KExpr<KBoolSort>>) :
        List<KExpr<KBoolSort>> = with(ctx) {
        val qfAssertions = arrayListOf<KExpr<KBoolSort>>()

        for (assertion in assertions) {
            val quantifierExpressions: Set<KExpr<*>> =
                KExprCollector.collectDeclarations(assertion) { it is KQuantifier }
            if (quantifierExpressions.isEmpty())
                qfAssertions.add(assertion)
            for (qExpr in quantifierExpressions) {
                qExpr as KQuantifier

                BvConstants.init(ctx, qExpr.bounds[0])

                val (quantifierExpr, isUniversal) = prepareQuantifier(ctx, qExpr)
                var qfExpr = qeProcess(ctx, quantifierExpr)
                if (isUniversal)
                    qfExpr = mkNot(qfExpr)

                val substitutor = KQuantifierSubstitutor(this).apply {
                    substitute(qExpr, qfExpr)
                }
                val qfAssertion = substitutor.apply(assertion)
                qfAssertions.add(qfAssertion)
            }
        }
        return qfAssertions
    }

fun prepareQuantifier(ctx: KContext, assertion: KQuantifier):
        Pair<KExistentialQuantifier, Boolean> =
    with (ctx) {
        val isUniversal = (assertion is KUniversalQuantifier)
        var body = assertion.body
        val bounds = assertion.bounds
        if (isUniversal)
            body = mkNot(body)
        val quantifierAssertion = mkExistentialQuantifier(body, bounds)
        return quantifierAssertion to isUniversal
    }

fun qePreprocess(ctx: KContext, body: KExpr<KBoolSort>, bound: KDecl<*>) :
        Pair<KExpr<KBoolSort>, List<KDecl<*>>?> {
    // p.5, 1st and 2nd steps
    val collector = KExprCollector
    val boundExpressions = collector.collectDeclarations(body) {
            arg -> sameDecl(arg, bound)}
    if (boundExpressions.isEmpty())
        return body to null

    val (newBody, bounds) = @Suppress("UNCHECKED_CAST")
    KBvQETransformer.transformBody(body, bound as KDecl<KBvSort>)
    return newBody to bounds
}

fun qeProcess(ctx: KContext, assertion: KExistentialQuantifier): KExpr<KBoolSort> {
    var qfAssertion: KExpr<KBoolSort> = assertion
    var bounds = assertion.bounds
    var body = assertion.body
    for (curBound in bounds.reversed()) {
        val (newBody, newBounds) = qePreprocess(ctx, body, curBound)
        qfAssertion = if (newBounds != null) {
            if (newBounds.isNotEmpty())
                bounds = newBounds + bounds
            val expExpressions = KExprCollector.collectDeclarations(assertion) { expr ->
                occursInExponentialExpression(curBound, expr)
            }
            if (expExpressions.isEmpty())
                linearQE(ctx, newBody, curBound)
            else
                TODO()
        } else
            newBody
    }

    return qfAssertion
}


fun linearQE(ctx: KContext, body: KExpr<KBoolSort>, bound: KDecl<*>):
        KExpr<KBoolSort> = with(ctx) {

    fun createInequality(lessExpr: KExpr<KBvSort>, greaterExpr: KExpr<KBvSort>?):
            KExpr<KBoolSort> {
        var condition: KExpr<KBoolSort> = mkTrue()
        var expr0 = lessExpr
        if (expr0 is KAndExpr) {
            condition = expr0.args[0]
            expr0 = expr0.args[1].uncheckedCast()
        }
        var expr1 = greaterExpr
        if (expr1 is KAndExpr) {
            condition = mkAnd(expr1.args[0], condition)
            expr1 = expr1.args[1].uncheckedCast()
        }
        val lessOrEqual = if (expr1 == null) mkTrue() else mkBvUnsignedLessOrEqualExpr(expr0, expr1)
        val newInequality: KExpr<KBoolSort> = if (condition is KTrue) lessOrEqual else
            mkIte(condition, mkFalse().uncheckedCast(), lessOrEqual)
        return newInequality
    }

    fun orderInequalities(permutation: List<KExpr<KBvSort>>):
            Array<KExpr<KBoolSort>> {
        var orderedInequalities = arrayOf<KExpr<KBoolSort>>()

        if (permutation.isNotEmpty()) {
            var lastExpr = permutation[0]
            if (permutation.size == 1)
                orderedInequalities += createInequality(lastExpr, null)
            for ((i, expr) in permutation.withIndex()) {
                if (i % 2 == 0)
                    lastExpr = expr
                else {
                    val newInequality = createInequality(lastExpr, expr)
                    orderedInequalities += newInequality
                }
            }
        }
        return orderedInequalities
    }

    val coefficientExpressions = KExprCollector.collectDeclarations(body) { expr ->
        hasLinearCoefficient(bound, expr) }
    if (coefficientExpressions.isNotEmpty())
        TODO()
    var result: KExpr<KBoolSort> = KFalse(ctx)
    var orList = arrayOf<KExpr<KBoolSort>>()

    when (body) {
        is KAndExpr, is KNotExpr, is KBvUnsignedLessOrEqualExpr<*> -> {
            val leSet = mutableSetOf<KExpr<*>>()
            val geSet = mutableSetOf<KExpr<*>>()
            var freeSet = arrayOf<KExpr<KBoolSort>>()
            val args: MutableList<KExpr<KBoolSort>> = when(body) {
                is KAndExpr -> body.args.toMutableList()
                else -> mutableListOf(body)
            }
            while (args.isNotEmpty()) {
                val expr = args[0]
                val (freeSubExpr, isLess) = getFreeSubExpr(ctx, expr, bound)
                if (isLess == null) {
                    when (freeSubExpr)
                    {
                        null -> freeSet += expr
                        is KAndExpr -> args += freeSubExpr.args
                    }
                }
                else if (isLess)
                    leSet.add(freeSubExpr!!)
                else
                    geSet.add(freeSubExpr!!)
                args.removeFirst()
            }
            val lePermutations = Permutations.getPermutations(leSet)
            val gePermutations = Permutations.getPermutations(geSet)
            for (leP in lePermutations) {
                @Suppress("UNCHECKED_CAST")
                leP as List<KExpr<KBvSort>>
                val orderedLe = orderInequalities(leP)

                for (geP in gePermutations) {
                    @Suppress("UNCHECKED_CAST")
                    geP as List<KExpr<KBvSort>>
                    val orderedGe = orderInequalities(geP)

                    orList += if (leP.isEmpty() or geP.isEmpty())
                        mkAnd(*orderedLe, *orderedGe)
                    else {
                        val middleLe = createInequality(leP[leP.lastIndex], geP[0])
                        mkAnd(*orderedLe, *orderedGe, middleLe)
                    }
                }
            }
            result = mkOr(*orList)
            result = mkAnd(result, *freeSet)
        }
        else -> assert(false) { "Expression ${body}:${body.javaClass.typeName} " +
                "is not in required form." }
    }
    return result
}

fun getFreeSubExpr(ctx: KContext, expr: KExpr<*>, bound: KDecl<*>):
        Pair<KExpr<*>?, Boolean?> = with(ctx) {
    val curExpr = if (expr is KNotExpr) expr.arg else expr
    if (curExpr is KAndExpr)
        return expr to null
    else if (curExpr !is KBvUnsignedLessOrEqualExpr<*>)
        assert(false) { "Expression ${curExpr}:${curExpr.javaClass.typeName} " +
                "is not in required form." }

    @Suppress("UNCHECKED_CAST")
    curExpr as KBvUnsignedLessOrEqualExpr<KBvSort>
    val collector = KExprCollector
    val bvOneExpr: KExpr<KBvSort> = BvConstants.bvOne.uncheckedCast()
    var boundExpressions = collector.collectDeclarations(curExpr.arg0) {
        arg -> sameDecl(arg, bound)}
    if (boundExpressions.isEmpty()) {
        boundExpressions = collector.collectDeclarations(curExpr.arg1) {
                arg -> sameDecl(arg, bound)}
        if (boundExpressions.isEmpty())
            return null to null //
        return if (expr is KNotExpr) {
            val condition = mkBvUnsignedLessExpr(curExpr.arg0, bvOneExpr)
            val falseBranch = mkBvSubExpr(curExpr.arg0, bvOneExpr)
            val newFreeSubExpr = mkAnd(condition, falseBranch.uncheckedCast(), order = false)
            newFreeSubExpr to false // bvult
        }
        else
            curExpr.arg0 to true // bvuge
    }
    return if (expr is KNotExpr) {
        val condition: KExpr<KBoolSort> = mkBvUnsignedGreaterOrEqualExpr(
            curExpr.arg1,
            BvConstants.bvMaxValueUnsigned.uncheckedCast()
        )
        val falseBranch = mkBvAddExpr(curExpr.arg1, bvOneExpr)
        val newFreeSubExpr = mkAnd(condition, falseBranch.uncheckedCast(), order = false)
        newFreeSubExpr to true // bvugt
    }
    else
        curExpr.arg1 to false // bvule
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
