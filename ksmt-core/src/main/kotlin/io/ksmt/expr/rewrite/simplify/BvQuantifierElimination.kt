package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.*
import io.ksmt.expr.rewrite.KBvQETransformer
import io.ksmt.sort.*
import io.ksmt.expr.rewrite.KExprCollector
import io.ksmt.expr.rewrite.KQuantifierSubstitutor
import io.ksmt.utils.*
import io.ksmt.utils.BvUtils.bvMaxValueUnsigned
import io.ksmt.utils.BvUtils.bvZero
import io.ksmt.utils.BvUtils.bvOne
import io.ksmt.utils.BvUtils.isBvOne

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

// p.5, 1st and 2nd steps: https://www.researchgate.net/publication/228686393_On_the_expansion_N_2x_of_Presburger_arithmetic
fun qePreprocess(ctx: KContext, body: KExpr<KBoolSort>, bound: KDecl<*>) :
        Pair<KExpr<KBoolSort>, List<KDecl<*>>?> {
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
        var conditionChanged = false
        val argList = mutableListOf<KExpr<KBvSort>>()

        for (item in mutableListOf(lessExpr, greaterExpr ?: BvConstants.bvMaxValueUnsigned)) {
            if (item is KAndExpr) {
                condition = mkAnd(item.args[0], condition)
                conditionChanged = true
                argList.add(item.args[1].uncheckedCast())
            } else {
                argList.add(item.uncheckedCast())
            }
        }

        val lessOrEqual = mkBvUnsignedLessOrEqualExpr(argList[0], argList[1])
        val newInequality: KExpr<KBoolSort> = if (!conditionChanged) lessOrEqual else
            mkAnd(condition, lessOrEqual)
        return newInequality
    }

    fun orderInequalities(permutation: List<KExpr<KBvSort>>):
            Array<KExpr<KBoolSort>> {
        var orderedInequalities = arrayOf<KExpr<KBoolSort>>()

        if (permutation.isNotEmpty()) {
            var lastExpr = permutation[0]
            if (permutation.size == 1)
                return arrayOf(createInequality(lastExpr, null))
            for (expr in permutation) {
                val newInequality = createInequality(lastExpr, expr)
                orderedInequalities += newInequality
                lastExpr = expr
            }
        }
        return orderedInequalities
    }

    fun getOneCoefficientExpressions(coefficientExpressions: MutableSet<KExpr<*>>, coefficient: KBitVecValue<*>?):
            MutableSet<KExpr<*>> {
        val resultSet = mutableSetOf<KExpr<*>>()
        for (expr in coefficientExpressions) {
            val (isLinear, coef) = hasLinearCoefficient(this, bound, expr)
            if (isLinear && coefficient == coef)
                resultSet.add(expr)
        }
        return resultSet
    }

    var resultArray = arrayOf<KExpr<KBoolSort>>()
    var freeArray = arrayOf<KExpr<KBoolSort>>()
    when (body) {
        is KAndExpr, is KNotExpr, is KBvUnsignedLessOrEqualExpr<*> -> {
            val bodyArgs: MutableList<KExpr<*>> = when (body) {
                is KAndExpr -> body.args.toMutableList()
                else -> mutableListOf(body)
            }
            val bodyArgMap = mutableMapOf<KExpr<*>, Pair<KExpr<*>?, Boolean?>>()
            while (bodyArgs.isNotEmpty()) {
                val curExpr = bodyArgs[0]
                val (freeSubExpr, isLess) = getFreeSubExpr(ctx, curExpr, bound)
                if (isLess == null) {
                    when (freeSubExpr) {
                        null -> freeArray += curExpr as KExpr<KBoolSort>
                        is KAndExpr -> bodyArgs += freeSubExpr.args
                    }
                }
                else
                    bodyArgMap[curExpr] = freeSubExpr to isLess
                bodyArgs.removeFirst()
            }

            while (bodyArgMap.isNotEmpty()){
                val bodyArg = bodyArgMap.keys.first()
                val coefficient = hasLinearCoefficient(this, bound, bodyArg).second
                val oneCoefExprSet = getOneCoefficientExpressions(bodyArgMap.keys, coefficient)

                val leSet = mutableSetOf<KExpr<*>>()
                val geSet = mutableSetOf<KExpr<*>>()
                for (curExpr in oneCoefExprSet) {
                    val (freeSubExpr, isLess) = bodyArgMap[curExpr]!!
                    if (isLess!!)
                        leSet.add(freeSubExpr!!)
                    else
                        geSet.add(freeSubExpr!!)

                    bodyArgMap.remove(curExpr)
                }
                val lePermutations = Permutations.getPermutations(leSet)
                val gePermutations = Permutations.getPermutations(geSet)
                var orList = arrayOf<KExpr<KBoolSort>>()

                for (leP in lePermutations) {
                    @Suppress("UNCHECKED_CAST")
                    leP as List<KExpr<KBvSort>>
                    val orderedLe = orderInequalities(leP)

                    for (geP in gePermutations) {
                        @Suppress("UNCHECKED_CAST")
                        geP as List<KExpr<KBvSort>>
                        val orderedGe = orderInequalities(geP)

                        var coefficientExpr: KExpr<KBoolSort> = mkTrue()
                        if (!coefficient!!.isBvOne()) {
                            var addNotOverflow: KExpr<KBoolSort> = mkTrue()
                            var remainderIsZero: KExpr<KBoolSort> = mkTrue()
                            var nextMultiple: KExpr<KBvSort> = BvConstants.bvZero.uncheckedCast()

                            var withOverflowCheck: KExpr<KBoolSort> = mkTrue()
                            if (leP.isNotEmpty()) {
                                val initLastLe = leP[leP.lastIndex]
                                var lastLe = initLastLe
                                if (initLastLe is KAndExpr) {
                                    lastLe = initLastLe.args[1].uncheckedCast()
                                    withOverflowCheck = mkAnd(withOverflowCheck, initLastLe.args[0])
                                }

                                val bigIntegerBvSize = powerOfTwo(BvConstants.bvSize).toInt().toBigInteger()
                                val gcd = coefficient.gcd(ctx, BvConstants.bvSize, bigIntegerBvSize)
                                val remainder = mkBvUnsignedRemExpr(lastLe, gcd.uncheckedCast())

                                remainderIsZero =
                                    mkBvUnsignedLessOrEqualExpr(remainder, BvConstants.bvZero.uncheckedCast())
                                val gcdMinusRemainder = mkBvSubExpr(gcd.uncheckedCast(), remainder)
                                addNotOverflow = mkBvAddNoOverflowExpr(lastLe, gcdMinusRemainder, false)
                                nextMultiple = mkBvAddExpr(lastLe, gcdMinusRemainder)
                            }
                            var fstGe = BvConstants.bvMaxValueUnsigned
                            if (geP.isNotEmpty()) {
                                val initFstGe = geP[0]
                                if (initFstGe is KAndExpr) {
                                    fstGe = initFstGe.args[1].uncheckedCast()
                                    withOverflowCheck = mkAnd(withOverflowCheck, initFstGe.args[0])
                                }
                            }
                            val isNextLess = mkBvUnsignedLessExpr(nextMultiple, fstGe.uncheckedCast())
                            coefficientExpr = mkAnd(withOverflowCheck, isNextLess, mkOr(addNotOverflow, remainderIsZero))
                        }
                        orList += if (leP.isEmpty() or geP.isEmpty())
                            mkAnd(*orderedLe, *orderedGe, coefficientExpr)
                        else {
                            val middleLe = createInequality(leP[leP.lastIndex], geP[0])
                            mkAnd(*orderedLe, *orderedGe, middleLe, coefficientExpr)
                        }
                    }
                }
                resultArray += mkAnd(mkOr(*orList))
            }
        }
        else -> assert(false) {
            "Expression ${body}:${body.javaClass.typeName} " +
                    "is not in required form."
        }
    }
    return mkAnd(*resultArray, *freeArray)
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
            val condition = mkBvUnsignedGreaterOrEqualExpr(curExpr.arg0, bvOneExpr)
            val trueBranch = mkBvSubExpr(curExpr.arg0, bvOneExpr)
            val newFreeSubExpr = mkAndNoSimplify(condition, trueBranch.uncheckedCast())
            newFreeSubExpr to false // bvult
        }
        else
            curExpr.arg0 to true // bvuge
    }
    return if (expr is KNotExpr) {
        val condition: KExpr<KBoolSort> = mkBvUnsignedLessExpr(
            curExpr.arg1,
            BvConstants.bvMaxValueUnsigned.uncheckedCast()
        )
        val trueBranch = mkBvAddExpr(curExpr.arg1, bvOneExpr)
        val newFreeSubExpr = mkAndNoSimplify(condition, trueBranch.uncheckedCast())
        newFreeSubExpr to true // bvugt
    }
    else
        curExpr.arg1 to false // bvule
}
