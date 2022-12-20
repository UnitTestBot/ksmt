package org.ksmt.solver.model

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFunctionApp
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.expr.rewrite.KExprSubstitutor
import org.ksmt.expr.rewrite.KExprUninterpretedDeclCollector.Companion.collectUninterpretedDeclarations
import org.ksmt.expr.rewrite.simplify.KExprSimplifier
import org.ksmt.expr.rewrite.simplify.simplifyApp
import org.ksmt.solver.KModel
import org.ksmt.solver.model.DefaultValueSampler.Companion.sampleValue
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.utils.asExpr
import org.ksmt.utils.uncheckedCast

open class KModelEvaluator(
    ctx: KContext,
    private val model: KModel,
    private val isComplete: Boolean
) : KExprSimplifier(ctx) {
    private val evaluatedFunctionApp: MutableMap<Pair<KDecl<*>, List<KExpr<*>>>, KExpr<*>> = hashMapOf()
    private val evaluatedFunctionArray: MutableMap<KDecl<*>, KExpr<*>> = hashMapOf()

    @Suppress("UNCHECKED_CAST")
    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> =
        simplifyApp(expr as KApp<T, KExpr<KSort>>) { args ->
            evalFunction(expr.decl, args).also { rewrite(it) }
        }

    override fun <D : KSort, R : KSort> transform(expr: KFunctionAsArray<D, R>): KExpr<KArraySort<D, R>> {
        val evaluatedArray = evaluatedFunctionArray.getOrPut(expr.function) {
            val interpretation = model.interpretation(expr.function)

            if (interpretation == null && !isComplete) {
                // return without cache
                return expr
            }

            if (interpretation == null) {
                // isComplete = true, return and cache
                return@getOrPut completeModelValue(expr.sort)
            }

            when (interpretation.vars.size) {
                0 -> evalArrayInterpretation(expr.sort, interpretation)
                1 -> evalArrayFunction(
                    expr.sort,
                    expr.function,
                    interpretation.vars.single().uncheckedCast(),
                    interpretation
                )
                else -> error("Function ${expr.function} has free vars but used in as-array")
            }
        }
        return evaluatedArray.asExpr(expr.sort).also { rewrite(it) }
    }

    override fun simplifyEqUninterpreted(
        lhs: KExpr<KUninterpretedSort>,
        rhs: KExpr<KUninterpretedSort>
    ): KExpr<KBoolSort> = with(ctx) {
        if (isUninterpretedValue(lhs.sort, lhs) && isUninterpretedValue(lhs.sort, rhs)) {
            return (lhs == rhs).expr
        }
        super.simplifyEqUninterpreted(lhs, rhs)
    }

    override fun areDefinitelyDistinctUninterpreted(
        lhs: KExpr<KUninterpretedSort>,
        rhs: KExpr<KUninterpretedSort>
    ): Boolean {
        if (isUninterpretedValue(lhs.sort, lhs) && isUninterpretedValue(lhs.sort, rhs)) {
            return lhs != rhs
        }
        return super.areDefinitelyDistinctUninterpreted(lhs, rhs)
    }

    private fun isUninterpretedValue(sort: KUninterpretedSort, expr: KExpr<KUninterpretedSort>): Boolean {
        val sortUniverse = model.uninterpretedSortUniverse(sort) ?: return false
        return expr in sortUniverse
    }

    private fun <D : KSort, R : KSort> evalArrayFunction(
        sort: KArraySort<D, R>,
        function: KDecl<R>,
        indexVar: KDecl<D>,
        interpretation: KModel.KFuncInterp<R>
    ): KExpr<KArraySort<D, R>> {
        val usedDeclarations = interpretation.usedDeclarations()

        // argument value is unused in function interpretation.
        if (indexVar !in usedDeclarations) {
            return evalArrayInterpretation(sort, interpretation)
        }

        val index = ctx.mkConstApp(indexVar)
        val evaluated = evalFunction(function, listOf(index))
        return ctx.mkArrayLambda(index.decl, evaluated)
    }

    private fun <D : KSort, R : KSort> evalArrayInterpretation(
        sort: KArraySort<D, R>,
        interpretation: KModel.KFuncInterp<R>
    ): KExpr<KArraySort<D, R>> = with(ctx) {
        val defaultValue = interpretation.default ?: completeModelValue(sort.range)
        val defaultArray: KExpr<KArraySort<D, R>> = mkArrayConst(sort, defaultValue)

        interpretation.entries.foldRight(defaultArray) { entry, acc ->
            val idx = entry.args.single().asExpr(sort.domain)
            acc.store(idx, entry.value)
        }
    }

    private fun KModel.KFuncInterp<*>.usedDeclarations(): Set<KDecl<*>> {
        val result = hashSetOf<KDecl<*>>()
        entries.forEach { entry ->
            result += collectUninterpretedDeclarations(entry.value)
            entry.args.forEach {
                result += collectUninterpretedDeclarations(it)
            }
        }
        default?.also { result += collectUninterpretedDeclarations(it) }
        return result
    }

    private fun <T : KSort> evalFunction(decl: KDecl<T>, args: List<KExpr<*>>): KExpr<T> {
        val evaluated = evaluatedFunctionApp.getOrPut(decl to args) {
            val interpretation = model.interpretation(decl)

            if (interpretation == null && !isComplete) {
                // return without cache
                return ctx.mkApp(decl, args)
            }

            // Check if expr is an uninterpreted value of a sort
            if (interpretation == null && decl.sort is KUninterpretedSort) {
                val universe = model.uninterpretedSortUniverse(decl.sort) ?: emptySet()
                val expr = ctx.mkApp(decl, args)
                if (expr.uncheckedCast() in universe) {
                    return expr
                }
            }

            // isComplete = true, return and cache
            if (interpretation == null) {
                return@getOrPut completeModelValue(decl.sort)
            }

            check(args.size == interpretation.vars.size) {
                "${interpretation.vars.size} arguments expected but ${args.size} provided"
            }

            evalFuncInterp(interpretation, args)
        }
        return evaluated.asExpr(decl.sort)
    }

    @Suppress("UNCHECKED_CAST")
    open fun <T : KSort> evalFuncInterp(
        interpretation: KModel.KFuncInterp<T>,
        args: List<KExpr<*>>
    ): KExpr<T> = with(ctx) {
        val varSubstitution = KExprSubstitutor(ctx).apply {
            interpretation.vars.zip(args).forEach { (v, a) ->
                val app = mkApp(v, emptyList())
                substitute(app as KExpr<KSort>, a as KExpr<KSort>)
            }
        }

        val entries = interpretation.entries.map { entry ->
            KModel.KFuncInterpEntry(
                entry.args.map { varSubstitution.apply(it) },
                varSubstitution.apply(entry.value)
            )
        }

        // in case of partial interpretation we can generate any default expr to preserve expression correctness
        val defaultExpr = interpretation.default ?: completeModelValue(interpretation.sort)
        val default = varSubstitution.apply(defaultExpr)

        return entries.foldRight(default) { entry, acc ->
            val argBinding = mkAnd(entry.args.zip(args) { ea, a -> mkEq(ea as KExpr<KSort>, a as KExpr<KSort>) })
            mkIte(argBinding, entry.value, acc)
        }
    }

    private fun <T : KSort> completeModelValue(sort: T): KExpr<T> {
        val value = when (sort) {
            is KUninterpretedSort ->
                model.uninterpretedSortUniverse(sort)
                    ?.randomOrNull()
                    ?: sort.sampleValue()

            is KArraySort<*, *> -> {
                val arrayValue = completeModelValue(sort.range)
                ctx.mkArrayConst(sort, arrayValue)
            }

            else -> sort.sampleValue()
        }
        return value.asExpr(sort)
    }
}
