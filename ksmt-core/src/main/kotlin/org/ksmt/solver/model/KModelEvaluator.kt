package org.ksmt.solver.model

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFunctionApp
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.expr.rewrite.KExprSubstitutor
import org.ksmt.expr.rewrite.simplify.KExprSimplifier
import org.ksmt.expr.rewrite.simplify.simplifyApp
import org.ksmt.solver.KModel
import org.ksmt.solver.model.DefaultValueSampler.Companion.sampleValue
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort
import org.ksmt.utils.asExpr

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
                return@getOrPut expr.sort.sampleValue()
            }

            check(interpretation.vars.isEmpty()) {
                "Function ${expr.function} has free vars but used in as-array"
            }

            with(ctx) {
                val defaultValue = interpretation.default ?: interpretation.sort.sampleValue()
                val defaultArray: KExpr<KArraySort<D, R>> = mkArrayConst(expr.sort, defaultValue)

                interpretation.entries.foldRight(defaultArray) { entry, acc ->
                    val idx = entry.args.single().asExpr(expr.domainSort)
                    acc.store(idx, entry.value)
                }
            }
        }
        return evaluatedArray.asExpr(expr.sort).also { rewrite(it) }
    }

    private fun <T : KSort> evalFunction(decl: KDecl<T>, args: List<KExpr<*>>): KExpr<T> {
        val evaluated = evaluatedFunctionApp.getOrPut(decl to args) {
            val interpretation = model.interpretation(decl)

            if (interpretation == null && !isComplete) {
                // return without cache
                return ctx.mkApp(decl, args)
            }

            if (interpretation == null) {
                // isComplete = true, return and cache
                return@getOrPut decl.sort.sampleValue()
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
        val defaultExpr = interpretation.default ?: interpretation.sort.sampleValue()
        val default = varSubstitution.apply(defaultExpr)

        return entries.foldRight(default) { entry, acc ->
            val argBinding = mkAnd(entry.args.zip(args) { ea, a -> mkEq(ea as KExpr<KSort>, a as KExpr<KSort>) })
            mkIte(argBinding, entry.value, acc)
        }
    }
}
