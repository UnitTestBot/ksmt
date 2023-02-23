package org.ksmt.solver.model

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KArray2Lambda
import org.ksmt.expr.KArray3Lambda
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KArrayNLambda
import org.ksmt.expr.KConst
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFunctionApp
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.expr.KInterpretedValue
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.rewrite.KExprSubstitutor
import org.ksmt.expr.rewrite.KExprUninterpretedDeclCollector.Companion.collectUninterpretedDeclarations
import org.ksmt.expr.rewrite.simplify.KExprSimplifier
import org.ksmt.expr.rewrite.simplify.simplifyExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.model.DefaultValueSampler.Companion.sampleValue
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.utils.asExpr
import org.ksmt.utils.uncheckedCast

open class KModelEvaluator(
    ctx: KContext,
    private val model: KModel,
    private val isComplete: Boolean,
    private val quantifiedVars: Set<KDecl<*>> = emptySet()
) : KExprSimplifier(ctx) {
    private val evaluatedFunctionApp: MutableMap<Pair<KDecl<*>, List<KExpr<*>>>, KExpr<*>> = hashMapOf()
    private val evaluatedFunctionArray: MutableMap<KDecl<*>, KExpr<*>> = hashMapOf()

    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> =
        simplifyExpr(expr, expr.args) { args ->
            /**
             * Don't evaluate expr when it is quantified since
             * it is definitely not present in the model.
             * */
            if (expr.decl in quantifiedVars) {
                return@simplifyExpr expr.decl.apply(args)
            }

            evalFunction(expr.decl, args).also { rewrite(it) }
        }

    override fun <D : KSort, R : KSort> transformSelect(array: KExpr<KArraySort<D, R>>, index: KExpr<D>): KExpr<R> =
        super.transformSelect(tryEvalArrayConst(array), index)

    override fun <D0 : KSort, D1 : KSort, R : KSort> transformSelect(
        array: KExpr<KArray2Sort<D0, D1, R>>, index0: KExpr<D0>, index1: KExpr<D1>
    ): KExpr<R> = super.transformSelect(tryEvalArrayConst(array), index0, index1)

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transformSelect(
        array: KExpr<KArray3Sort<D0, D1, D2, R>>, index0: KExpr<D0>, index1: KExpr<D1>, index2: KExpr<D2>
    ): KExpr<R> = super.transformSelect(tryEvalArrayConst(array), index0, index1, index2)

    override fun <R : KSort> transformSelect(array: KExpr<KArrayNSort<R>>, indices: List<KExpr<KSort>>): KExpr<R> =
        super.transformSelect(tryEvalArrayConst(array), indices)

    // If base array in uninterpreted, try to replace it with model value
    private fun <A : KArraySortBase<R>, R : KSort> tryEvalArrayConst(array: KExpr<A>): KExpr<A> {
        if (array !is KConst<A>) return array
        val interpretation = model.interpretation(array.decl) ?: return array
        if (interpretation.entries.isNotEmpty()) return array
        return interpretation.default ?: array
    }

    override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KFunctionAsArray<A, R>): KExpr<A> {
        // No way to evaluate f when it is quantified in (as-array f)
        if (expr.function in quantifiedVars) {
            return expr
        }

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

            val usedDeclarations = interpretation.usedDeclarations()

            // argument value is unused in function interpretation.
            if (interpretation.vars.all { it !in usedDeclarations }) {
                return evalArrayInterpretation(expr.sort, interpretation)
            }

            val evaluated = evalFunction(expr.function, interpretation.vars.map { ctx.mkConstApp(it) })

            return when (expr.sort as KArraySortBase<R>) {
                is KArraySort<*, *> -> ctx.mkArrayLambda(
                    interpretation.vars.single(),
                    evaluated
                ).uncheckedCast()

                is KArray2Sort<*, *, *> -> ctx.mkArrayLambda(
                    interpretation.vars.first(),
                    interpretation.vars.last(),
                    evaluated
                ).uncheckedCast()

                is KArray3Sort<*, *, *, *> -> ctx.mkArrayLambda(
                    interpretation.vars[0],
                    interpretation.vars[1],
                    interpretation.vars[2],
                    evaluated
                ).uncheckedCast()

                is KArrayNSort<*> -> ctx.mkArrayLambda(
                    interpretation.vars,
                    evaluated
                ).uncheckedCast()
            }
        }
        return evaluatedArray.asExpr(expr.sort).also { rewrite(it) }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>> =
        transformQuantifiedExpression(setOf(expr.indexVarDecl), expr.body) { body ->
            ctx.simplifyArrayLambda(expr.indexVarDecl, body)
        }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Lambda<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> =
        transformQuantifiedExpression(setOf(expr.indexVar0Decl, expr.indexVar1Decl), expr.body) { body ->
            ctx.simplifyArrayLambda(expr.indexVar0Decl, expr.indexVar1Decl, body)
        }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Lambda<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> =
        transformQuantifiedExpression(
            setOf(expr.indexVar0Decl, expr.indexVar1Decl, expr.indexVar2Decl),
            expr.body
        ) { body ->
            ctx.simplifyArrayLambda(expr.indexVar0Decl, expr.indexVar1Decl, expr.indexVar2Decl, body)
        }

    override fun <R : KSort> transform(expr: KArrayNLambda<R>): KExpr<KArrayNSort<R>> =
        transformQuantifiedExpression(expr.indexVarDeclarations.toSet(), expr.body) { body ->
            ctx.simplifyArrayLambda(expr.indexVarDeclarations, body)
        }

    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> =
        transformQuantifiedExpression(expr.bounds.toSet(), expr.body) { body ->
            ctx.simplifyExistentialQuantifier(expr.bounds, body)
        }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> =
        transformQuantifiedExpression(expr.bounds.toSet(), expr.body) { body ->
            ctx.simplifyUniversalQuantifier(expr.bounds, body)
        }

    private inline fun <B : KSort, T : KSort> transformQuantifiedExpression(
        quantifiedVars: Set<KDecl<*>>,
        body: KExpr<B>,
        crossinline quantifierBuilder: (KExpr<B>) -> KExpr<T>
    ): KExpr<T> {
        val allQuantifiedVars = this.quantifiedVars.union(quantifiedVars)
        val quantifierBodyEvaluator = KModelEvaluator(ctx, model, isComplete, allQuantifiedVars)
        val evaluatedBody = quantifierBodyEvaluator.apply(body)
        return quantifierBuilder(evaluatedBody)
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

    private fun <A : KArraySortBase<R>, R : KSort> evalArrayInterpretation(
        sort: A,
        interpretation: KModel.KFuncInterp<R>
    ): KExpr<A> = when (sort as KArraySortBase<R>) {
        is KArraySort<*, R> -> sort.evalArrayInterpretation(interpretation) { array: KExpr<KArraySort<KSort, R>>, args, value ->
            mkArrayStore(array, args.single(), value)
        }

        is KArray2Sort<*, *, *> -> sort.evalArrayInterpretation(interpretation) { array: KExpr<KArray2Sort<KSort, KSort, R>>, args, value ->
            mkArrayStore(array, args.first(), args.last(), value)
        }

        is KArray3Sort<*, *, *, *> -> sort.evalArrayInterpretation(interpretation) { array: KExpr<KArray3Sort<KSort, KSort, KSort, R>>, args, value ->
            mkArrayStore(array, args[0], args[1], args[2], value)
        }

        is KArrayNSort<*> -> sort.evalArrayInterpretation(interpretation) { array: KExpr<KArrayNSort<R>>, args, value ->
            mkArrayStore(array, args, value)
        }
    }

    private inline fun <A : KArraySortBase<R>, R : KSort, reified S : KArraySortBase<R>> A.evalArrayInterpretation(
        interpretation: KModel.KFuncInterp<R>,
        mkEntryStore: KContext.(KExpr<S>, List<KExpr<KSort>>, KExpr<R>) -> KExpr<S>
    ): KExpr<A> = with(ctx) {
        val defaultValue = interpretation.default ?: completeModelValue(range)
        val defaultArray: KExpr<A> = mkArrayConst(this@evalArrayInterpretation, defaultValue)

        interpretation.entries.foldRight(defaultArray) { entry, acc ->
            mkEntryStore(acc.uncheckedCast(), entry.args.uncheckedCast(), entry.value).uncheckedCast()
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

            val resolvedInterpretation = ctx.resolveFunctionInterpretationComplete(interpretation, args)

            // Interpretation was fully resolved
            if (resolvedInterpretation.entries.isEmpty() && resolvedInterpretation.default != null) {
                return@getOrPut resolvedInterpretation.default
            }

            // Interpretation was not fully resolved --> generate ITE chain
            evalResolvedFunctionInterpretation(resolvedInterpretation, args)
        }
        return evaluated.asExpr(decl.sort)
    }

    private fun <T : KSort> evalResolvedFunctionInterpretation(
        resolvedInterpretation: KModel.KFuncInterp<T>,
        args: List<KExpr<*>>
    ): KExpr<T> = with(ctx) {
        // in case of partial interpretation we can generate any default expr to preserve expression correctness
        val defaultExpr = resolvedInterpretation.default ?: completeModelValue(resolvedInterpretation.sort)
        return resolvedInterpretation.entries.foldRight(defaultExpr) { entry, acc ->
            val argBinding = entry.args.zip(args) { ea, a ->
                val entryArg: KExpr<KSort> = ea.uncheckedCast()
                val actualArg: KExpr<KSort> = a.uncheckedCast()
                mkEq(entryArg, actualArg)
            }
            mkIte(mkAnd(argBinding), entry.value, acc)
        }
    }

    private fun <T : KSort> KContext.resolveFunctionInterpretationComplete(
        interpretation: KModel.KFuncInterp<T>,
        args: List<KExpr<*>>
    ): KModel.KFuncInterp<T> {
        // Replace function parameters vars with actual arguments
        val varSubstitution = KExprSubstitutor(ctx).apply {
            interpretation.vars.zip(args).forEach { (v, a) ->
                val app: KExpr<KSort> = mkConstApp(v).uncheckedCast()
                substitute(app, a.uncheckedCast())
            }
        }

        val argsAreConstants = args.all { it is KInterpretedValue<*> }

        val resolvedEntries = arrayListOf<KModel.KFuncInterpEntry<T>>()
        for (entry in interpretation.entries) {
            val entryArgs = entry.args.map { varSubstitution.apply(it) }
            val entryValue = varSubstitution.apply(entry.value)

            if (resolvedEntries.isEmpty() && entryArgs == args) {
                // We have no possibly matching entries and we found a matched entry
                return KModel.KFuncInterp(
                    decl = interpretation.decl,
                    vars = interpretation.vars,
                    entries = emptyList(),
                    default = entryValue
                )
            }

            val definitelyDontMatch = argsAreConstants && entryArgs.all { it is KInterpretedValue<*> }
            if (definitelyDontMatch) {
                // No need to keep entry, since it doesn't match arguments
                continue
            }

            resolvedEntries += KModel.KFuncInterpEntry(entryArgs, entryValue)
        }

        val resolvedDefault = interpretation.default?.let { varSubstitution.apply(it) }

        return KModel.KFuncInterp(
            decl = interpretation.decl,
            vars = interpretation.vars,
            entries = resolvedEntries,
            default = resolvedDefault
        )
    }

    private fun <T : KSort> completeModelValue(sort: T): KExpr<T> {
        val value = when (sort) {
            is KUninterpretedSort ->
                model.uninterpretedSortUniverse(sort)
                    ?.randomOrNull()
                    ?: sort.sampleValue()

            is KArraySortBase<*> -> {
                val arrayValue = completeModelValue(sort.range)
                ctx.mkArrayConst(sort.uncheckedCast(), arrayValue)
            }

            else -> sort.sampleValue()
        }
        return value.asExpr(sort)
    }
}
