package org.ksmt.solver.model

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KArraySelect
import org.ksmt.expr.KArrayStore
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
import org.ksmt.sort.KArraySort
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

    private val rewrittenArraySelects = hashMapOf<KArraySelect<*, *>, KArraySelect<*, *>?>()
    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> {
        val rewrittenSelect: KArraySelect<D, R>? = rewrittenArraySelects.getOrPut(expr) {
            ctx.tryEliminateFunctionAsArray(expr)
        }?.uncheckedCast()

        return if (rewrittenSelect == null) {
            super.transform(expr)
        } else {
            simplifyApp(expr, preprocess = { rewrittenSelect }) {
                error("Always preprocessed")
            }
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KFunctionAsArray<D, R>): KExpr<KArraySort<D, R>> {
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

            val idxDecl = interpretation.vars.singleOrNull()
                ?: error("Function ${expr.function} has ${interpretation.vars} vars but used in as-array")

            evalArrayFunction(
                expr.sort,
                expr.function,
                idxDecl.uncheckedCast(),
                interpretation
            )
        }
        return evaluatedArray.asExpr(expr.sort).also { rewrite(it) }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>> =
        transformQuantifiedExpression(setOf(expr.indexVarDecl), expr.body) { body ->
            ctx.simplifyArrayLambda(expr.indexVarDecl, body)
        }

    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> =
        transformQuantifiedExpression(expr.bounds.toSet(), expr.body) { body ->
            ctx.simplifyExistentialQuantifier(expr.bounds, body)
        }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> =
        transformQuantifiedExpression(expr.bounds.toSet(), expr.body) { body ->
            ctx.simplifyUniversalQuantifier(expr.bounds, body)
        }

    private inline fun <B : KSort, T: KSort> transformQuantifiedExpression(
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

    /**
     * Usually, [KFunctionAsArray] will be expanded into large array store chain.
     * In case of array select, we can avoid such expansion and replace
     * (select (as-array f) i) with (f i).
     * */
    private fun <D : KSort, R : KSort> KContext.tryEliminateFunctionAsArray(
        expr: KArraySelect<D, R>
    ): KArraySelect<D, R>? {
        // Unroll stores until we find some base array
        val parentStores = arrayListOf<KArrayStore<D, R>>()
        var base = expr.array
        while (base is KArrayStore<D, R>) {
            parentStores += base
            base = base.array
        }

        // If base array in uninterpreted, try to replace it with model value
        if (base is KConst<KArraySort<D, R>>) {
            val interpretation = model.interpretation(base.decl) ?: return null
            if (interpretation.entries.isNotEmpty()) return null
            base = interpretation.default ?: return null
        }

        if (base !is KFunctionAsArray<D, R>) return null

        /**
         * Replace as-array with (const (f i)) since:
         * 1. we may have parent stores here and we need an array expression
         * 2. (select (const (f i)) i) ==> (f i)
         * */
        val defaultSelectValue = base.function.apply(listOf(expr.index))
        var newArrayBase: KExpr<KArraySort<D, R>> = mkArrayConst(base.sort, defaultSelectValue)

        // Rebuild array
        for (store in parentStores.asReversed()) {
            newArrayBase = newArrayBase.store(store.index, store.value)
        }

        return mkArraySelectNoSimplify(newArrayBase, expr.index)
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
