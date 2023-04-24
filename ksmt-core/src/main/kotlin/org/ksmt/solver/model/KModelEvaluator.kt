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
import org.ksmt.expr.rewrite.simplify.areDefinitelyDistinct
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
    private val evaluatedFunctionApp = hashMapOf<Pair<KDecl<*>, List<KExpr<*>>>, KExpr<*>>()
    private val evaluatedFunctionArray = hashMapOf<KDecl<*>, KExpr<*>>()
    private val resolvedFunctionInterpretations = hashMapOf<KModel.KFuncInterp<*>, ResolvedFunctionInterpretation<*>>()

    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> {
        println("S transform(expr KFunctionApp $expr ")
        return simplifyExpr(expr, expr.args) { args ->
            /**
             * Don't evaluate expr when it is quantified since
             * it is definitely not present in the model.
             * */
            if (expr.decl in quantifiedVars) {
                return@simplifyExpr expr.decl.apply(args)
            }

            evalFunction(expr.decl, args).also { rewrite(it) }
        }.also { println("E transform(expr KFunctionApp $expr ====> $it" ) }
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

    // If base array is uninterpreted, try to replace it with model value
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

            @Suppress("USELESS_CAST") // Exhaustive when
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

                is KArray3Sort<*, *, *, *> -> {
                    val (v0, v1, v2) = interpretation.vars
                    ctx.mkArrayLambda(v0, v1, v2, evaluated).uncheckedCast()
                }

                is KArrayNSort<*> -> ctx.mkArrayNLambda(
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

    private fun isValueInModel(expr: KExpr<*>): Boolean = expr is KInterpretedValue<*>

    @Suppress("USELESS_CAST") // Exhaustive when
    private fun <A : KArraySortBase<R>, R : KSort> evalArrayInterpretation(
        sort: A,
        interpretation: KModel.KFuncInterp<R>
    ): KExpr<A> = when (sort as KArraySortBase<R>) {
        is KArraySort<*, R> -> sort.evalArrayInterpretation(
            interpretation
        ) { array: KExpr<KArraySort<KSort, R>>, args, value ->
            mkArrayStore(array, args.single(), value)
        }

        is KArray2Sort<*, *, *> -> sort.evalArrayInterpretation(
            interpretation
        ) { array: KExpr<KArray2Sort<KSort, KSort, R>>, (idx0, idx1), value ->
            mkArrayStore(array, idx0, idx1, value)
        }

        is KArray3Sort<*, *, *, *> -> sort.evalArrayInterpretation(
            interpretation
        ) { array: KExpr<KArray3Sort<KSort, KSort, KSort, R>>, (idx0, idx1, idx2), value ->
            mkArrayStore(array, idx0, idx1, idx2, value)
        }

        is KArrayNSort<*> -> sort.evalArrayInterpretation(
            interpretation
        ) { array: KExpr<KArrayNSort<R>>, args, value ->
            mkArrayNStore(array, args, value)
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

            // isComplete = true, return and cache
            if (interpretation == null) {
                return@getOrPut completeModelValue(decl.sort)
            }

            check(args.size == interpretation.vars.size) {
                "${interpretation.vars.size} arguments expected but ${args.size} provided"
            }

            val resolvedInterpretation = resolvedFunctionInterpretations.getOrPut(interpretation) {
                resolveFunctionInterpretation(interpretation)
            }

            ctx.applyResolvedInterpretation(resolvedInterpretation, args)
        }
        return evaluated.asExpr(decl.sort)
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

    private fun <T : KSort> resolveFunctionInterpretation(
        interpretation: KModel.KFuncInterp<T>
    ): ResolvedFunctionInterpretation<T> {
        var resolvedEntry: ResolvedFunctionEntry<T> = ResolvedFunctionDefaultEntry(interpretation.default)

        for (entry in interpretation.entries.asReversed()) {
            val isValueEntry = entry.args.all { isValueInModel(it) }

            resolvedEntry = if (isValueEntry) {
                resolvedEntry.addValueEntry(entry.args, entry.value)
            } else {
                resolvedEntry.addUninterpretedEntry(entry.args, entry.value)
            }
        }

        return ResolvedFunctionInterpretation(interpretation, resolvedEntry)
    }

    private fun <T : KSort> KContext.applyResolvedInterpretation(
        interpretation: ResolvedFunctionInterpretation<T>,
        args: List<KExpr<*>>
    ): KExpr<T> {
        val argsAreConstants = args.all { isValueInModel(it) }

        // Replace function parameters vars with actual arguments
        val varSubstitution = createVariableSubstitution(interpretation, args)

        val resultEntries = mutableListOf<Pair<List<KExpr<*>>, KExpr<T>>>()

        var currentEntries = interpretation.rootEntry
        while (true) {
            when (currentEntries) {
                is ResolvedFunctionUninterpretedEntry -> {
                    currentEntries.tryResolveArgs(
                        varSubstitution, args, resultEntries
                    )?.let { return it }

                    currentEntries = currentEntries.next
                }

                is ResolvedFunctionValuesEntry -> {
                    currentEntries.tryResolveArgs(
                        varSubstitution, args, argsAreConstants, resultEntries
                    )?.let { return it }

                    currentEntries = currentEntries.next
                }

                is ResolvedFunctionDefaultEntry -> return currentEntries.resolveArgs(
                    interpretation.interpretation.sort,
                    varSubstitution, args, resultEntries
                )
            }
        }
    }

    private fun KContext.createVariableSubstitution(
        interpretation: ResolvedFunctionInterpretation<*>,
        args: List<KExpr<*>>
    ) = KExprSubstitutor(this).apply {
        interpretation.interpretation.vars.zip(args).forEach { (v, a) ->
            val app: KExpr<KSort> = mkConstApp(v).uncheckedCast()
            substitute(app, a.uncheckedCast())
        }
    }

    private fun <T : KSort> rewriteFunctionAppAsIte(
        base: KExpr<T>,
        args: List<KExpr<*>>,
        entries: List<Pair<List<KExpr<*>>, KExpr<T>>>
    ): KExpr<T> = with(ctx) {
        entries.foldRight(base) { entry, acc ->
            val argBinding = entry.first.zip(args) { ea, a ->
                val entryArg: KExpr<KSort> = ea.uncheckedCast()
                val actualArg: KExpr<KSort> = a.uncheckedCast()
                mkEq(entryArg, actualArg)
            }
            mkIte(mkAnd(argBinding), entry.second, acc)
        }
    }

    private fun <T : KSort> ResolvedFunctionDefaultEntry<T>.resolveArgs(
        sort: T,
        varSubstitution: KExprSubstitutor,
        args: List<KExpr<*>>,
        resultEntries: List<Pair<List<KExpr<*>>, KExpr<T>>>
    ): KExpr<T> {
        val resolvedDefault = expr?.let { varSubstitution.apply(it) }

        // in case of partial interpretation we can generate any default expr to preserve expression correctness
        val defaultExpr = resolvedDefault ?: completeModelValue(sort)

        return rewriteFunctionAppAsIte(defaultExpr, args, resultEntries)
    }

    private fun <T : KSort> ResolvedFunctionValuesEntry<T>.tryResolveArgs(
        varSubstitution: KExprSubstitutor,
        args: List<KExpr<*>>,
        argsAreConstants: Boolean,
        resultEntries: MutableList<Pair<List<KExpr<*>>, KExpr<T>>>
    ): KExpr<T>? {
        if (argsAreConstants) {
            val entryValue = entries[args]?.let { varSubstitution.apply(it) }
            if (entryValue != null) {
                // We have no possibly matching entries and we found a matched entry
                if (resultEntries.isEmpty()) return entryValue

                // We don't need to process next entries but we need to handle parent entries
                return rewriteFunctionAppAsIte(entryValue, args, resultEntries)
            }
            return null
        } else {
            // Args are not values, entry args are values -> args are definitely not in current entry
            for ((entryArgs, entryValue) in entries) {
                addEntryIfArgsAreNotDistinct(resultEntries, args, entryArgs) {
                    varSubstitution.apply(entryValue)
                }
            }
            return null
        }
    }

    private fun <T : KSort> ResolvedFunctionUninterpretedEntry<T>.tryResolveArgs(
        varSubstitution: KExprSubstitutor,
        args: List<KExpr<*>>,
        resultEntries: MutableList<Pair<List<KExpr<*>>, KExpr<T>>>
    ): KExpr<T>? {
        for (entry in reversedEntries.asReversed()) {
            val entryArgs = entry.first.map { varSubstitution.apply(it) }
            val entryValue = varSubstitution.apply(entry.second)

            if (entryArgs == args) {
                // We have no possibly matching entries and we found a matched entry
                if (resultEntries.isEmpty()) return entryValue

                // We don't need to process next entries but we need to handle parent entries
                return rewriteFunctionAppAsIte(entryValue, args, resultEntries)
            }

            addEntryIfArgsAreNotDistinct(resultEntries, args, entryArgs) { entryValue }
        }
        return null
    }

    private inline fun <T : KSort> addEntryIfArgsAreNotDistinct(
        entries: MutableList<Pair<List<KExpr<*>>, KExpr<T>>>,
        args: List<KExpr<*>>,
        entryArgs: List<KExpr<*>>,
        entryValue: () -> KExpr<T>
    ) {
        if (areDefinitelyDistinct(args, entryArgs)) return

        val value = entryValue()
        entries.add(entryArgs to value)
    }

    private class ResolvedFunctionInterpretation<T : KSort>(
        val interpretation: KModel.KFuncInterp<T>,
        val rootEntry: ResolvedFunctionEntry<T>
    )

    private sealed interface ResolvedFunctionEntry<T : KSort> {
        fun addUninterpretedEntry(args: List<KExpr<*>>, value: KExpr<T>): ResolvedFunctionEntry<T> =
            ResolvedFunctionUninterpretedEntry(arrayListOf(args to value), this)

        fun addValueEntry(args: List<KExpr<*>>, value: KExpr<T>): ResolvedFunctionEntry<T> =
            ResolvedFunctionValuesEntry(hashMapOf(args to value), this)
    }

    private class ResolvedFunctionUninterpretedEntry<T : KSort>(
        val reversedEntries: MutableList<Pair<List<KExpr<*>>, KExpr<T>>>,
        val next: ResolvedFunctionEntry<T>
    ) : ResolvedFunctionEntry<T>{
        override fun addUninterpretedEntry(args: List<KExpr<*>>, value: KExpr<T>): ResolvedFunctionEntry<T> {
            reversedEntries.add(args to value)
            return this
        }
    }

    private class ResolvedFunctionValuesEntry<T : KSort>(
        val entries: MutableMap<List<KExpr<*>>, KExpr<T>>,
        val next: ResolvedFunctionEntry<T>
    ) : ResolvedFunctionEntry<T> {
        override fun addValueEntry(args: List<KExpr<*>>, value: KExpr<T>): ResolvedFunctionEntry<T> {
            entries[args] = value
            return this
        }
    }

    private class ResolvedFunctionDefaultEntry<T : KSort>(
        val expr: KExpr<T>?
    ) : ResolvedFunctionEntry<T>
}
