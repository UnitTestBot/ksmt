package io.ksmt.solver.model

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KArray2Lambda
import io.ksmt.expr.KArray3Lambda
import io.ksmt.expr.KArrayLambda
import io.ksmt.expr.KArrayNLambda
import io.ksmt.expr.KConst
import io.ksmt.expr.KExistentialQuantifier
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFunctionApp
import io.ksmt.expr.KFunctionAsArray
import io.ksmt.expr.KInterpretedValue
import io.ksmt.expr.KUniversalQuantifier
import io.ksmt.expr.rewrite.KExprSubstitutor
import io.ksmt.expr.rewrite.simplify.KExprSimplifier
import io.ksmt.expr.rewrite.simplify.areDefinitelyDistinct
import io.ksmt.expr.rewrite.simplify.simplifyExpr
import io.ksmt.solver.KModel
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import io.ksmt.utils.asExpr
import io.ksmt.utils.sampleValue
import io.ksmt.utils.uncheckedCast
import kotlin.math.absoluteValue

open class KModelEvaluator(
    ctx: KContext,
    private val model: KModel,
    private val isComplete: Boolean,
    private val quantifiedVars: Set<KDecl<*>> = emptySet()
) : KExprSimplifier(ctx) {
    private val evaluatedFunctionApp = hashMapOf<Pair<KDecl<*>, List<KExpr<*>>>, KExpr<*>>()
    private val evaluatedFunctionArray = hashMapOf<KDecl<*>, KExpr<*>>()
    private val resolvedFunctionInterpretations = hashMapOf<KDecl<*>, ResolvedFunctionInterpretation<*>>()

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

            if (interpretation is KFuncInterpVarsFree) {
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

    private fun <A : KArraySortBase<R>, R : KSort> evalArrayInterpretation(
        sort: A,
        interpretation: KFuncInterpVarsFree<R>
    ): KExpr<A> = evalRawArrayInterpretation(sort, interpretation).uncheckedCast()

    private fun <R : KSort> evalRawArrayInterpretation(
        rawSort: KArraySortBase<R>,
        interpretation: KFuncInterpVarsFree<R>
    ): KExpr<*> = when (rawSort) {
        is KArraySort<*, R> -> {
            val sort: KArraySort<KSort, R> = rawSort.uncheckedCast()
            sort.evalArrayInterpretation<_, R, KFuncInterpEntryOneAry<R>>(
                interpretation
            ) { array: KExpr<KArraySort<KSort, R>>, entry ->
                mkArrayStore(array, entry.arg.uncheckedCast(), entry.value)
            }
        }

        is KArray2Sort<*, *, *> -> {
            val sort: KArray2Sort<KSort, KSort, R> = rawSort.uncheckedCast()
            sort.evalArrayInterpretation<_, R, KFuncInterpEntryTwoAry<R>>(
                interpretation
            ) { array: KExpr<KArray2Sort<KSort, KSort, R>>, entry ->
                mkArrayStore(array, entry.arg0.uncheckedCast(), entry.arg1.uncheckedCast(), entry.value)
            }
        }

        is KArray3Sort<*, *, *, *> -> {
            val sort: KArray3Sort<KSort, KSort, KSort, R> = rawSort.uncheckedCast()
            sort.evalArrayInterpretation<_, R, KFuncInterpEntryThreeAry<R>>(
                interpretation
            ) { array: KExpr<KArray3Sort<KSort, KSort, KSort, R>>, entry ->
                mkArrayStore(
                    array,
                    entry.arg0.uncheckedCast(),
                    entry.arg1.uncheckedCast(),
                    entry.arg2.uncheckedCast(),
                    entry.value
                )
            }
        }

        is KArrayNSort<*> -> {
            val sort: KArrayNSort<R> = rawSort.uncheckedCast()
            sort.evalArrayInterpretation<_, R, KFuncInterpEntryNAry<R>>(
                interpretation
            ) { array: KExpr<KArrayNSort<R>>, entry ->
                mkArrayNStore(array, entry.args, entry.value)
            }
        }
    }

    private inline fun <A : KArraySortBase<R>, R : KSort, reified E : KFuncInterpEntry<R>> A.evalArrayInterpretation(
        interpretation: KFuncInterpVarsFree<R>,
        mkEntryStore: KContext.(KExpr<A>, E) -> KExpr<A>
    ): KExpr<A> = with(ctx) {
        val defaultValue = interpretation.default ?: completeModelValue(range)
        val defaultArray: KExpr<A> = mkArrayConst(this@evalArrayInterpretation, defaultValue)

        interpretation.entries.foldRight(defaultArray) { entry, acc ->
            mkEntryStore(acc, entry as E)
        }
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

            check(args.size == decl.argSorts.size) {
                "${decl.argSorts.size} arguments expected but ${args.size} provided"
            }

            if (interpretation.entries.isEmpty()) {
                return@getOrPut interpretation.resolveDefaultValue(args)
            }

            val resolvedInterpretation = resolveFunctionInterpretation(interpretation)
            resolvedInterpretation.apply(args)
        }
        return evaluated.asExpr(decl.sort)
    }

    private fun <T : KSort> KFuncInterp<T>.resolveDefaultValue(args: List<KExpr<*>>): KExpr<T> = when (this) {
        is KFuncInterpVarsFree<T> -> default
        is KFuncInterpWithVars -> default?.let { ctx.createVariableSubstitution(vars, args).apply(it) }
    } ?: completeModelValue(sort)

    private fun <T : KSort> completeModelValue(sort: T): KExpr<T> {
        val value = when (sort) {
            is KUninterpretedSort ->
                model.uninterpretedSortUniverse(sort)
                    // Prefer values, closest to the zero
                    ?.minByOrNull { it.valueIdx.absoluteValue }
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
        interpretation: KFuncInterp<T>
    ): ResolvedFunctionInterpretation<T> = resolvedFunctionInterpretations.getOrPut(interpretation.decl) {
        when (interpretation) {
            is KFuncInterpVarsFree -> resolveFunctionInterpretation(interpretation, hasVars = false)
            is KFuncInterpWithVars -> resolveFunctionInterpretation(interpretation, hasVars = true)
        }
    }.uncheckedCast()


    private fun <T : KSort> resolveFunctionInterpretation(
        interpretation: KFuncInterp<T>,
        hasVars: Boolean
    ): ResolvedFunctionInterpretation<T> {
        var resolvedEntry: ResolvedFunctionEntry<T> = ResolvedFunctionDefaultEntry(interpretation.default)

        for (entry in interpretation.entries.asReversed()) {
            val isValueEntry = entry.isValueEntry()

            resolvedEntry = if (isValueEntry) {
                resolvedEntry.addValueEntry(entry)
            } else {
                resolvedEntry.addUninterpretedEntry(entry)
            }
        }

        return ResolvedFunctionInterpretation(interpretation, resolvedEntry, hasVars)
    }

    private fun KContext.createVariableSubstitution(
        vars: List<KDecl<*>>,
        args: List<KExpr<*>>
    ) = KExprSubstitutor(this).apply {
        vars.zip(args).forEach { (v, a) ->
            val app: KExpr<KSort> = mkConstApp(v).uncheckedCast()
            substitute(app, a.uncheckedCast())
        }
    }

    private fun <T : KSort> rewriteFunctionAppAsIte(
        base: KExpr<T>,
        args: List<KExpr<*>>,
        entries: List<KFuncInterpEntry<T>>
    ): KExpr<T> = with(ctx) {
        entries.foldRight(base) { entry, acc ->
            val argBinding = entry.args.zip(args) { ea, a ->
                val entryArg: KExpr<KSort> = ea.uncheckedCast()
                val actualArg: KExpr<KSort> = a.uncheckedCast()
                mkEq(entryArg, actualArg)
            }
            mkIte(mkAnd(argBinding), entry.value, acc)
        }
    }

    private inner class FunctionAppResolutionCtx<T : KSort>(
        val interpretation: KFuncInterp<T>,
        val rootEntry: ResolvedFunctionEntry<T>,
        val args: List<KExpr<*>>,
        val hasVars: Boolean
    ) {
        private val argsAreConstants = args.all { isValueInModel(it) }

        private val varSubstitution by lazy {
            ctx.createVariableSubstitution(interpretation.vars, args)
        }

        private val resultEntries = mutableListOf<KFuncInterpEntry<T>>()

        fun resolve(): KExpr<T> {
            var currentEntries = rootEntry
            while (true) {
                currentEntries = when (currentEntries) {
                    is ResolvedFunctionUninterpretedEntry -> {
                        currentEntries.tryResolveArgs()?.let { return it }
                        currentEntries.next
                    }

                    is ResolvedFunctionValuesEntry -> {
                        currentEntries.tryResolveArgs()?.let { return it }
                        currentEntries.next
                    }

                    is ResolvedFunctionDefaultEntry -> return currentEntries.resolveArgs()
                }
            }
        }

        private fun <T : KSort> KExpr<T>.substituteVars(): KExpr<T> =
            if (hasVars) varSubstitution.apply(this) else this

        private fun ResolvedFunctionDefaultEntry<T>.resolveArgs(): KExpr<T> {
            val resolvedDefault = expr?.substituteVars()

            // in case of partial interpretation we can generate any default expr to preserve expression correctness
            val defaultExpr = resolvedDefault ?: completeModelValue(interpretation.sort)

            return rewriteFunctionAppAsIte(defaultExpr, args, resultEntries)
        }

        private fun ResolvedFunctionValuesEntry<T>.tryResolveArgs(): KExpr<T>? {
            if (argsAreConstants) {
                val entryValue = findValueEntry(args)?.substituteVars()?.value
                if (entryValue != null) {
                    // We have no possibly matching entries and we found a matched entry
                    if (resultEntries.isEmpty()) return entryValue

                    // We don't need to process next entries but we need to handle parent entries
                    return rewriteFunctionAppAsIte(entryValue, args, resultEntries)
                }
                return null
            } else {
                // Args are not values, entry args are values -> args are definitely not in current entry
                for (entry in entries) {
                    addEntryIfArgsAreNotDistinct(entry) { entry.substituteVars() }
                }
                return null
            }
        }

        private fun ResolvedFunctionUninterpretedEntry<T>.tryResolveArgs(): KExpr<T>? {
            for (entry in entries) {
                val resolvedEntry = entry.substituteVars()

                if (resolvedEntry.argsAreEqual(args)) {
                    // We have no possibly matching entries and we found a matched entry
                    if (resultEntries.isEmpty()) return resolvedEntry.value

                    // We don't need to process next entries but we need to handle parent entries
                    return rewriteFunctionAppAsIte(resolvedEntry.value, args, resultEntries)
                }

                addEntryIfArgsAreNotDistinct(resolvedEntry) { resolvedEntry }
            }
            return null
        }

        private inline fun addEntryIfArgsAreNotDistinct(
            entry: KFuncInterpEntry<T>,
            resolveEntry: () -> KFuncInterpEntry<T>
        ) {
            if (entry.argsAreDistinct(args)) return
            val resolvedEntry = resolveEntry()
            resultEntries.add(resolvedEntry)
        }

        private fun KFuncInterpEntry<T>.substituteVars(): KFuncInterpEntry<T> {
            if (this is KFuncInterpEntryVarsFree) return this
            return when (this) {
                is KFuncInterpEntryOneAry<T> -> modify(
                    arg.substituteVars(),
                    value.substituteVars()
                )

                is KFuncInterpEntryTwoAry<T> -> modify(
                    arg0.substituteVars(),
                    arg1.substituteVars(),
                    value.substituteVars()
                )

                is KFuncInterpEntryThreeAry<T> -> modify(
                    arg0.substituteVars(),
                    arg1.substituteVars(),
                    arg2.substituteVars(),
                    value.substituteVars()
                )

                is KFuncInterpEntryNAry<T> -> modify(
                    args.map { it.substituteVars() },
                    value.substituteVars()
                )
            }
        }

        private fun KFuncInterpEntry<T>.argsAreEqual(args: List<KExpr<*>>): Boolean = when (this) {
            is KFuncInterpEntryOneAry<*> -> arg == args.single()
            is KFuncInterpEntryTwoAry<*> -> arg0 == args.first() && arg1 == args.last()
            is KFuncInterpEntryThreeAry<*> -> {
                val (a0, a1, a2) = args
                arg0 == a0 && arg1 == a1 && arg2 == a2
            }
            is KFuncInterpEntryNAry<*> -> this.args == args
        }

        private fun KFuncInterpEntry<T>.argsAreDistinct(args: List<KExpr<*>>): Boolean = when (this) {
            is KFuncInterpEntryOneAry<*> ->
                areDistinct(arg, args.single())

            is KFuncInterpEntryTwoAry<*> -> {
                val (a0, a1) = args
                areDistinct(arg0, a0) || areDistinct(arg1, a1)
            }

            is KFuncInterpEntryThreeAry<*> -> {
                val (a0, a1, a2) = args
                areDistinct(arg0, a0) || areDistinct(arg1, a1) || areDistinct(arg2, a2)
            }

            is KFuncInterpEntryNAry<*> -> areDefinitelyDistinct(this.args, args)
        }

        private fun areDistinct(lhs: KExpr<*>, rhs: KExpr<*>): Boolean =
            areDefinitelyDistinct(lhs.uncheckedCast<_, KExpr<KSort>>(), rhs.uncheckedCast())
    }

    private inner class ResolvedFunctionInterpretation<T : KSort>(
        val interpretation: KFuncInterp<T>,
        val rootEntry: ResolvedFunctionEntry<T>,
        val hasVars: Boolean
    ) {
        fun apply(args: List<KExpr<*>>): KExpr<T> =
            FunctionAppResolutionCtx(interpretation, rootEntry, args, hasVars).resolve()
    }

    private sealed interface ResolvedFunctionEntry<T : KSort> {
        val entries: Iterable<KFuncInterpEntry<T>>

        fun addUninterpretedEntry(entry: KFuncInterpEntry<T>): ResolvedFunctionEntry<T> =
            ResolvedFunctionUninterpretedEntry(arrayListOf(), this).addUninterpretedEntry(entry)

        fun addValueEntry(entry: KFuncInterpEntry<T>): ResolvedFunctionEntry<T> =
            ResolvedFunctionValuesEntry(entry.arity, hashMapOf(), this).addValueEntry(entry)
    }

    private class ResolvedFunctionUninterpretedEntry<T : KSort>(
        private val reversedEntries: MutableList<KFuncInterpEntry<T>>,
        val next: ResolvedFunctionEntry<T>
    ) : ResolvedFunctionEntry<T> {
        override val entries: Iterable<KFuncInterpEntry<T>>
            get() = reversedEntries.asReversed()

        override fun addUninterpretedEntry(entry: KFuncInterpEntry<T>): ResolvedFunctionEntry<T> {
            reversedEntries.add(entry)
            return this
        }
    }

    private class ResolvedFunctionValuesEntry<T : KSort>(
        private val arity: Int,
        private val entriesMap: MutableMap<Any, KFuncInterpEntry<T>>,
        val next: ResolvedFunctionEntry<T>
    ) : ResolvedFunctionEntry<T> {
        override val entries: Iterable<KFuncInterpEntry<T>>
            get() = entriesMap.values

        override fun addValueEntry(entry: KFuncInterpEntry<T>): ResolvedFunctionEntry<T> {
            check(entry.arity == arity) { "Incorrect model: entry arity mismatch" }
            entriesMap[entry.argsSearchKey()] = entry
            return this
        }

        fun findValueEntry(args: List<KExpr<*>>): KFuncInterpEntry<T>? {
            check(args.size == arity) { "Incorrect model: args arity mismatch" }
            return entriesMap[argsSearchKey(args)]
        }

        private fun KFuncInterpEntry<*>.argsSearchKey(): Any = when (this) {
            is KFuncInterpEntryOneAry<*> -> arg
            is KFuncInterpEntryTwoAry<*> -> Pair(arg0, arg1)
            is KFuncInterpEntryThreeAry<*> -> Triple(arg0, arg1, arg2)
            is KFuncInterpEntryNAry<*> -> args
        }

        private fun argsSearchKey(args: List<KExpr<*>>): Any = when (arity) {
            KFuncInterpEntryOneAry.ARITY -> args.single()
            KFuncInterpEntryTwoAry.ARITY -> Pair(args.first(), args.last())
            KFuncInterpEntryThreeAry.ARITY -> {
                val (a0, a1, a2) = args
                Triple(a0, a1, a2)
            }
            else -> args
        }
    }

    private class ResolvedFunctionDefaultEntry<T : KSort>(
        val expr: KExpr<T>?
    ) : ResolvedFunctionEntry<T> {
        override val entries: Iterable<KFuncInterpEntry<T>>
            get() = emptyList()
    }

    private fun KFuncInterpEntry<*>.isValueEntry(): Boolean = when (this) {
        is KFuncInterpEntryOneAry<*> -> isValueInModel(arg)
        is KFuncInterpEntryTwoAry<*> -> isValueInModel(arg0) && isValueInModel(arg1)
        is KFuncInterpEntryThreeAry<*> -> isValueInModel(arg0) && isValueInModel(arg1) && isValueInModel(arg2)
        is KFuncInterpEntryNAry<*> -> args.all { isValueInModel(it) }
    }
}
