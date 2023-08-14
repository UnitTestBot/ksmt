package io.ksmt.solver.yices

import com.sri.yices.Yices
import io.ksmt.KAst
import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.solver.KForkingSolver
import io.ksmt.solver.KForkingSolverManager
import io.ksmt.solver.util.KExprIntInternalizerBase
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap
import it.unimi.dsi.fastutil.ints.IntOpenHashSet
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap
import java.util.Collections.newSetFromMap
import java.util.Collections.synchronizedSet
import java.util.IdentityHashMap
import java.util.concurrent.atomic.AtomicInteger

class KYicesForkingSolverManager(
    private val ctx: KContext
) : KForkingSolverManager<KYicesSolverConfiguration> {

    private val solvers = synchronizedSet(newSetFromMap(IdentityHashMap<KYicesForkingSolver, _>()))
    private val sharedCacheReferences = IdentityHashMap<KYicesForkingSolver, AtomicInteger>()

    private val expressionsCache = IdentityHashMap<KYicesForkingSolver, ExpressionsCache>()
    private val expressionsReversedCache = IdentityHashMap<KYicesForkingSolver, ExpressionsReversedCache>()
    private val sortsCache = IdentityHashMap<KYicesForkingSolver, SortsCache>()
    private val sortsReversedCache = IdentityHashMap<KYicesForkingSolver, SortsReversedCache>()
    private val declsCache = IdentityHashMap<KYicesForkingSolver, DeclsCache>()
    private val declsReversedCache = IdentityHashMap<KYicesForkingSolver, DeclsReversedCache>()
    private val varsCache = IdentityHashMap<KYicesForkingSolver, VarsCache>()
    private val varsReversedCache = IdentityHashMap<KYicesForkingSolver, VarsReversedCache>()
    private val typesCache = IdentityHashMap<KYicesForkingSolver, TypesCache>()
    private val termsCache = IdentityHashMap<KYicesForkingSolver, TermsCache>()
    private val maxUninterpretedSortValueIndex = IdentityHashMap<KYicesForkingSolver, AtomicInteger>()

    private val scopedExpressions = IdentityHashMap<KYicesForkingSolver, ScopedExpressions>()
    private val scopedUninterpretedValues = IdentityHashMap<KYicesForkingSolver, ScopedUninterpretedSortValues>()
    private val expressionLevels = IdentityHashMap<KYicesForkingSolver, ExpressionLevels>()

    internal fun findExpressionsCache(s: KYicesForkingSolver): ExpressionsCache = expressionsCache.getValue(s)
    internal fun findExpressionsReversedCache(s: KYicesForkingSolver): ExpressionsReversedCache =
        expressionsReversedCache.getValue(s)

    internal fun findSortsCache(s: KYicesForkingSolver): SortsCache = sortsCache.getValue(s)
    internal fun findSortsReversedCache(s: KYicesForkingSolver): SortsReversedCache = sortsReversedCache.getValue(s)
    internal fun findDeclsCache(s: KYicesForkingSolver): DeclsCache = declsCache.getValue(s)
    internal fun findDeclsReversedCache(s: KYicesForkingSolver): DeclsReversedCache = declsReversedCache.getValue(s)
    internal fun findVarsCache(s: KYicesForkingSolver): VarsCache = varsCache.getValue(s)
    internal fun findVarsReversedCache(s: KYicesForkingSolver): VarsReversedCache = varsReversedCache.getValue(s)
    internal fun findTypesCache(s: KYicesForkingSolver): TypesCache = typesCache.getValue(s)
    internal fun findTermsCache(s: KYicesForkingSolver): TermsCache = termsCache.getValue(s)
    internal fun findMaxUninterpretedSortValueIdx(s: KYicesForkingSolver) = maxUninterpretedSortValueIndex.getValue(s)

    override fun mkForkingSolver(): KForkingSolver<KYicesSolverConfiguration> =
        KYicesForkingSolver(ctx, this, null).also {
            solvers += it
            sharedCacheReferences[it] = AtomicInteger(1)
            expressionsCache[it] = ExpressionsCache().withNotInternalizedDefaultValue()
            expressionsReversedCache[it] = ExpressionsReversedCache()
            sortsCache[it] = SortsCache().withNotInternalizedDefaultValue()
            sortsReversedCache[it] = SortsReversedCache()
            declsCache[it] = DeclsCache().withNotInternalizedDefaultValue()
            declsReversedCache[it] = DeclsReversedCache()
            varsCache[it] = VarsCache().withNotInternalizedDefaultValue()
            varsReversedCache[it] = VarsReversedCache()
            typesCache[it] = TypesCache()
            termsCache[it] = TermsCache()
            maxUninterpretedSortValueIndex[it] = AtomicInteger(0)
            scopedExpressions[it] = ScopedExpressions(::HashSet, ::HashSet)
            scopedUninterpretedValues[it] = ScopedUninterpretedSortValues(::HashMap, ::HashMap)
            expressionLevels[it] = ExpressionLevels()
        }

    internal fun mkForkingSolver(parent: KYicesForkingSolver) = KYicesForkingSolver(ctx, this, parent).also {
        solvers += it
        sharedCacheReferences[it] = sharedCacheReferences.getValue(parent).apply { incrementAndGet() }
        expressionsCache[it] = expressionsCache[parent]
        expressionsReversedCache[it] = expressionsReversedCache[parent]
        sortsCache[it] = sortsCache[parent]
        sortsReversedCache[it] = sortsReversedCache[parent]
        declsCache[it] = declsCache[parent]
        declsReversedCache[it] = declsReversedCache[parent]
        varsCache[it] = varsCache[parent]
        varsReversedCache[it] = varsReversedCache[parent]
        typesCache[it] = typesCache[parent]
        termsCache[it] = termsCache[parent]
        scopedExpressions[it] = ScopedExpressions(::HashSet, ::HashSet)
            .apply { fork(scopedExpressions.getValue(parent)) }
        scopedUninterpretedValues[it] = ScopedUninterpretedSortValues(::HashMap, ::HashMap)
            .apply { fork(scopedUninterpretedValues.getValue(parent)) }
        expressionLevels[it] = ExpressionLevels(expressionLevels.getValue(parent))

        val parentMaxUninterpretedSortValueIdx = maxUninterpretedSortValueIndex.getValue(parent).get()
        maxUninterpretedSortValueIndex[it] = AtomicInteger(parentMaxUninterpretedSortValueIdx)
    }

    internal fun createUninterpretedValuesTracker(solver: KYicesForkingSolver) = UninterpretedValuesTracker(
        ctx,
        scopedExpressions.getValue(solver),
        scopedUninterpretedValues.getValue(solver),
        expressionLevels.getValue(solver)
    )

    /**
     * Unregisters [solver] for this manager
     */
    internal fun close(solver: KYicesForkingSolver) {
        solvers -= solver
        decRef(solver)
    }

    override fun close() {
        solvers.forEach(KYicesForkingSolver::close)
    }

    private fun decRef(solver: KYicesForkingSolver) {
        val referencesAfterDec = sharedCacheReferences.getValue(solver).decrementAndGet()
        if (referencesAfterDec == 0) {
            sharedCacheReferences -= solver
            expressionsCache -= solver
            expressionsReversedCache -= solver
            sortsCache -= solver
            sortsReversedCache -= solver
            declsCache -= solver
            declsReversedCache -= solver
            varsCache -= solver
            varsReversedCache -= solver
            typesCache.remove(solver)?.forEach(Yices::yicesDecrefType)
            termsCache.remove(solver)?.forEach(Yices::yicesDecrefTerm)
            maxUninterpretedSortValueIndex -= solver
            scopedExpressions -= solver
            scopedUninterpretedValues -= solver
            expressionLevels -= solver
        }
    }

    private fun <T : KAst> Object2IntOpenHashMap<T>.withNotInternalizedDefaultValue() = apply {
        defaultReturnValue(KExprIntInternalizerBase.NOT_INTERNALIZED)
    }
}

private typealias ExpressionsCache = Object2IntOpenHashMap<KExpr<*>>
private typealias ExpressionsReversedCache = Int2ObjectOpenHashMap<KExpr<*>>
private typealias SortsCache = Object2IntOpenHashMap<KSort>
private typealias SortsReversedCache = Int2ObjectOpenHashMap<KSort>
private typealias DeclsCache = Object2IntOpenHashMap<KDecl<*>>
private typealias DeclsReversedCache = Int2ObjectOpenHashMap<KDecl<*>>
private typealias VarsCache = Object2IntOpenHashMap<KDecl<*>>
private typealias VarsReversedCache = Int2ObjectOpenHashMap<KDecl<*>>
private typealias TypesCache = IntOpenHashSet
private typealias TermsCache = IntOpenHashSet
private typealias ScopedExpressions = ScopedLinkedFrame<HashSet<KExpr<*>>>
@Suppress("MaxLineLength")
private typealias ScopedUninterpretedSortValues = ScopedLinkedFrame<HashMap<KUninterpretedSort, HashSet<KUninterpretedSortValue>>>
private typealias ExpressionLevels = Object2IntOpenHashMap<KExpr<*>>
