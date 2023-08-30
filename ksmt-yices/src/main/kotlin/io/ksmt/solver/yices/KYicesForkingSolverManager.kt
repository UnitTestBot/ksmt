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
import java.util.IdentityHashMap
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/**
 * Responsible for creation and managing of [KYicesForkingSolver].
 *
 * It's cheaper to create multiple copies of solvers with [KYicesForkingSolver.fork]
 * instead of assertions transferring in [KYicesSolver] instances manually.
 *
 * All created solvers with one manager (via both [KYicesForkingSolver.fork] and [mkForkingSolver]) use the same cache.
 */
class KYicesForkingSolverManager(
    private val ctx: KContext
) : KForkingSolverManager<KYicesSolverConfiguration> {

    private val solvers = ConcurrentHashMap.newKeySet<KYicesForkingSolver>()

    private fun ensureSolverRegistered(s: KYicesForkingSolver) = check(s in solvers) {
        "Solver is not registered by the manager."
    }

    private val expressionsCache = ExpressionsCache().withNotInternalizedDefaultValue()
    private val expressionsReversedCache = ExpressionsReversedCache()
    private val sortsCache = SortsCache().withNotInternalizedDefaultValue()
    private val sortsReversedCache = SortsReversedCache()
    private val declsCache = DeclsCache().withNotInternalizedDefaultValue()
    private val declsReversedCache = DeclsReversedCache()
    private val varsCache = VarsCache().withNotInternalizedDefaultValue()
    private val varsReversedCache = VarsReversedCache()
    private val typesCache = TypesCache()
    private val termsCache = TermsCache()
    private val maxUninterpretedSortValueIndex = IdentityHashMap<KYicesForkingSolver, AtomicInteger>()

    private val scopedExpressions = IdentityHashMap<KYicesForkingSolver, ScopedExpressions>()
    private val scopedUninterpretedValues = IdentityHashMap<KYicesForkingSolver, ScopedUninterpretedSortValues>()
    private val expressionLevels = IdentityHashMap<KYicesForkingSolver, ExpressionLevels>()

    internal fun getExpressionsCache(s: KYicesForkingSolver): ExpressionsCache = ensureSolverRegistered(s).let {
        expressionsCache
    }
    internal fun getExpressionsReversedCache(s: KYicesForkingSolver) = ensureSolverRegistered(s).let {
        expressionsReversedCache
    }
    internal fun getSortsCache(s: KYicesForkingSolver): SortsCache = ensureSolverRegistered(s).let { sortsCache }
    internal fun getSortsReversedCache(s: KYicesForkingSolver): SortsReversedCache = ensureSolverRegistered(s).let {
        sortsReversedCache
    }
    internal fun getDeclsCache(s: KYicesForkingSolver): DeclsCache = ensureSolverRegistered(s).let { declsCache }
    internal fun getDeclsReversedCache(s: KYicesForkingSolver): DeclsReversedCache = ensureSolverRegistered(s).let {
        declsReversedCache
    }
    internal fun getVarsCache(s: KYicesForkingSolver): VarsCache = ensureSolverRegistered(s).let { varsCache }
    internal fun getVarsReversedCache(s: KYicesForkingSolver): VarsReversedCache = ensureSolverRegistered(s).let {
        varsReversedCache
    }
    internal fun getTypesCache(s: KYicesForkingSolver): TypesCache = ensureSolverRegistered(s).let { typesCache }
    internal fun getTermsCache(s: KYicesForkingSolver): TermsCache = ensureSolverRegistered(s).let { termsCache }
    internal fun getMaxUninterpretedSortValueIdx(s: KYicesForkingSolver) = ensureSolverRegistered(s).let {
        maxUninterpretedSortValueIndex.getValue(s)
    }

    override fun mkForkingSolver(): KForkingSolver<KYicesSolverConfiguration> =
        KYicesForkingSolver(ctx, this, null).also {
            solvers += it
            maxUninterpretedSortValueIndex[it] = AtomicInteger(0)
            scopedExpressions[it] = ScopedExpressions(::HashSet, ::HashSet)
            scopedUninterpretedValues[it] = ScopedUninterpretedSortValues(::HashMap, ::HashMap)
            expressionLevels[it] = ExpressionLevels()
        }

    internal fun mkForkingSolver(parent: KYicesForkingSolver) = KYicesForkingSolver(ctx, this, parent).also {
        solvers += it
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
        scopedExpressions -= solver
        scopedUninterpretedValues -= solver
        maxUninterpretedSortValueIndex -= solver
        expressionLevels -= solver

        if (solvers.isEmpty()) {
            expressionsCache.clear()
            expressionsReversedCache.clear()
            sortsCache.clear()
            sortsReversedCache.clear()
            declsCache.clear()
            declsReversedCache.clear()
            varsCache.clear()
            varsReversedCache.clear()
            typesCache.forEach(Yices::yicesDecrefType)
            termsCache.forEach(Yices::yicesDecrefTerm)
            typesCache.clear()
            termsCache.clear()
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
