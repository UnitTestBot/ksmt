package io.ksmt.solver.cvc5

import io.github.cvc5.Solver
import io.github.cvc5.Sort
import io.github.cvc5.Term
import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.solver.KForkingSolver
import io.ksmt.solver.KForkingSolverManager
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import java.util.TreeMap
import java.util.concurrent.ConcurrentHashMap

/**
 * Responsible for creation and managing of [KCvc5ForkingSolver].
 *
 * It's cheaper to create multiple copies of solvers with [KCvc5ForkingSolver.fork]
 * instead of assertions transferring in [KCvc5Solver] instances manually.
 *
 * All solvers created with one manager (via both [KCvc5ForkingSolver.fork] and [mkForkingSolver])
 * use the same [mkExprContext]*, cache, and registered uninterpreted sort values.
 *
 * (*) [mkExprContext] is responsible for native expressions creation for each [KCvc5ForkingSolver]
 * in one [KCvc5ForkingSolverManager]. Therefore, life scope of native expressions is the same with
 * life scope of [KCvc5ForkingSolverManager]
 */
open class KCvc5ForkingSolverManager(private val ctx: KContext) : KForkingSolverManager<KCvc5SolverConfiguration> {
    private val mkExprContext by lazy { Solver() }
    private val solvers: MutableSet<KCvc5ForkingSolver> = ConcurrentHashMap.newKeySet()

    // shared cache
    private val expressionsCache = ExpressionsCache()
    private val expressionsReversedCache = ExpressionsReversedCache()
    private val sortsCache = SortsCache()
    private val sortsReversedCache = SortsReversedCache()
    private val declsCache = DeclsCache()
    private val declsReversedCache = DeclsReversedCache()
    private val uninterpretedSortValueInterpretersCache = UninterpretedSortValueInterpretersCache()
    private val uninterpretedSortValues = UninterpretedSortValues()

    private fun Solver.ensureMkExprContextMatches() = require(this == mkExprContext) {
        "Solver is not registered by this manager"
    }

    internal fun KCvc5Context.getExpressionsCache() = mkExprSolver.ensureMkExprContextMatches().let {
        expressionsCache
    }

    internal fun KCvc5Context.getExpressionsReversedCache() = mkExprSolver.ensureMkExprContextMatches().let {
        expressionsReversedCache
    }

    internal fun KCvc5Context.getSortsCache() = mkExprSolver.ensureMkExprContextMatches().let {
        sortsCache
    }

    internal fun KCvc5Context.getSortsReversedCache() = mkExprSolver.ensureMkExprContextMatches().let {
        sortsReversedCache
    }

    internal fun KCvc5Context.getDeclsCache() = mkExprSolver.ensureMkExprContextMatches().let {
        declsCache
    }

    internal fun KCvc5Context.getDeclsReversedCache() = mkExprSolver.ensureMkExprContextMatches().let {
        declsReversedCache
    }

    internal fun KCvc5Context.getUninterpretedSortsValueInterpretersCache() = mkExprSolver
        .ensureMkExprContextMatches().let { uninterpretedSortValueInterpretersCache }

    internal fun KCvc5Context.getUninterpretedSortValues() = mkExprSolver.ensureMkExprContextMatches().let {
        uninterpretedSortValues
    }

    override fun mkForkingSolver(): KForkingSolver<KCvc5SolverConfiguration> {
        return KCvc5ForkingSolver(ctx, this, null).also {
            solvers += it
        }
    }

    internal fun mkForkingSolver(parent: KCvc5ForkingSolver): KForkingSolver<KCvc5SolverConfiguration> {
        return KCvc5ForkingSolver(ctx, this, parent).also {
            solvers += it
        }
    }

    /**
     * unregister [solver] for this manager
     */
    internal fun close(solver: KCvc5ForkingSolver) {
        solvers -= solver
        closeContextIfStale()
    }

    override fun close() {
        solvers.forEach(KCvc5ForkingSolver::close)
    }

    internal fun createCvc5ForkingContext(solver: Solver, parent: KCvc5ForkingContext? = null) = parent
        ?.fork(solver, this)
        ?: KCvc5ForkingContext(solver, mkExprContext, ctx, this)

    private fun closeContextIfStale() {
        if (solvers.isNotEmpty()) return

        expressionsCache.clear()
        expressionsReversedCache.clear()
        sortsCache.clear()
        sortsReversedCache.clear()
        declsCache.clear()
        declsReversedCache.clear()
        uninterpretedSortValueInterpretersCache.clear()
        uninterpretedSortValues.clear()

        mkExprContext.close()
    }

    companion object {
        init {
            KCvc5SolverBase.ensureCvc5LibLoaded()
        }
    }
}

private typealias ExpressionsCache = HashMap<KExpr<*>, Term>
private typealias ExpressionsReversedCache = TreeMap<Term, KExpr<*>>
private typealias SortsCache = HashMap<KSort, Sort>
private typealias SortsReversedCache = TreeMap<Sort, KSort>
private typealias DeclsCache = HashMap<KDecl<*>, Term>
private typealias DeclsReversedCache = TreeMap<Term, KDecl<*>>
private typealias UninterpretedSortValueInterpretersCache = HashMap<KUninterpretedSort, Term>
@Suppress("MaxLineLength")
private typealias UninterpretedSortValues = HashMap<KUninterpretedSort, MutableList<Pair<Term, KUninterpretedSortValue>>>
