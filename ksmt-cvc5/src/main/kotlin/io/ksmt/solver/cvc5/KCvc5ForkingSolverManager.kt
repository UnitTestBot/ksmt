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
import java.util.IdentityHashMap
import java.util.TreeMap
import java.util.concurrent.ConcurrentHashMap

open class KCvc5ForkingSolverManager(private val ctx: KContext) : KForkingSolverManager<KCvc5SolverConfiguration> {

    private val solvers: MutableSet<KCvc5ForkingSolver> = ConcurrentHashMap.newKeySet()

    /**
     * for each parent-to-child hierarchy created only one mkExprSolver,
     * which is responsible for native expressions lifetime
     */
    private val forkingSolverToMkExprSolver = IdentityHashMap<KCvc5ForkingSolver, Solver>()
    private val mkExprSolverReferences = IdentityHashMap<Solver, Int>()

    // shared cache
    private val expressionsCache = IdentityHashMap<Solver, ExpressionsCache>()
    private val expressionsReversedCache = IdentityHashMap<Solver, ExpressionsReversedCache>()
    private val sortsCache = IdentityHashMap<Solver, SortsCache>()
    private val sortsReversedCache = IdentityHashMap<Solver, SortsReversedCache>()
    private val declsCache = IdentityHashMap<Solver, DeclsCache>()
    private val declsReversedCache = IdentityHashMap<Solver, DeclsReversedCache>()

    private val uninterpretedSortValueDescriptors = IdentityHashMap<Solver, UninterpretedSortValueDescriptors>()
    private val uninterpretedSortValueInterpretersCache =
        IdentityHashMap<Solver, UninterpretedSortValueInterpretersCache>()
    private val uninterpretedSortValues = IdentityHashMap<Solver, UninterpretedSortValues>()

    private fun Solver.ensureRegisteredAsMkExprSolver() = require(this in mkExprSolverReferences) {
        "Solver is not registered by this manager"
    }

    internal fun KCvc5Context.findExpressionsCache() = mkExprSolver.ensureRegisteredAsMkExprSolver().let {
        expressionsCache.getOrPut(mkExprSolver) { ExpressionsCache() }
    }

    internal fun KCvc5Context.findExpressionsReversedCache() = mkExprSolver.ensureRegisteredAsMkExprSolver().let {
        expressionsReversedCache.getOrPut(mkExprSolver) { ExpressionsReversedCache() }
    }

    internal fun KCvc5Context.findSortsCache() = mkExprSolver.ensureRegisteredAsMkExprSolver().let {
        sortsCache.getOrPut(mkExprSolver) { SortsCache() }
    }

    internal fun KCvc5Context.findSortsReversedCache() = mkExprSolver.ensureRegisteredAsMkExprSolver().let {
        sortsReversedCache.getOrPut(mkExprSolver) { SortsReversedCache() }
    }

    internal fun KCvc5Context.findDeclsCache() = mkExprSolver.ensureRegisteredAsMkExprSolver().let {
        declsCache.getOrPut(mkExprSolver) { DeclsCache() }
    }

    internal fun KCvc5Context.findDeclsReversedCache() = mkExprSolver.ensureRegisteredAsMkExprSolver().let {
        declsReversedCache.getOrPut(mkExprSolver) { DeclsReversedCache() }
    }

    internal fun KCvc5Context.findUninterpretedSortsValueDescriptors() = mkExprSolver.ensureRegisteredAsMkExprSolver()
        .let {
            uninterpretedSortValueDescriptors.getOrPut(mkExprSolver) { UninterpretedSortValueDescriptors() }
        }

    internal fun KCvc5Context.findUninterpretedSortsValueInterpretersCache() = mkExprSolver
        .ensureRegisteredAsMkExprSolver().let {
            uninterpretedSortValueInterpretersCache.getOrPut(mkExprSolver) { UninterpretedSortValueInterpretersCache() }
        }

    internal fun KCvc5Context.findUninterpretedSortValues() = mkExprSolver.ensureRegisteredAsMkExprSolver().let {
        uninterpretedSortValues.getOrPut(mkExprSolver) { UninterpretedSortValues() }
    }

    override fun mkForkingSolver(): KForkingSolver<KCvc5SolverConfiguration> {
        val mkExprSolver = Solver()
        incRef(mkExprSolver)
        return KCvc5ForkingSolver(ctx, this, mkExprSolver, null).also {
            solvers += it
            forkingSolverToMkExprSolver[it] = mkExprSolver
        }
    }

    internal fun mkForkingSolver(parent: KCvc5ForkingSolver): KForkingSolver<KCvc5SolverConfiguration> {
        val mkExprSolver = forkingSolverToMkExprSolver.getValue(parent)
        incRef(mkExprSolver)
        return KCvc5ForkingSolver(ctx, this, mkExprSolver, parent).also {
            solvers += it
            forkingSolverToMkExprSolver[it] = mkExprSolver
        }
    }

    /**
     * unregister [solver] for this manager
     */
    internal fun close(solver: KCvc5ForkingSolver) {
        solvers -= solver
        val mkExprSolver = forkingSolverToMkExprSolver.getValue(solver)
        forkingSolverToMkExprSolver -= solver
        decRef(mkExprSolver)
    }

    override fun close() {
        solvers.forEach(KCvc5ForkingSolver::close)
    }

    private fun incRef(mkExprSolver: Solver) {
        mkExprSolverReferences[mkExprSolver] = mkExprSolverReferences.getOrDefault(mkExprSolver, 0) + 1
    }

    private fun decRef(mkExprSolver: Solver) {
        val referencesAfterDec = mkExprSolverReferences.getValue(mkExprSolver) - 1
        if (referencesAfterDec == 0) {
            mkExprSolverReferences -= mkExprSolver
            expressionsCache -= mkExprSolver
            expressionsReversedCache -= mkExprSolver
            sortsCache -= mkExprSolver
            sortsReversedCache -= mkExprSolver
            declsCache -= mkExprSolver
            declsReversedCache -= mkExprSolver
            uninterpretedSortValueDescriptors -= mkExprSolver
            uninterpretedSortValueInterpretersCache -= mkExprSolver
            uninterpretedSortValues -= mkExprSolver

            mkExprSolver.close()
        } else {
            mkExprSolverReferences[mkExprSolver] = referencesAfterDec
        }
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
private typealias UninterpretedSortValueDescriptors = ArrayList<KCvc5Context.UninterpretedSortValueDescriptor>
private typealias UninterpretedSortValueInterpretersCache = HashMap<KUninterpretedSort, Term>
@Suppress("MaxLineLength")
private typealias UninterpretedSortValues = HashMap<KUninterpretedSort, MutableList<Pair<Term, KUninterpretedSortValue>>>
