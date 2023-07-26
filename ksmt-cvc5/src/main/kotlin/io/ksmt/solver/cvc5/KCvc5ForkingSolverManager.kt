package io.ksmt.solver.cvc5

import io.github.cvc5.Solver
import io.ksmt.KContext
import io.ksmt.solver.KForkingSolver
import io.ksmt.solver.KForkingSolverManager
import java.util.IdentityHashMap
import java.util.concurrent.ConcurrentHashMap

open class KCvc5ForkingSolverManager(private val ctx: KContext) : KForkingSolverManager<KCvc5SolverConfiguration> {

    private val solvers: MutableSet<KCvc5ForkingSolver> = ConcurrentHashMap.newKeySet()

    /**
     * for each parent to child hierarchy created only one mkExprSolver,
     * which is responsible for native expressions lifetime
     */
    private val forkingSolverToMkExprSolver = IdentityHashMap<KCvc5ForkingSolver, Solver>()
    private val mkExprSolverReferences = IdentityHashMap<Solver, Int>()

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
