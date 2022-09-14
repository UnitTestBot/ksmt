package org.ksmt.solver

import org.ksmt.expr.KExpr
import org.ksmt.sort.KBoolSort
import kotlin.time.Duration

interface KSolver : AutoCloseable {

    /**
     * Assert an expression into solver.
     *
     * @see check
     * */
    fun assert(expr: KExpr<KBoolSort>)

    /**
     * Assert an expression into solver.
     *
     * @return [KBoolSort] constant which is used to track a given assertion
     * in unsat cores.
     * @see checkWithAssumptions
     * @see unsatCore
     * */
    fun assertAndTrack(expr: KExpr<KBoolSort>): KExpr<KBoolSort>

    /**
     * Create a backtracking point for assertion stack.
     *
     * @see pop
     * */
    fun push()

    /**
     * Revert solver assertions state to previously created backtracking point.
     *
     * @param n number of pushed scopes to revert.
     * @see push
     * */
    fun pop(n: UInt = 1u)

    /**
     *  Performs satisfiability check of currently asserted expressions.
     *
     * @param timeout solver check timeout. When time limit is reached [KSolverStatus.UNKNOWN] is returned.
     * @return satisfiability check result.
     * * [KSolverStatus.SAT] assertions are satisfiable. Satisfying assignment can be retrieved via [model].
     * * [KSolverStatus.UNSAT] assertions are unsatisfiable. Unsat core can be retrieved via [unsatCore].
     * * [KSolverStatus.UNKNOWN] solver failed to check satisfiability due to timeout or internal reasons.
     * Brief reason description may be obtained via [reasonOfUnknown].
     * */
    fun check(timeout: Duration = Duration.INFINITE): KSolverStatus

    /**
     *  Performs satisfiability check of currently asserted expressions and provided assumptions.
     *
     * In case of [KSolverStatus.UNSAT] result assumptions are used for unsat core generation.
     * @see check
     * @see unsatCore
     * */
    fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration = Duration.INFINITE): KSolverStatus

    /**
     * Retrieve the model for the last [check] or [checkWithAssumptions].
     * */
    fun model(): KModel

    /**
     * Retrieve the unsat core for the last [check] or [checkWithAssumptions].
     *
     * Unsat core consists only of:
     * 1. assumptions provided in [checkWithAssumptions]
     * 2. track variables corresponding to expressions asserted with [assertAndTrack]
     * */
    fun unsatCore(): List<KExpr<KBoolSort>>

    /**
     * Retrieve a brief explanation of an [KSolverStatus.UNKNOWN] result.
     * The format of resulting string is solver implementation dependent.
     * */
    fun reasonOfUnknown(): String
}
