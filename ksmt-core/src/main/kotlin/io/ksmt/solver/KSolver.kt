package io.ksmt.solver

import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort
import kotlin.time.Duration

@Suppress("OVERLOADS_INTERFACE", "INAPPLICABLE_JVM_NAME")
interface KSolver<Config: KSolverConfiguration> : AutoCloseable {

    /**
     * Set solver specific options.
     * */
    fun configure(configurator: Config.() -> Unit)

    /**
     * Assert an expression into solver.
     *
     * @see check
     * */
    @JvmName("assertExpr")
    fun assert(expr: KExpr<KBoolSort>)


    /**
     * Assert multiple expressions into solver.
     *
     * @see check
     * */
    @JvmName("assertExprs")
    fun assert(exprs: List<KExpr<KBoolSort>>) = exprs.forEach { assert(it) }

    /**
     * Assert an expression into solver and track it in unsat cores.
     *
     * @see checkWithAssumptions
     * @see unsatCore
     * */
    fun assertAndTrack(expr: KExpr<KBoolSort>)

    /**
     * Assert multiple expressions into solver and track them in unsat cores.
     *
     * @see checkWithAssumptions
     * @see unsatCore
     * */
    fun assertAndTrack(exprs: List<KExpr<KBoolSort>>) = exprs.forEach { assertAndTrack(it) }

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
    @JvmOverloads
    @JvmName("pop")
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
    @JvmOverloads
    @JvmName("check")
    fun check(timeout: Duration = Duration.INFINITE): KSolverStatus

    /**
     *  Performs satisfiability check of currently asserted expressions and provided assumptions.
     *
     * In case of [KSolverStatus.UNSAT] result assumptions are used for unsat core generation.
     * @see check
     * @see unsatCore
     * */
    @JvmOverloads
    @JvmName("checkWithAssumptions")
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
     * 2. expressions asserted with [assertAndTrack]
     * */
    fun unsatCore(): List<KExpr<KBoolSort>>

    /**
     * Retrieve a brief explanation of an [KSolverStatus.UNKNOWN] result.
     * The format of resulting string is solver implementation dependent.
     * */
    fun reasonOfUnknown(): String

    /**
     * Cancel currently performing check-sat.
     * */
    fun interrupt()

    /**
     * Close solver and release acquired native resources.
     * */
    override fun close()
}
