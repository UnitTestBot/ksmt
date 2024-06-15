package io.ksmt.solver.maxsmt.solvers

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.maxsmt.KMaxSMTResult
import io.ksmt.solver.maxsmt.constraints.SoftConstraint
import io.ksmt.solver.maxsmt.scope.MaxSMTScopeManager
import io.ksmt.solver.maxsmt.statistics.KMaxSMTStatistics
import io.ksmt.sort.KBoolSort
import kotlin.time.Duration

/**
 * This class implements a logic general for any MaxSMT solver.
 */
abstract class KMaxSMTSolverBase<T>(
    private val ctx: KContext,
    private val solver: KSolver<out T>,
) : KMaxSMTSolverInterface<T> where T : KSolverConfiguration {
    private val scopeManager = MaxSMTScopeManager()
    protected var softConstraints = mutableListOf<SoftConstraint>()
    protected lateinit var maxSMTStatistics: KMaxSMTStatistics
    protected var isInterrupted = false

    /**
     * Softly assert an expression with weight (aka soft constraint) into solver.
     *
     * @see checkMaxSMT
     * */
    override fun assertSoft(expr: KExpr<KBoolSort>, weight: UInt) {
        require(weight > 0u) { "Soft constraint weight cannot be equal to $weight as it must be greater than 0" }

        val softConstraint = SoftConstraint(expr, weight)
        softConstraints.add(softConstraint)
        scopeManager.incrementSoft()
    }

    /**
     * Solve maximum satisfiability modulo theories problem.
     */
    fun checkMaxSMT(timeout: Duration): KMaxSMTResult =
        checkMaxSMT(timeout, collectStatistics = false)

    /**
     * Solve maximum satisfiability modulo theories problem.
     *
     * @param collectStatistics specifies whether statistics (elapsed time to execute method etc.) should be collected or not.
     */
    abstract override fun checkMaxSMT(timeout: Duration, collectStatistics: Boolean): KMaxSMTResult

    /**
     * Get last MaxSMT launch statistics (number of queries to solver, MaxSMT timeout etc.).
     */
    override fun collectMaxSMTStatistics(): KMaxSMTStatistics {
        require(this::maxSMTStatistics.isInitialized) {
            "MaxSMT statistics is only available after MaxSMT launches with statistics collection enabled"
        }

        return maxSMTStatistics
    }

    protected fun getSatSoftConstraintsByModel(model: KModel): List<SoftConstraint> {
        return softConstraints.filter {
            model.eval(it.expression, true) == ctx.trueExpr || model.eval(
                it.expression,
                true,
            ) != ctx.falseExpr
        }.also { model.close() }
    }

    override fun configure(configurator: T.() -> Unit) {
        solver.configure(configurator)
    }

    override fun assert(expr: KExpr<KBoolSort>) {
        solver.assert(expr)
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>) {
        solver.assertAndTrack(expr)
    }

    override fun push() {
        solver.push()
        scopeManager.push()
    }

    override fun pop(n: UInt) {
        solver.pop(n)
        softConstraints = scopeManager.pop(n, softConstraints)
    }

    override fun check(timeout: Duration): KSolverStatus {
        return solver.check(timeout)
    }

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus {
        return solver.checkWithAssumptions(assumptions, timeout)
    }

    override fun model(): KModel {
        return solver.model()
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> {
        return solver.unsatCore()
    }

    override fun reasonOfUnknown(): String {
        return solver.reasonOfUnknown()
    }

    override fun interrupt() {
        isInterrupted = true
        solver.interrupt()
    }

    override fun close() {
        solver.close()
    }
}
