package io.ksmt.maxsat.solvers

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.expr.KTrue
import io.ksmt.maxsat.KMaxSATResult
import io.ksmt.maxsat.MaxSATScopeManager
import io.ksmt.maxsat.SoftConstraint
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KBoolSort
import kotlin.time.Duration
import kotlin.time.DurationUnit
import kotlin.time.toDuration

abstract class KMaxSATSolver<T>(private val ctx: KContext, private val solver: KSolver<T>) : KSolver<KSolverConfiguration>
    where T : KSolverConfiguration {
    private val scopeManager = MaxSATScopeManager()
    protected var softConstraints = mutableListOf<SoftConstraint>()

    /**
     * Assert softly an expression with weight (aka soft constraint) into solver.
     *
     * @see checkMaxSAT
     * */
    fun assertSoft(expr: KExpr<KBoolSort>, weight: UInt) {
        require(weight > 0u) { "Soft constraint weight cannot be equal to $weight as it must be greater than 0" }

        val softConstraint = SoftConstraint(expr, weight)
        softConstraints.add(softConstraint)
        scopeManager.incrementSoft()
    }

    /**
     * Solve maximum satisfiability problem.
     *
     * @throws NotImplementedError
     */
    abstract fun checkMaxSAT(timeout: Duration = Duration.INFINITE): KMaxSATResult

    /**
     * Check on satisfiability hard constraints with assumed soft constraints.
     *
     * @return a triple of solver status, unsat core (if exists, empty list otherwise) and model
     * (if exists, null otherwise).
     */
    protected fun checkSAT(assumptions: List<SoftConstraint>, timeout: Duration):
        Triple<KSolverStatus, List<KExpr<KBoolSort>>, KModel?> =
        when (val status = solver.checkWithAssumptions(assumptions.map { x -> x.expression }, timeout)) {
            KSolverStatus.SAT -> Triple(status, listOf(), solver.model())
            KSolverStatus.UNSAT -> Triple(status, solver.unsatCore(), null)
            KSolverStatus.UNKNOWN -> Triple(status, listOf(), null)
        }

    protected fun getSatSoftConstraintsByModel(model: KModel): List<SoftConstraint> {
        return softConstraints.filter { model.eval(it.expression).internEquals(KTrue(ctx)) }
    }

    protected fun computeRemainingTime(timeout: Duration, clockStart: Long): Duration {
        val msUnit = DurationUnit.MILLISECONDS
        return timeout - (System.currentTimeMillis().toDuration(msUnit) - clockStart.toDuration(msUnit))
    }

    override fun configure(configurator: KSolverConfiguration.() -> Unit) {
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
        solver.interrupt()
    }

    override fun close() {
        solver.close()
    }
}