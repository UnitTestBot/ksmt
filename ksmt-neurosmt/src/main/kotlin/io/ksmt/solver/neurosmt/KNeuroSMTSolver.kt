package io.ksmt.solver.neurosmt

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KBoolSort
import kotlin.time.Duration

class KNeuroSMTSolver(private val ctx: KContext) : KSolver<KNeuroSMTSolverConfiguration> {
    override fun configure(configurator: KNeuroSMTSolverConfiguration.() -> Unit) {
        TODO("Not yet implemented")
    }

    override fun assert(expr: KExpr<KBoolSort>) {
        // TODO("Not yet implemented")
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>) {
        TODO("Not yet implemented")
    }

    override fun push() {
        TODO("Not yet implemented")
    }

    override fun pop(n: UInt) {
        TODO("Not yet implemented")
    }

    override fun check(timeout: Duration): KSolverStatus {
        return KSolverStatus.SAT
    }

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus {
        TODO("Not yet implemented")
    }

    override fun model(): KModel {
        TODO("Not yet implemented")
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> {
        TODO("Not yet implemented")
    }

    override fun reasonOfUnknown(): String {
        TODO("Not yet implemented")
    }

    override fun interrupt() {
        TODO("Not yet implemented")
    }

    override fun close() {
        // TODO("Not yet implemented")
    }
}