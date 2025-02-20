package io.ksmt.solver.neurosmt

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.neurosmt.runtime.NeuroSMTModelRunner
import io.ksmt.sort.KBoolSort
import kotlin.time.Duration

class KNeuroSMTSolver(
    private val ctx: KContext,
    ordinalsPath: String, embeddingPath: String, convPath: String, decoderPath: String,
    private val threshold: Double = 0.5
) : KSolver<KNeuroSMTSolverConfiguration> {

    private val modelRunner = NeuroSMTModelRunner(ctx, ordinalsPath, embeddingPath, convPath, decoderPath)
    private val asserts = mutableListOf<MutableList<KExpr<KBoolSort>>>(mutableListOf())

    override fun configure(configurator: KNeuroSMTSolverConfiguration.() -> Unit) {
        TODO("Not yet implemented")
    }

    override fun assert(expr: KExpr<KBoolSort>) {
        asserts.last().add(expr)
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>) {
        assert(expr)
    }

    override fun push() {
        asserts.add(mutableListOf())
    }

    override fun pop(n: UInt) {
        repeat(n.toInt()) {
            asserts.removeLast()
        }
    }

    override fun check(timeout: Duration): KSolverStatus {
        val prob = with(ctx) {
            modelRunner.run(mkAnd(asserts.flatten()))
        }

        return if (prob > threshold) {
            KSolverStatus.SAT
        } else {
            KSolverStatus.UNSAT
        }
    }

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus {
        val prob = with(ctx) {
            modelRunner.run(mkAnd(asserts.flatten() + assumptions))
        }

        return if (prob > threshold) {
            KSolverStatus.SAT
        } else {
            KSolverStatus.UNSAT
        }
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
        modelRunner.close()
        asserts.clear()
    }
}