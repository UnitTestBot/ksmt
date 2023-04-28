package io.ksmt.solver.runner

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.KSolverUniversalConfigurationBuilder
import io.ksmt.solver.model.KModelImpl
import io.ksmt.sort.KBoolSort
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.time.Duration

class CustomSolverTest {

    @Test
    fun testCustomSolverRunner() {
        KSolverRunnerManager().use { solverManager ->
            solverManager.registerSolver(CustomSolverStub::class, CustomSolverConfigBuilder::class)

            val ctx = KContext()
            solverManager.createSolver(ctx, CustomSolverStub::class).use { solver ->
                val status = solver.check()
                assertEquals(KSolverStatus.UNKNOWN, status)

                val reason = solver.reasonOfUnknown()
                assertEquals(CUSTOM_SOLVER_STUB_UNKNOWN_REASON, reason)
            }
        }
    }

    interface CustomSolverConfig : KSolverConfiguration

    class CustomSolverStub(private val ctx: KContext) : KSolver<CustomSolverConfig> {
        override fun check(timeout: Duration): KSolverStatus = KSolverStatus.UNKNOWN

        override fun checkWithAssumptions(
            assumptions: List<KExpr<KBoolSort>>,
            timeout: Duration
        ): KSolverStatus = KSolverStatus.UNKNOWN

        override fun model(): KModel = KModelImpl(ctx, emptyMap(), emptyMap())

        override fun unsatCore(): List<KExpr<KBoolSort>> = emptyList()

        override fun reasonOfUnknown(): String = CUSTOM_SOLVER_STUB_UNKNOWN_REASON

        override fun configure(configurator: CustomSolverConfig.() -> Unit) {
        }

        override fun assert(expr: KExpr<KBoolSort>) {
        }

        override fun assertAndTrack(expr: KExpr<KBoolSort>) {
        }

        override fun push() {
        }

        override fun pop(n: UInt) {
        }

        override fun interrupt() {
        }

        override fun close() {
        }
    }

    class CustomSolverConfigBuilder(
        val builder: KSolverUniversalConfigurationBuilder
    ) : CustomSolverConfig {
        override fun setBoolParameter(param: String, value: Boolean) {
            error("Solver stub is not configurable")
        }

        override fun setIntParameter(param: String, value: Int) {
            error("Solver stub is not configurable")
        }

        override fun setStringParameter(param: String, value: String) {
            error("Solver stub is not configurable")
        }

        override fun setDoubleParameter(param: String, value: Double) {
            error("Solver stub is not configurable")
        }
    }

    companion object {
        const val CUSTOM_SOLVER_STUB_UNKNOWN_REASON = "CUSTOM SOLVER STUB"
    }
}
