package io.ksmt.symfpu.solver

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KBoolSort
import kotlin.time.Duration

open class KSymFpuSolver<Config : KSolverConfiguration>(
    val solver: KSolver<Config>,
    val ctx: KContext,
    packedBvOptimizationEnabled: Boolean = true
) : KSolver<Config> {
    private val transformer = FpToBvTransformer(ctx, packedBvOptimizationEnabled)
    private val mapTransformedToOriginalAssertions =
        mutableMapOf<KExpr<KBoolSort>, KExpr<KBoolSort>>()

    override fun assert(expr: KExpr<KBoolSort>) = solver.assert(transformer.applyAndGetExpr(expr))

    override fun assertAndTrack(expr: KExpr<KBoolSort>) {
        val transformedExpr = transformer.applyAndGetExpr(expr).also { mapTransformedToOriginalAssertions[it] = expr }
        solver.assertAndTrack(transformedExpr)
    }

    override fun check(timeout: Duration): KSolverStatus = solver.check(timeout)

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus =
        solver.checkWithAssumptions(assumptions.map { expr ->
            transformer.applyAndGetExpr(expr).also { mapTransformedToOriginalAssertions[it] = expr }
        }, timeout)

    override fun model(): KModel = KSymFpuModel(solver.model(), ctx, transformer)

    override fun unsatCore(): List<KExpr<KBoolSort>> = solver.unsatCore().map {
        mapTransformedToOriginalAssertions[it]
            ?: error("Unsat core contains an expression that was not transformed")
    }

    override fun push() = solver.push()
    override fun pop(n: UInt) = solver.pop(n)

    override fun configure(configurator: Config.() -> Unit) = solver.configure(configurator)

    override fun reasonOfUnknown(): String = solver.reasonOfUnknown()

    override fun interrupt() = solver.interrupt()
    override fun close() = solver.close()
}
