package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.decl.KConstDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.KSolverStatus
import org.ksmt.sort.KBoolSort
import kotlin.time.Duration

open class SymfpuSolver<Config : KSolverConfiguration>(
    val solver: KSolver<Config>,
    val ctx: KContext,
) : KSolver<Config> {

    private val transformer = FpToBvTransformer(ctx)

    override fun configure(configurator: Config.() -> Unit) {
        solver.configure(configurator)
    }

    override fun assert(expr: KExpr<KBoolSort>) = solver.assert(transformer.applyAndGetExpr(expr)) // AndGetExpr

    override fun assertAndTrack(expr: KExpr<KBoolSort>, trackVar: KConstDecl<KBoolSort>) =
        solver.assertAndTrack(transformer.applyAndGetExpr(expr), trackVar)

    override fun push() = solver.push()

    override fun pop(n: UInt) = solver.pop(n)


    override fun check(timeout: Duration): KSolverStatus = solver.check(timeout)

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus =
        solver.checkWithAssumptions(assumptions.map(transformer::applyAndGetExpr), timeout)

    override fun model(): KModel = SymFPUModel(solver.model(), ctx, transformer)

    override fun unsatCore(): List<KExpr<KBoolSort>> = solver.unsatCore()

    override fun reasonOfUnknown(): String = solver.reasonOfUnknown()

    override fun interrupt() = solver.interrupt()

    override fun close() = solver.close()

}
