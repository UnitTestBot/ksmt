package io.ksmt.solver.maxsmt.solvers.runner

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.runner.generated.ConfigurationBuilder
import io.ksmt.runner.generated.models.SolverType
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.maxsmt.KMaxSMTContext
import io.ksmt.solver.maxsmt.KMaxSMTResult
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolverInterface
import io.ksmt.solver.maxsmt.solvers.KPrimalDualMaxResSolver
import io.ksmt.solver.runner.KSolverRunner
import io.ksmt.solver.runner.KSolverRunnerManager
import io.ksmt.sort.KBoolSort
import kotlin.time.Duration

class KMaxSMTSolverRunner<C : KSolverConfiguration>(
    manager: KMaxSMTSolverRunnerManager,
    ctx: KContext,
    maxSmtCtx: KMaxSMTContext,
    configurationBuilder: ConfigurationBuilder<C>,
    solverType: SolverType,
    customSolverInfo: KSolverRunnerManager.CustomSolverInfo? = null,
) : KSolverRunner<C>(manager, ctx, configurationBuilder, solverType, customSolverInfo),
    KMaxSMTSolverInterface<C> {

    private val maxSMTSolver = KPrimalDualMaxResSolver(ctx, this, maxSmtCtx)

    override fun assertSoft(expr: KExpr<KBoolSort>, weight: UInt) {
        maxSMTSolver.assertSoft(expr, weight)
    }

    override fun checkSubOptMaxSMT(
        timeout: Duration,
        collectStatistics: Boolean
    ): KMaxSMTResult = maxSMTSolver.checkSubOptMaxSMT(timeout, collectStatistics)

    override fun checkMaxSMT(timeout: Duration, collectStatistics: Boolean)
            : KMaxSMTResult = maxSMTSolver.checkMaxSMT(timeout, collectStatistics)

    override fun collectMaxSMTStatistics() = maxSMTSolver.collectMaxSMTStatistics()

    override fun configure(configurator: C.() -> Unit) =
        maxSMTSolver.configure(configurator)
}
