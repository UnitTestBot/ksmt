package io.ksmt.solver.maxsmt.test.configurations

import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.maxsmt.KMaxSMTContext
import io.ksmt.solver.maxsmt.KMaxSMTContext.Strategy.PrimalMaxRes
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolverBase
import io.ksmt.solver.maxsmt.solvers.KPrimalDualMaxResSolver
import io.ksmt.solver.maxsmt.test.smt.KMaxSMTBenchmarkTest
import io.ksmt.solver.maxsmt.test.utils.Solver

class KPrimalMaxRes3SMTBenchmarkTest : KMaxSMTBenchmarkTest() {
    override val maxSmtCtx = KMaxSMTContext(strategy = PrimalMaxRes, minimizeCores = true)

    override fun getSolver(solver: Solver): KMaxSMTSolverBase<KSolverConfiguration> = with(ctx) {
        val smtSolver = getSmtSolver(solver)
        return KPrimalDualMaxResSolver(this, smtSolver, maxSmtCtx)
    }
}
