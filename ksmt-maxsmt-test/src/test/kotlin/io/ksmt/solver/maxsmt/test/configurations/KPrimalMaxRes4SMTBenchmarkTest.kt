package io.ksmt.solver.maxsmt.test.configurations

import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.maxsmt.KMaxSMTContext
import io.ksmt.solver.maxsmt.KMaxSMTContext.Strategy.PrimalMaxRes
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolver
import io.ksmt.solver.maxsmt.solvers.KPrimalDualMaxResSolver
import io.ksmt.solver.maxsmt.test.smt.KMaxSMTBenchmarkTest
import io.ksmt.solver.maxsmt.test.utils.Solver

class KPrimalMaxRes4SMTBenchmarkTest : KMaxSMTBenchmarkTest() {
    override val maxSmtCtx = KMaxSMTContext(strategy = PrimalMaxRes, getMultipleCores = true)

    override fun getSolver(solver: Solver): KMaxSMTSolver<KSolverConfiguration> = with(ctx) {
        val smtSolver = getSmtSolver(solver)
        return KPrimalDualMaxResSolver(this, smtSolver, maxSmtCtx)
    }
}
