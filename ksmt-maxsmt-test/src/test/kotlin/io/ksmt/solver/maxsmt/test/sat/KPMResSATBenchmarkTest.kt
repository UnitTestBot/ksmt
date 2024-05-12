package io.ksmt.solver.maxsmt.test.sat

import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolver
import io.ksmt.solver.maxsmt.solvers.KPMResSolver
import io.ksmt.solver.maxsmt.test.utils.Solver

class KPMResSATBenchmarkTest : KMaxSATBenchmarkTest() {
    override fun getSolver(solver: Solver): KMaxSMTSolver<KSolverConfiguration> = with(ctx) {
        val smtSolver = getSmtSolver(solver)
        return KPMResSolver(this, smtSolver)
    }
}
