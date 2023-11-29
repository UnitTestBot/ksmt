package io.ksmt.solver.maxsmt.test.sat

import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolver
import io.ksmt.solver.maxsmt.solvers.KPMResSolver
import io.ksmt.solver.z3.KZ3Solver

class KPMResSATBenchmarkTest : KMaxSATBenchmarkTest() {
    override fun getSolver(): KMaxSMTSolver<KSolverConfiguration> = with(ctx) {
        val z3Solver = KZ3Solver(this)
        return KPMResSolver(this, z3Solver)
    }
}
