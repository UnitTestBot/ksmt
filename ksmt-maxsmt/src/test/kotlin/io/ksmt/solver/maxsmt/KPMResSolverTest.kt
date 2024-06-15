package io.ksmt.solver.maxsmt

import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolverBase
import io.ksmt.solver.maxsmt.solvers.KPMResSolver
import io.ksmt.solver.z3.KZ3Solver

class KPMResSolverTest : KMaxSMTSolverTest() {
    override fun getSolver(): KMaxSMTSolverBase<out KSolverConfiguration> = with(ctx) {
        val z3Solver = KZ3Solver(this)
        return KPMResSolver(this, z3Solver)
    }
}
