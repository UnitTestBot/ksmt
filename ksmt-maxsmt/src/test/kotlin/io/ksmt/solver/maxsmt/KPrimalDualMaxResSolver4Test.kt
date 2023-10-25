package io.ksmt.solver.maxsmt

import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolver
import io.ksmt.solver.maxsmt.solvers.KPrimalDualMaxResSolver
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.solver.z3.KZ3SolverConfiguration

class KPrimalDualMaxResSolver4Test : KMaxSMTSolverTest() {
    override fun getSolver(): KMaxSMTSolver<KZ3SolverConfiguration> = with(ctx) {
        val z3Solver = KZ3Solver(this)
        return KPrimalDualMaxResSolver(this, z3Solver, KMaxSMTContext(getMultipleCores = false))
    }
}
