package io.ksmt.solver.maxsat

import io.ksmt.solver.maxsat.solvers.KMaxSATSolver
import io.ksmt.solver.maxsat.solvers.KPrimalDualMaxResSolver
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.solver.z3.KZ3SolverConfiguration

class KPrimalDualMaxResSolver4Test : KMaxSATSolverTest() {
    override fun getSolver(): KMaxSATSolver<KZ3SolverConfiguration> = with(ctx) {
        val z3Solver = KZ3Solver(this)
        return KPrimalDualMaxResSolver(this, z3Solver, KMaxSATContext(getMultipleCores = false))
    }
}
