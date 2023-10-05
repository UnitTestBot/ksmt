package io.ksmt.solver.maxsat.test.sat

import io.ksmt.solver.maxsat.solvers.KMaxSATSolver
import io.ksmt.solver.maxsat.solvers.KPMResSolver
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.solver.z3.KZ3SolverConfiguration

class KPMResSATBenchmarkTest : KMaxSATBenchmarkTest() {
    override fun getSolver(): KMaxSATSolver<KZ3SolverConfiguration> = with(ctx) {
        val z3Solver = KZ3Solver(this)
        return KPMResSolver(this, z3Solver)
    }
}
