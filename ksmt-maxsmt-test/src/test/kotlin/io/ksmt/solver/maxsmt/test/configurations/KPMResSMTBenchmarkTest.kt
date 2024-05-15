package io.ksmt.solver.maxsmt.test.configurations

import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.maxsmt.KMaxSMTContext
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolver
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolverInterface
import io.ksmt.solver.maxsmt.solvers.KPMResSolver
import io.ksmt.solver.maxsmt.test.smt.KMaxSMTBenchmarkTest
import io.ksmt.solver.maxsmt.test.utils.MaxSmtSolver
import io.ksmt.solver.maxsmt.test.utils.Solver

class KPMResSMTBenchmarkTest : KMaxSMTBenchmarkTest() {
    // TODO: in fact for KPMRes we don't need MaxSMTContext.
    override val maxSmtCtx = KMaxSMTContext()

    override fun getSolver(solver: Solver): KMaxSMTSolverInterface<out KSolverConfiguration> {
        val smtSolver = getSmtSolver(solver)
        return getMaxSmtSolver(MaxSmtSolver.PMRES, smtSolver)
    }
}
