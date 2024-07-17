package io.ksmt.solver.maxsmt.test.configurations

import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.maxsmt.KMaxSMTContext
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolverInterface
import io.ksmt.solver.maxsmt.test.smt.KMaxSMTBenchmarkTest
import io.ksmt.solver.maxsmt.test.subopt.KSubOptMaxSMTBenchmarkTest
import io.ksmt.solver.maxsmt.test.utils.MaxSmtSolver
import io.ksmt.solver.maxsmt.test.utils.Solver

class KDualMaxRes3SMTBenchmarkTest : KMaxSMTBenchmarkTest() {
    override val maxSmtCtx = KMaxSMTContext(minimizeCores = true)

    override fun getSolver(solver: Solver): KMaxSMTSolverInterface<out KSolverConfiguration> {
        val smtSolver = getSmtSolver(solver)
        return getMaxSmtSolver(MaxSmtSolver.PRIMAL_DUAL_MAXRES, smtSolver)
    }
}

class KSubOptDualMaxRes3SMTBenchmarkTest : KSubOptMaxSMTBenchmarkTest() {
    override val maxSmtCtx = KMaxSMTContext(minimizeCores = true)

    override fun getSolver(solver: Solver): KMaxSMTSolverInterface<out KSolverConfiguration> {
        val smtSolver = getSmtSolver(solver)
        return getMaxSmtSolver(MaxSmtSolver.PRIMAL_DUAL_MAXRES, smtSolver)
    }
}
