package io.ksmt.solver.maxsmt.test.configurations

import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.maxsmt.KMaxSMTContext
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolver
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolverInterface
import io.ksmt.solver.maxsmt.solvers.KPrimalDualMaxResSolver
import io.ksmt.solver.maxsmt.test.smt.KMaxSMTBenchmarkTest
import io.ksmt.solver.maxsmt.test.subopt.KSubOptMaxSMTBenchmarkTest
import io.ksmt.solver.maxsmt.test.utils.MaxSmtSolver
import io.ksmt.solver.maxsmt.test.utils.Solver

class KDualMaxRes2SMTBenchmarkTest : KMaxSMTBenchmarkTest() {
    override val maxSmtCtx = KMaxSMTContext(preferLargeWeightConstraintsForCores = true)

    override fun getSolver(solver: Solver): KMaxSMTSolver<KSolverConfiguration> = with(ctx) {
        val smtSolver = getSmtSolver(solver)
        return KPrimalDualMaxResSolver(this, smtSolver, maxSmtCtx)
    }
}

class KSubOptDualMaxRes2SMTBenchmarkTest : KSubOptMaxSMTBenchmarkTest() {
    override val maxSmtCtx = KMaxSMTContext(preferLargeWeightConstraintsForCores = true)

    override fun getSolver(solver: Solver): KMaxSMTSolverInterface<out KSolverConfiguration> {
        val smtSolver = getSmtSolver(solver)
        return getMaxSmtSolver(MaxSmtSolver.PRIMAL_DUAL_MAXRES, smtSolver)
    }
}
