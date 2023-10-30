package io.ksmt.solver.maxsmt.test.smt

import io.ksmt.solver.maxsmt.KMaxSMTContext
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolver
import io.ksmt.solver.maxsmt.solvers.KPrimalDualMaxResSolver
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.solver.z3.KZ3SolverConfiguration
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import java.nio.file.Path

class KPrimalDualMaxResSMTBenchmarkTest : KMaxSMTBenchmarkTest() {
    override fun getSolver(): KMaxSMTSolver<KZ3SolverConfiguration> = with(ctx) {
        val z3Solver = KZ3Solver(this)
        return KPrimalDualMaxResSolver(this, z3Solver, KMaxSMTContext())
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSMTTestData")
    fun maxSMTTest(name: String, samplePath: Path) {
        maxSMTTest(name, samplePath) { assertions ->
            assertions
        }
    }
}
