package io.ksmt.solver.maxsmt.test.smt

import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolver
import io.ksmt.solver.maxsmt.solvers.KPMResSolver
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.solver.z3.KZ3SolverConfiguration
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import java.nio.file.Path

class KPMResSMTBenchmarkTest : KMaxSMTBenchmarkTest() {
    override fun getSolver(): KMaxSMTSolver<KZ3SolverConfiguration> = with(ctx) {
        val z3Solver = KZ3Solver(this)
        return KPMResSolver(this, z3Solver)
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSMTTestData")
    fun maxSMTTest(name: String, samplePath: Path) {
        maxSMTTest(name, samplePath) { assertions ->
            assertions
        }
    }
}
