package io.ksmt.solver.maxsmt.test.smt

import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.maxsmt.KMaxSMTContext
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolver
import io.ksmt.solver.maxsmt.solvers.KPrimalDualMaxResSolver
import io.ksmt.solver.maxsmt.test.util.Solver
import io.ksmt.solver.maxsmt.test.util.Solver.BITWUZLA
import io.ksmt.solver.maxsmt.test.util.Solver.CVC5
import io.ksmt.solver.maxsmt.test.util.Solver.YICES
import io.ksmt.solver.maxsmt.test.util.Solver.Z3
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import java.nio.file.Path

class KPrimalDualMaxResSMTBenchmarkTest : KMaxSMTBenchmarkTest() {
    override fun getSolver(solver: Solver): KMaxSMTSolver<KSolverConfiguration> = with(ctx) {
        val smtSolver: KSolver<KSolverConfiguration> = getSmtSolver(solver)
        return KPrimalDualMaxResSolver(this, smtSolver, KMaxSMTContext(preferLargeWeightConstraintsForCores = true))
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSMTTestData")
    fun maxSMTZ3Test(name: String, samplePath: Path) {
        maxSMTTest(name, samplePath, { assertions -> assertions }, Z3)
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSMTTestData")
    fun maxSMTBitwuzlaTest(name: String, samplePath: Path) {
        maxSMTTest(name, samplePath, { assertions -> internalizeAndConvertBitwuzla(assertions) }, BITWUZLA)
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSMTTestData")
    fun maxSMTCvc5Test(name: String, samplePath: Path) {
        maxSMTTest(name, samplePath, { assertions -> internalizeAndConvertCvc5(assertions) }, CVC5)
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSMTTestData")
    fun maxSMTTestYices(name: String, samplePath: Path) {
        maxSMTTest(name, samplePath, { assertions -> internalizeAndConvertYices(assertions) }, YICES)
    }
}
