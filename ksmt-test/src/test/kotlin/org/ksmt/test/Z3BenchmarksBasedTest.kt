package org.ksmt.test

import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import org.ksmt.solver.z3.KZ3Solver
import java.nio.file.Path

class Z3BenchmarksBasedTest : BenchmarksBasedTest() {

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("z3TestData")
    fun testConverter(name: String, samplePath: Path) =
        testConverter(name, samplePath) { assertions ->
            assertions
        }

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("z3TestData")
    fun testModelConversion(name: String, samplePath: Path) =
        testModelConversion(name, samplePath, KZ3Solver::class)

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("z3SolverTestData")
    fun testSolver(name: String, samplePath: Path) =
        testSolver(name, samplePath, KZ3Solver::class)


    companion object {
        @JvmStatic
        fun z3TestData() = testData

        @JvmStatic
        fun z3SolverTestData() = testData
            .filter { it.name !in KnownZ3Issues.z3FpFmaFalseSatSamples }
            .filter { it.name !in KnownZ3Issues.z3FpFmaFalseUnsatSamples }
            .ensureNotEmpty()
    }
}
