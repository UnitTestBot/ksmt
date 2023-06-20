package io.ksmt.test.benchmarks

import io.ksmt.solver.z3.KZ3SolverUniversalConfiguration
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import java.nio.file.Path

@Execution(ExecutionMode.CONCURRENT)
class SymFPUBenchmarksBasedTest : BenchmarksBasedTest() {

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("testData")
    fun testModelConversion(name: String, samplePath: Path) = testModelConversion(name, samplePath) { ctx ->
        solverManager.run {
            registerSolver(SymfpuZ3Solver::class, KZ3SolverUniversalConfiguration::class)
            createSolver(ctx, SymfpuZ3Solver::class)
        }
    }

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("testData")
    fun testSolver(name: String, samplePath: Path) = testSolver(name, samplePath) { ctx ->
        solverManager.run {
            registerSolver(SymfpuZ3Solver::class, KZ3SolverUniversalConfiguration::class)
            createSolver(ctx, SymfpuZ3Solver::class)
        }
    }

    companion object {
        @JvmStatic
        fun testData() = testData {
            "FP" in it
        }.ensureNotEmpty()
    }
}


