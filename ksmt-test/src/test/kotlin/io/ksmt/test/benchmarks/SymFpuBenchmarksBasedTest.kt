package io.ksmt.test.benchmarks

import io.ksmt.KContext
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.solver.z3.KZ3SolverConfiguration
import io.ksmt.solver.z3.KZ3SolverUniversalConfiguration
import io.ksmt.symfpu.solver.KSymFpuSolver
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import java.nio.file.Path

class SymFpuBenchmarksBasedTest : BenchmarksBasedTest() {

    class SymFpuZ3Solver(ctx: KContext) : KSymFpuSolver<KZ3SolverConfiguration>(KZ3Solver(ctx), ctx)

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("testData")
    fun testModelConversion(name: String, samplePath: Path) = testModelConversion(name, samplePath) { ctx ->
        solverManager.run {
            registerSolver(SymFpuZ3Solver::class, KZ3SolverUniversalConfiguration::class)
            createSolver(ctx, SymFpuZ3Solver::class)
        }
    }

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("testData")
    fun testSolver(name: String, samplePath: Path) = testSolver(name, samplePath) { ctx ->
        solverManager.run {
            registerSolver(SymFpuZ3Solver::class, KZ3SolverUniversalConfiguration::class)
            createSolver(ctx, SymFpuZ3Solver::class)
        }
    }

    companion object {
        @JvmStatic
        fun testData() = testData { "FP" in it }
    }
}


