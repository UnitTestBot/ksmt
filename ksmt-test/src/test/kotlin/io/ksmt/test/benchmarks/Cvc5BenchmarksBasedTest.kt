package io.ksmt.test.benchmarks

import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import io.ksmt.KContext
import io.ksmt.solver.cvc5.KCvc5Solver
import io.ksmt.solver.runner.KSolverRunnerManager
import java.nio.file.Path

class Cvc5BenchmarksBasedTest : BenchmarksBasedTest() {

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("cvc5TestData")
    fun testSolver(name: String, samplePath: Path) = testSolver(name, samplePath) { ctx ->
        solverManager.createCvc5TestSolver(ctx)
    }

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("cvc5TestData")
    fun testModelConversion(name: String, samplePath: Path) = testModelConversion(name, samplePath) { ctx ->
        solverManager.createCvc5TestSolver(ctx)
    }

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("cvc5TestData")
    fun testConverter(name: String, samplePath: Path) = testConverter(name, samplePath) { assertions ->
        internalizeAndConvertCvc5(assertions)
    }

    companion object {
        @JvmStatic
        fun cvc5TestData() = testData()

        /**
         * Resource limit to prevent solver from consuming huge amounts of native memory.
         * The value is measured in some abstract resource usage units and chosen to maintain
         * a balance between resource usage and the number of unknown check-sat results.
         * */
        const val RESOURCE_LIMIT = 4000

        fun KSolverRunnerManager.createCvc5TestSolver(ctx: KContext) =
            createSolver(ctx, KCvc5Solver::class).apply {
                configure { setCvc5Option("rlimit", "$RESOURCE_LIMIT") }
            }
    }
}
