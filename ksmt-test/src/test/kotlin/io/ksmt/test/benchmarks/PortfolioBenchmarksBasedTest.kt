package io.ksmt.test.benchmarks

import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import io.ksmt.solver.bitwuzla.KBitwuzlaSolver
import io.ksmt.solver.portfolio.KPortfolioSolverManager
import io.ksmt.solver.z3.KZ3Solver
import java.nio.file.Path
import kotlin.time.Duration.Companion.seconds

class PortfolioBenchmarksBasedTest : BenchmarksBasedTest() {

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("portfolioTestData")
    fun testModelConversion(name: String, samplePath: Path) =
        testModelConversion(name, samplePath) { ctx ->
            portfolioSolverManager.createPortfolioSolver(ctx)
        }

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("portfolioTestData")
    fun testSolver(name: String, samplePath: Path) =
        testSolver(name, samplePath) { ctx ->
            portfolioSolverManager.createPortfolioSolver(ctx)
        }

    companion object {
        @JvmStatic
        fun portfolioTestData() = testData
            .filter { it.name.startsWith("QF_") }
            .filter { "BV" in it.name }
            .ensureNotEmpty()

        private lateinit var portfolioSolverManager: KPortfolioSolverManager

        @BeforeAll
        @JvmStatic
        fun initPortfolioPool() {
            portfolioSolverManager = KPortfolioSolverManager(
                solvers = listOf(KZ3Solver::class, KBitwuzlaSolver::class),
                portfolioPoolSize = 4,
                hardTimeout = 3.seconds,
                workerProcessIdleTimeout = 50.seconds
            )
        }

        @AfterAll
        @JvmStatic
        fun closePortfolioPool() {
            portfolioSolverManager.close()
        }
    }
}
