package io.ksmt.test.benchmarks

import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.bitwuzla.KBitwuzlaSolver
import io.ksmt.solver.bitwuzla.KBitwuzlaSolverConfiguration
import io.ksmt.solver.yices.KYicesSolver
import io.ksmt.solver.yices.KYicesSolverConfiguration
import io.ksmt.solver.z3.KZ3SMTLibParser
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.solver.z3.KZ3SolverConfiguration
import io.ksmt.sort.KBoolSort
import io.ksmt.symfpu.solver.SymfpuSolver
import java.nio.file.Path
import kotlin.io.path.Path
import kotlin.io.path.appendText
import kotlin.io.path.createFile
import kotlin.io.path.exists
import kotlin.system.measureNanoTime
import kotlin.time.Duration.Companion.seconds
import kotlin.time.ExperimentalTime
import kotlin.time.measureTimedValue

@Execution(ExecutionMode.SAME_THREAD)
class SymFPUBenchmarks : BenchmarksBasedTest() {


    @ParameterizedTest(name = "{0}")
    @MethodSource("testData")
    fun testSolver(name: String, samplePath: Path) {
        val solverName = System.getenv("solver")
            ?: throw IllegalStateException("solver environment variable is not set")
        val solver = mapSolvers[solverName] ?: throw IllegalStateException("solver $solverName is not supported")
        measureKsmtAssertionTime(name, samplePath, solverName, solver)
    }

    private val mapSolvers = mapOf(
        "SymfpuZ3" to ::SymfpuZ3Solver,
        "SymfpuYices" to ::SymfpuYicesSolver,
        "SymfpuBitwuzla" to ::SymfpuBitwuzlaSolver,
        "Z3" to ::KZ3Solver,
        "Bitwuzla" to ::KBitwuzlaSolver,
    )

    private fun getTheory(name: String) = when {
        name.startsWith("QF_FP_") -> "QF_FP"
        name.startsWith("QF_BVFP") -> "QF_BVFP"
        name.startsWith("QF_ABVFP") -> "QF_ABVFP"
        else -> throw IllegalStateException("unknown theory for $name")
    }


//./gradlew :ksmt-test:test --tests "io.ksmt.test.benchmarks.SymFPUBenchmarksBasedTest" --no-daemon --continue -PrunBenchmarksBasedTests=true -Psolver=Z3
// --no-daemon --continue -PrunBenchmarksBasedTests=true -Psolver=Z3

    @OptIn(ExperimentalTime::class)
    private fun measureKsmtAssertionTime(
        sampleName: String, samplePath: Path, solverName: String,
        solverConstructor: (ctx: KContext) -> KSolver<*>,
    ) {
        try {
            with(KContext()) {
                val assertions: List<KExpr<KBoolSort>> = KZ3SMTLibParser(this).parse(samplePath)
                solverConstructor(this).use { solver ->
                    val assertTime = measureNanoTime {
                        assertions.forEach { solver.assert(it) }
                    }
                    val (status, duration) = measureTimedValue {
                        solver.check(TIMEOUT)
                    }
                    val checkTime = duration.inWholeNanoseconds
                    saveData(sampleName, getTheory(sampleName), solverName, assertTime, checkTime, assertTime + checkTime, status)
                }
            }
        } catch (t: Throwable) {
            System.err.println("THROWS $solverName.$sampleName: ${t.message}")
        }
    }

    private fun saveData(
        sampleName: String, theory: String,
        solverName: String, assertTime: Long,
        checkTime: Long, totalTime: Long,
        status: KSolverStatus,
    ) {
        val data = "$sampleName,$theory,$solverName,$assertTime,$checkTime,$totalTime,$status"
        Path("data.csv").appendText("$data\n")
    }


    companion object {
        private val TIMEOUT = 1.seconds

        @JvmStatic
        fun testData() = testData {
            it.startsWith("QF_FP_") || it.startsWith("QF_BVFP") || it.startsWith("QF_ABVFP")
        }.ensureNotEmpty().also { println("current chunk: ${it.size}") }.let {
            it + it + it + it + it // 5 repeats for each test
        } // 68907 total

        @BeforeAll
        @JvmStatic
        fun createData() {
            Path("data.csv").apply {
                if (!exists()) createFile()
            }
        }
    }
}


class SymfpuZ3Solver(ctx: KContext) : SymfpuSolver<KZ3SolverConfiguration>(KZ3Solver(ctx), ctx)
class SymfpuYicesSolver(ctx: KContext) : SymfpuSolver<KYicesSolverConfiguration>(KYicesSolver(ctx), ctx)
class SymfpuBitwuzlaSolver(ctx: KContext) : SymfpuSolver<KBitwuzlaSolverConfiguration>(KBitwuzlaSolver(ctx), ctx)
