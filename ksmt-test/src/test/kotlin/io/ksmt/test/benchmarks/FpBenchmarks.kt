package io.ksmt.test.benchmarks

import org.junit.jupiter.api.BeforeAll
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
import io.ksmt.symfpu.solver.KSymFpuSolver
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import java.nio.file.Path
import kotlin.io.path.*
import kotlin.system.measureNanoTime
import kotlin.time.Duration.Companion.seconds

class Z3FpBenchmarks : FpBenchmarks() {
    @Execution(ExecutionMode.SAME_THREAD)
    @ParameterizedTest(name = "{0}")
    @MethodSource("testData")
    fun testSolver(name: String, samplePath: Path) =
        measureKsmtAssertionTime(name, samplePath, "Z3") { ctx -> KZ3Solver(ctx) }
}

class BitwuzlaFpBenchmarks : FpBenchmarks() {
    @Execution(ExecutionMode.SAME_THREAD)
    @ParameterizedTest(name = "{0}")
    @MethodSource("testData")
    fun testSolver(name: String, samplePath: Path) =
        measureKsmtAssertionTime(name, samplePath, "Bitwuzla") { ctx -> KBitwuzlaSolver(ctx) }
}

class SymFpuZ3Solver(ctx: KContext) : KSymFpuSolver<KZ3SolverConfiguration>(KZ3Solver(ctx), ctx)

class Z3WithSymFpuFpBenchmarks : FpBenchmarks() {
    @Execution(ExecutionMode.SAME_THREAD)
    @ParameterizedTest(name = "{0}")
    @MethodSource("testData")
    fun testSolver(name: String, samplePath: Path) =
        measureKsmtAssertionTime(name, samplePath, "SymfpuZ3") { ctx -> SymFpuZ3Solver(ctx) }
}

class SymFpuBitwuzlaSolver(ctx: KContext) : KSymFpuSolver<KBitwuzlaSolverConfiguration>(KBitwuzlaSolver(ctx), ctx)

class BitwuzlaWithSymFpuFpBenchmarks : FpBenchmarks() {
    @Execution(ExecutionMode.SAME_THREAD)
    @ParameterizedTest(name = "{0}")
    @MethodSource("testData")
    fun testSolver(name: String, samplePath: Path) =
        measureKsmtAssertionTime(name, samplePath, "SymfpuBitwuzla") { ctx -> SymFpuBitwuzlaSolver(ctx) }
}

class SymFpuYicesSolver(ctx: KContext) : KSymFpuSolver<KYicesSolverConfiguration>(KYicesSolver(ctx), ctx)

class YicesWithSymFpuFpBenchmarks : FpBenchmarks() {
    @Execution(ExecutionMode.SAME_THREAD)
    @ParameterizedTest(name = "{0}")
    @MethodSource("testData")
    fun testSolver(name: String, samplePath: Path) =
        measureKsmtAssertionTime(name, samplePath, "SymfpuYices") { ctx -> SymFpuYicesSolver(ctx) }
}

abstract class FpBenchmarks : BenchmarksBasedTest() {
    private fun getTheory(name: String) = when {
        name.startsWith("QF_FP_") -> "QF_FP"
        name.startsWith("QF_BVFP") -> "QF_BVFP"
        name.startsWith("QF_ABVFP") -> "QF_ABVFP"
        else -> throw IllegalStateException("unknown theory for $name")
    }

    fun measureKsmtAssertionTime(
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

                    val status: KSolverStatus
                    val checkTime = measureNanoTime {
                        status = solver.check(TIMEOUT)
                    }

                    saveData(
                        sampleName,
                        getTheory(sampleName),
                        solverName,
                        assertTime,
                        checkTime,
                        assertTime + checkTime,
                        status
                    )
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
        dataPath.appendText("$data\n")
    }

    companion object {
        private val TIMEOUT = 1.seconds

        private const val REPORT_FILE_NAME = "data.csv"
        private val dataPath by lazy { Path(REPORT_FILE_NAME) }

        @JvmStatic
        fun testData() = testData {
            it.startsWith("QF_FP_") || it.startsWith("QF_BVFP") || it.startsWith("QF_ABVFP")
        }.ensureNotEmpty()
            .also { println("current chunk: ${it.size}") }
            .let {
                it + it + it + it + it // 5 repeats for each test
            } // 68907 total

        @BeforeAll
        @JvmStatic
        fun createData() {
            dataPath.createIfNotExists()
        }

        @JvmStatic
        fun main(args: Array<String>) {
            mergeData(Path(args.single()))
        }

        @OptIn(ExperimentalPathApi::class)
        private fun mergeData(rootPath: Path) {
            val mergedReport = Path("merged_$REPORT_FILE_NAME").also { it.createIfNotExists() }
            rootPath.walk().filter { it.name == REPORT_FILE_NAME }.forEach { report ->
                mergedReport.appendText(report.readText())
            }
        }

        private fun Path.createIfNotExists() {
            if (!exists()) createFile()
        }
    }
}
