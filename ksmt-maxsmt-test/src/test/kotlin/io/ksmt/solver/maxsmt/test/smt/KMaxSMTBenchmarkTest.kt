package io.ksmt.solver.maxsmt.test.smt

import io.github.oshai.kotlinlogging.KotlinLogging
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.runner.core.KsmtWorkerArgs
import io.ksmt.runner.core.KsmtWorkerFactory
import io.ksmt.runner.core.KsmtWorkerPool
import io.ksmt.runner.core.RdServer
import io.ksmt.runner.core.WorkerInitializationFailedException
import io.ksmt.runner.generated.models.TestProtocolModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus.SAT
import io.ksmt.solver.bitwuzla.KBitwuzlaSolver
import io.ksmt.solver.cvc5.KCvc5Solver
import io.ksmt.solver.maxsmt.KMaxSMTResult
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolver
import io.ksmt.solver.maxsmt.test.KMaxSMTBenchmarkBasedTest
import io.ksmt.solver.maxsmt.test.parseMaxSMTTestInfo
import io.ksmt.solver.maxsmt.test.statistics.JsonStatisticsHelper
import io.ksmt.solver.maxsmt.test.statistics.MaxSMTTestStatistics
import io.ksmt.solver.maxsmt.test.utils.Solver
import io.ksmt.solver.maxsmt.test.utils.Solver.BITWUZLA
import io.ksmt.solver.maxsmt.test.utils.Solver.CVC5
import io.ksmt.solver.maxsmt.test.utils.Solver.YICES
import io.ksmt.solver.maxsmt.test.utils.Solver.Z3
import io.ksmt.solver.maxsmt.test.utils.getRandomString
import io.ksmt.solver.yices.KYicesSolver
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.sort.KBoolSort
import io.ksmt.test.TestRunner
import io.ksmt.test.TestWorker
import io.ksmt.test.TestWorkerProcess
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import java.io.File
import java.nio.file.Path
import java.nio.file.Paths
import kotlin.io.path.extension
import kotlin.system.measureTimeMillis
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

abstract class KMaxSMTBenchmarkTest : KMaxSMTBenchmarkBasedTest {
    protected fun getSmtSolver(solver: Solver): KSolver<out KSolverConfiguration> = with(ctx) {
        return when (solver) {
            Z3 -> KZ3Solver(this)
            BITWUZLA -> KBitwuzlaSolver(this)
            CVC5 -> KCvc5Solver(this)
            YICES -> KYicesSolver(this)
        }
    }

    abstract fun getSolver(solver: Solver): KMaxSMTSolver<KSolverConfiguration>

    protected val ctx: KContext = KContext()
    private lateinit var maxSMTSolver: KMaxSMTSolver<out KSolverConfiguration>
    private val logger = KotlinLogging.logger {}

    private fun initSolver(solver: Solver) {
        maxSMTSolver = getSolver(solver)
    }

    @AfterEach
    fun closeSolver() = maxSMTSolver.close()

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSMTTestData")
    fun maxSMTZ3Test(name: String, samplePath: Path) {
        testMaxSMTSolver(name, samplePath, { assertions -> assertions }, Z3)
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSMTTestData")
    fun maxSMTBitwuzlaTest(name: String, samplePath: Path) {
        testMaxSMTSolver(name, samplePath, { assertions -> internalizeAndConvertBitwuzla(assertions) }, BITWUZLA)
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSMTTestData")
    fun maxSMTCvc5Test(name: String, samplePath: Path) {
        testMaxSMTSolver(name, samplePath, { assertions -> internalizeAndConvertCvc5(assertions) }, CVC5)
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSMTTestData")
    fun maxSMTYicesTest(name: String, samplePath: Path) {
        testMaxSMTSolver(name, samplePath, { assertions -> internalizeAndConvertYices(assertions) }, YICES)
    }

    private fun testMaxSMTSolver(
        name: String,
        samplePath: Path,
        mkKsmtAssertions: suspend TestRunner.(List<KExpr<KBoolSort>>) -> List<KExpr<KBoolSort>>,
        solver: Solver = Z3,
    ) {
        initSolver(solver)

        val extension = "smt2"
        require(samplePath.extension == extension) {
            "File extension cannot be '${samplePath.extension}' as it must be $extension"
        }

        lateinit var ksmtAssertions: List<KExpr<KBoolSort>>
        val testStatistics = MaxSMTTestStatistics(name, solver)

        try {
            testWorkers.withWorker(ctx) { worker ->
                val assertions = worker.parseFile(samplePath)
                val convertedAssertions = worker.convertAssertions(assertions)
                ksmtAssertions = worker.mkKsmtAssertions(convertedAssertions)
            }
        } catch (ex: IgnoreTestException) {
            testStatistics.ignoredTest = true
            testStatistics.exceptionMessage = ex.message.toString()
            jsonHelper.appendTestStatisticsToFile(testStatistics)
            throw ex
        } catch (ex: Exception) {
            testStatistics.failedOnParsingOrConvertingExpressions = true
            testStatistics.exceptionMessage = ex.message.toString()
            jsonHelper.appendTestStatisticsToFile(testStatistics)
            throw ex
        }

        val maxSmtTestPath = File(samplePath.toString().removeSuffix(extension) + "maxsmt").toPath()
        val maxSmtTestInfo = parseMaxSMTTestInfo(maxSmtTestPath)

        val softConstraintsSize = maxSmtTestInfo.softConstraintsWeights.size

        val softExpressions =
            ksmtAssertions.subList(ksmtAssertions.size - softConstraintsSize, ksmtAssertions.size)
        val hardExpressions = ksmtAssertions.subList(0, ksmtAssertions.size - softConstraintsSize)

        hardExpressions.forEach {
            maxSMTSolver.assert(it)
        }

        maxSmtTestInfo.softConstraintsWeights
            .zip(softExpressions)
            .forEach { (weight, expr) ->
                maxSMTSolver.assertSoft(expr, weight)
            }

        lateinit var maxSMTResult: KMaxSMTResult
        val elapsedTime = measureTimeMillis {
            try {
                maxSMTResult = maxSMTSolver.checkMaxSMT(INFINITE, true)
            } catch (ex: Exception) {
                testStatistics.maxSMTCallStatistics = maxSMTSolver.collectMaxSMTStatistics()
                testStatistics.exceptionMessage = ex.message.toString()
                jsonHelper.appendTestStatisticsToFile(testStatistics)
                throw ex
            }
        }

        testStatistics.maxSMTCallStatistics = maxSMTSolver.collectMaxSMTStatistics()

        logger.info { "Test name: [$name]" }
        logger.info { "Elapsed time (MaxSMT call): [${elapsedTime}ms]" }

        try {
            assertTrue(maxSMTResult.maxSMTSucceeded, "MaxSMT was not successful [$name]")
            assertEquals(SAT, maxSMTResult.hardConstraintsSatStatus, "Hard constraints must be SAT")

            val satSoftConstraintsWeightsSum = maxSMTResult.satSoftConstraints.sumOf { it.weight }
            if (maxSmtTestInfo.satSoftConstraintsWeightsSum != satSoftConstraintsWeightsSum.toULong()) {
                testStatistics.checkedSoftConstraintsSumIsWrong = true
            }
            assertEquals(
                maxSmtTestInfo.satSoftConstraintsWeightsSum,
                satSoftConstraintsWeightsSum.toULong(),
                "Soft constraints weights sum was [$satSoftConstraintsWeightsSum], " +
                    "but must be [${maxSmtTestInfo.satSoftConstraintsWeightsSum}]",
            )
            testStatistics.passed = true
        } finally {
            jsonHelper.appendTestStatisticsToFile(testStatistics)
        }
    }

    private fun KsmtWorkerPool<TestProtocolModel>.withWorker(
        ctx: KContext,
        body: suspend (TestRunner) -> Unit,
    ) = runBlocking {
        val worker = try {
            getOrCreateFreeWorker()
        } catch (ex: WorkerInitializationFailedException) {
            ignoreTest { "worker initialization failed -- ${ex.message}" }
        }
        worker.astSerializationCtx.initCtx(ctx)
        worker.lifetime.onTermination {
            worker.astSerializationCtx.resetCtx()
        }
        try {
            TestRunner(ctx, TEST_WORKER_SINGLE_OPERATION_TIMEOUT, worker).let {
                try {
                    it.init()
                    body(it)
                } finally {
                    it.delete()
                }
            }
        } catch (ex: TimeoutCancellationException) {
            ignoreTest { "worker timeout -- ${ex.message}" }
        } finally {
            worker.release()
        }
    }

    // See [handleIgnoredTests]
    private inline fun ignoreTest(message: () -> String?): Nothing {
        throw IgnoreTestException(message())
    }

    class IgnoreTestException(message: String?) : Exception(message)

    companion object {
        val TEST_WORKER_SINGLE_OPERATION_TIMEOUT = 25.seconds

        internal lateinit var testWorkers: KsmtWorkerPool<TestProtocolModel>
        private lateinit var jsonHelper: JsonStatisticsHelper

        @BeforeAll
        @JvmStatic
        fun initWorkerPools() {
            testWorkers = KsmtWorkerPool(
                maxWorkerPoolSize = 4,
                workerProcessIdleTimeout = 10.minutes,
                workerFactory = object : KsmtWorkerFactory<TestProtocolModel> {
                    override val childProcessEntrypoint = TestWorkerProcess::class
                    override fun updateArgs(args: KsmtWorkerArgs): KsmtWorkerArgs = args
                    override fun mkWorker(id: Int, process: RdServer) = TestWorker(id, process)
                },
            )

            jsonHelper =
                JsonStatisticsHelper(
                    File(
                        "${
                            Paths.get("").toAbsolutePath()
                        }/src/test/resources/maxsmt-statistics-${getRandomString(16)}.json",
                    ),
                )
        }

        @AfterAll
        @JvmStatic
        fun closeWorkerPools() {
            testWorkers.terminate()
            jsonHelper.markLastTestStatisticsAsProcessed()
        }
    }
}
