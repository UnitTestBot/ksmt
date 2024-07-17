package io.ksmt.solver.maxsmt.test.subopt

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
import io.ksmt.solver.maxsmt.KMaxSMTContext
import io.ksmt.solver.maxsmt.KMaxSMTResult
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolverInterface
import io.ksmt.solver.maxsmt.solvers.KPMResSolver
import io.ksmt.solver.maxsmt.solvers.KPrimalDualMaxResSolver
import io.ksmt.solver.maxsmt.test.KMaxSMTBenchmarkBasedTest
import io.ksmt.solver.maxsmt.test.parseMaxSMTTestInfo
import io.ksmt.solver.maxsmt.test.statistics.JsonStatisticsHelper
import io.ksmt.solver.maxsmt.test.statistics.MaxSMTTestStatistics
import io.ksmt.solver.maxsmt.test.utils.MaxSmtSolver
import io.ksmt.solver.maxsmt.test.utils.Solver
import io.ksmt.solver.maxsmt.test.utils.Solver.BITWUZLA
import io.ksmt.solver.maxsmt.test.utils.Solver.CVC5
import io.ksmt.solver.maxsmt.test.utils.Solver.PORTFOLIO
import io.ksmt.solver.maxsmt.test.utils.Solver.YICES
import io.ksmt.solver.maxsmt.test.utils.Solver.Z3
import io.ksmt.solver.maxsmt.test.utils.Solver.Z3_NATIVE
import io.ksmt.solver.maxsmt.test.utils.getRandomString
import io.ksmt.solver.portfolio.KPortfolioSolver
import io.ksmt.solver.portfolio.KPortfolioSolverManager
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
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

abstract class KSubOptMaxSMTBenchmarkTest : KMaxSMTBenchmarkBasedTest {
    protected fun getMaxSmtSolver(
        maxSmtSolver: MaxSmtSolver,
        solver: KSolver<out KSolverConfiguration>
    ): KMaxSMTSolverInterface<out KSolverConfiguration> {
        when (maxSmtSolver) {
            MaxSmtSolver.PMRES -> return KPMResSolver(ctx, solver)
            MaxSmtSolver.PRIMAL_DUAL_MAXRES -> {
                // Thus, MaxSMT algorithm will be executed in the backend process.
                if (solver is KPortfolioSolver) {
                    return solver
                }
                return KPrimalDualMaxResSolver(ctx, solver, maxSmtCtx)
            }
        }
    }

    protected fun getSmtSolver(solver: Solver): KSolver<out KSolverConfiguration> = with(ctx) {
        return when (solver) {
            Z3 -> KZ3Solver(this)
            BITWUZLA -> KBitwuzlaSolver(this)
            CVC5 -> KCvc5Solver(this)
            YICES -> KYicesSolver(this)
            PORTFOLIO -> {
                solverManager.createPortfolioSolver(this)
            }

            Z3_NATIVE -> error("Unexpected solver type: Z3_NATIVE")
        }
    }

    abstract fun getSolver(solver: Solver): KMaxSMTSolverInterface<out KSolverConfiguration>

    private val ctx: KContext = KContext()
    protected abstract val maxSmtCtx: KMaxSMTContext
    private lateinit var maxSMTSolver: KMaxSMTSolverInterface<out KSolverConfiguration>

    private val logger = KotlinLogging.logger {}

    private fun initSolver(solver: Solver) {
        maxSMTSolver = getSolver(solver)
    }

    @AfterEach
    fun close() {
        maxSMTSolver.close()
        ctx.close()
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSMTTestData")
    fun maxSMTZ3Test(name: String, samplePath: Path) {
        testMaxSMTSolver(name, samplePath, Z3)
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSMTTestData")
    fun maxSMTBitwuzlaTest(name: String, samplePath: Path) {
        testMaxSMTSolver(name, samplePath, BITWUZLA)
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSMTTestData")
    fun maxSMTCvc5Test(name: String, samplePath: Path) {
        testMaxSMTSolver(name, samplePath, CVC5)
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSMTTestData")
    fun maxSMTYicesTest(name: String, samplePath: Path) {
        testMaxSMTSolver(name, samplePath, YICES)
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSMTTestData")
    fun maxSMTPortfolioTest(name: String, samplePath: Path) {
        testMaxSMTSolver(name, samplePath, PORTFOLIO)
    }

    private fun testMaxSMTSolver(
        name: String,
        samplePath: Path,
        solver: Solver = Z3,
    ) {
        initSolver(solver)

        val extension = "smt2"
        require(samplePath.extension == extension) {
            "File extension cannot be '${samplePath.extension}' as it must be $extension"
        }

        logger.info { "Test name: [$name]" }

        lateinit var convertedAssertions: List<KExpr<KBoolSort>>
        val testStatistics = MaxSMTTestStatistics(name, solver)

        try {
            testWorkers.withWorker(ctx) { worker ->
                val assertions = worker.parseFile(samplePath)
                convertedAssertions = worker.convertAssertions(assertions)
            }
        } catch (ex: IgnoreTestException) {
            testStatistics.ignoredTest = true
            testStatistics.exceptionMessage = ex.message.toString()
            jsonHelper.appendTestStatisticsToFile(testStatistics)
            logger.error { ex.message + System.lineSeparator() }
            throw ex
        } catch (ex: Exception) {
            testStatistics.failedOnParsingOrConvertingExpressions = true
            testStatistics.exceptionMessage = ex.message.toString()
            jsonHelper.appendTestStatisticsToFile(testStatistics)
            logger.error { ex.message + System.lineSeparator() }
            throw ex
        }

        val maxSmtTestPath = File(samplePath.toString().removeSuffix(extension) + "maxsmt").toPath()
        val maxSmtTestInfo = parseMaxSMTTestInfo(maxSmtTestPath)

        val optimalWeight = maxSmtTestInfo.satSoftConstraintsWeightsSum
        testStatistics.optimalWeight = optimalWeight

        val softConstraintsSize = maxSmtTestInfo.softConstraintsWeights.size

        val softExpressions =
            convertedAssertions.subList(convertedAssertions.size - softConstraintsSize, convertedAssertions.size)
        val hardExpressions = convertedAssertions.subList(0, convertedAssertions.size - softConstraintsSize)

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
                maxSMTResult = maxSMTSolver.checkMaxSMT(500.milliseconds, true)
            } catch (ex: Exception) {
                testStatistics.maxSMTCallStatistics = maxSMTSolver.collectMaxSMTStatistics()
                testStatistics.exceptionMessage = ex.message.toString()
                jsonHelper.appendTestStatisticsToFile(testStatistics)
                logger.error { ex.message + System.lineSeparator() }
                throw ex
            }
        }
        testStatistics.elapsedTimeMs = elapsedTime
        testStatistics.timeoutExceededOrUnknown = maxSMTResult.timeoutExceededOrUnknown

        val foundSoFarWeight = maxSMTResult.satSoftConstraints.sumOf { it.weight.toULong() }
        testStatistics.foundSoFarWeight = foundSoFarWeight

        testStatistics.maxSMTCallStatistics = maxSMTSolver.collectMaxSMTStatistics()

        logger.info { "Elapsed time: $elapsedTime ms --- SubOpt MaxSMT call${System.lineSeparator()}" }

        try {
            assertEquals(SAT, maxSMTResult.hardConstraintsSatStatus, "Hard constraints must be SAT")

            if (foundSoFarWeight > optimalWeight) {
                testStatistics.checkedSoftConstraintsSumIsWrong = true
            }
            assertTrue(
                optimalWeight >= foundSoFarWeight,
                "Soft constraints weights sum was [$foundSoFarWeight], " +
                        "but must be no more than [${optimalWeight}]",
            )

            testStatistics.passed = true
        } catch (ex: Exception) {
            logger.error { ex.message + System.lineSeparator() }
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
            logger.error { ex.message + System.lineSeparator() }
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
            logger.error { ex.message + System.lineSeparator() }
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
        val TEST_WORKER_SINGLE_OPERATION_TIMEOUT = 45.seconds

        internal lateinit var testWorkers: KsmtWorkerPool<TestProtocolModel>
        private lateinit var jsonHelper: JsonStatisticsHelper
        private lateinit var solverManager: KPortfolioSolverManager

        @BeforeAll
        @JvmStatic
        fun initWorkerPools() {
            testWorkers = KsmtWorkerPool(
                maxWorkerPoolSize = 1,
                workerProcessIdleTimeout = 10.minutes,
                workerFactory = object : KsmtWorkerFactory<TestProtocolModel> {
                    override val childProcessEntrypoint = TestWorkerProcess::class
                    override fun updateArgs(args: KsmtWorkerArgs): KsmtWorkerArgs = args
                    override fun mkWorker(id: Int, process: RdServer) = TestWorker(id, process)
                },
            )
        }

        @AfterAll
        @JvmStatic
        fun closeWorkerPools() {
            testWorkers.terminate()
        }

        @BeforeAll
        @JvmStatic
        fun initSolverManager() {
            solverManager = KPortfolioSolverManager(
                listOf(
                    KZ3Solver::class, KBitwuzlaSolver::class, KYicesSolver::class
                ), 2
            )
        }

        @AfterAll
        @JvmStatic
        fun closeSolverManager() {
            solverManager.close()
        }

        @BeforeAll
        @JvmStatic
        fun initJsonHelper() {
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
        fun closeJsonHelper() {
            jsonHelper.markLastTestStatisticsAsProcessed()
        }
    }
}