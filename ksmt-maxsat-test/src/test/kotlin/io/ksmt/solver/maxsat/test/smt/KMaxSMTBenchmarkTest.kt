package io.ksmt.solver.maxsat.test.smt

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.runner.core.KsmtWorkerArgs
import io.ksmt.runner.core.KsmtWorkerFactory
import io.ksmt.runner.core.KsmtWorkerPool
import io.ksmt.runner.core.RdServer
import io.ksmt.runner.core.WorkerInitializationFailedException
import io.ksmt.runner.generated.models.TestProtocolModel
import io.ksmt.solver.KSolverStatus.SAT
import io.ksmt.solver.maxsat.solvers.KMaxSATSolver
import io.ksmt.solver.maxsat.test.KMaxSMTBenchmarkBasedTest
import io.ksmt.solver.maxsat.test.parseMaxSMTTestInfo
import io.ksmt.solver.z3.KZ3SolverConfiguration
import io.ksmt.sort.KBoolSort
import io.ksmt.test.TestRunner
import io.ksmt.test.TestWorker
import io.ksmt.test.TestWorkerProcess
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.BeforeEach
import java.io.File
import java.nio.file.Path
import kotlin.io.path.extension
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

abstract class KMaxSMTBenchmarkTest : KMaxSMTBenchmarkBasedTest {
    abstract fun getSolver(): KMaxSATSolver<KZ3SolverConfiguration>

    protected val ctx: KContext = KContext()
    private lateinit var maxSATSolver: KMaxSATSolver<KZ3SolverConfiguration>

    @BeforeEach
    fun initSolver() {
        maxSATSolver = getSolver()
    }

    @AfterEach
    fun closeSolver() = maxSATSolver.close()

    fun maxSMTTest(
        name: String,
        samplePath: Path,
        mkKsmtAssertions: suspend TestRunner.(List<KExpr<KBoolSort>>) -> List<KExpr<KBoolSort>>,
    ) {
        val extension = "smt2"
        require(samplePath.extension == extension) {
            "File extension cannot be '${samplePath.extension}' as it must be $extension"
        }

        lateinit var ksmtAssertions: List<KExpr<KBoolSort>>

        testWorkers.withWorker(ctx) { worker ->
            val assertions = worker.parseFile(samplePath)
            val convertedAssertions = worker.convertAssertions(assertions)
            ksmtAssertions = worker.mkKsmtAssertions(convertedAssertions)
        }

        val maxSmtTestPath = File(samplePath.toString().removeSuffix(extension) + "maxsmt").toPath()
        val maxSmtTestInfo = parseMaxSMTTestInfo(maxSmtTestPath)

        val softConstraintsSize = maxSmtTestInfo.softConstraintsWeights.size

        val softExpressions =
            ksmtAssertions.subList(ksmtAssertions.size - softConstraintsSize, ksmtAssertions.size)
        val hardExpressions = ksmtAssertions.subList(0, ksmtAssertions.size - softConstraintsSize)

        hardExpressions.forEach {
            maxSATSolver.assert(it)
        }

        maxSmtTestInfo.softConstraintsWeights
            .zip(softExpressions)
            .forEach { (weight, expr) ->
                maxSATSolver.assertSoft(expr, weight)
            }

        val maxSATResult = maxSATSolver.checkMaxSAT(60.seconds)
        val satSoftConstraintsWeightsSum = maxSATResult.satSoftConstraints.sumOf { it.weight }

        assertEquals(SAT, maxSATResult.hardConstraintsSATStatus)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertEquals(
            maxSmtTestInfo.satSoftConstraintsWeightsSum,
            satSoftConstraintsWeightsSum.toULong(),
        )
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
        }

        @AfterAll
        @JvmStatic
        fun closeWorkerPools() = testWorkers.terminate()
    }
}
