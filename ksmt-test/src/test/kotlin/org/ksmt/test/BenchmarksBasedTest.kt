package org.ksmt.test

import com.jetbrains.rd.util.reactive.RdFault
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.Assumptions
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.params.provider.Arguments
import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.expr.transformer.KTransformer
import org.ksmt.runner.core.KsmtWorkerArgs
import org.ksmt.runner.core.KsmtWorkerFactory
import org.ksmt.runner.core.KsmtWorkerPool
import org.ksmt.runner.core.RdServer
import org.ksmt.runner.core.WorkerInitializationFailedException
import org.ksmt.runner.models.generated.TestProtocolModel
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.KSolverException
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.KSolverUnsupportedFeatureException
import org.ksmt.solver.runner.KSolverRunnerManager
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import java.nio.file.Path
import java.nio.file.Paths
import kotlin.io.path.Path
import kotlin.io.path.listDirectoryEntries
import kotlin.io.path.relativeTo
import kotlin.reflect.KClass
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds

@Suppress("UnnecessaryAbstractClass")
abstract class BenchmarksBasedTest {

    fun testConverter(
        name: String,
        samplePath: Path,
        mkKsmtAssertions: suspend TestRunner.(List<KExpr<KBoolSort>>) -> List<KExpr<KBoolSort>>
    ) {
        ignoreNoTestDataStub(name)
        val ctx = KContext()
        testWorkers.withWorker(ctx) { worker ->
            worker.skipBadTestCases {
                val assertions = worker.parseFile(samplePath)
                val convertedAssertions = worker.convertAssertions(assertions)

                val ksmtAssertions = worker.mkKsmtAssertions(convertedAssertions)

                ksmtAssertions.forEach { SortChecker(ctx).apply(it) }

                worker.performEqualityChecks {
                    for ((originalZ3Expr, ksmtExpr) in assertions.zip(ksmtAssertions)) {
                        areEqual(actual = ksmtExpr, expected = originalZ3Expr)
                    }
                    check { "expressions are not equal" }
                }
            }
        }
    }

    fun <C : KSolverConfiguration> testModelConversion(
        name: String,
        samplePath: Path,
        solverType: KClass<out KSolver<C>>
    ) {
        ignoreNoTestDataStub(name)
        val ctx = KContext()
        testWorkers.withWorker(ctx) { worker ->
            worker.skipBadTestCases {
                val assertions = worker.parseFile(samplePath)
                val ksmtAssertions = worker.convertAssertions(assertions)

                val model = solverManager.createSolver(ctx, solverType).use { testSolver ->
                    ksmtAssertions.forEach { testSolver.assertAsync(it) }

                    val status = testSolver.checkAsync(timeout = 1.seconds)
                    Assumptions.assumeTrue(status == KSolverStatus.SAT, "No model to check")

                    testSolver.modelAsync()
                }

                checkAsArrayDeclsPresentInModel(ctx, model)

                val evaluatedAssertions = ksmtAssertions.map { model.eval(it, isComplete = false) }

                val cardinalityConstraints = model.uninterpretedSorts.mapNotNull { sort ->
                    model.uninterpretedSortUniverse(sort)?.let { sort to it }
                }.map { (sort, universe) ->
                    with(ctx) {
                        val x = mkFreshConst("x", sort)
                        val variants = mkOr(universe.map { x eq it })
                        val uniqueness = mkDistinct(universe.toList())
                        mkUniversalQuantifier(variants and uniqueness, listOf(x.decl))
                    }
                }

                worker.performEqualityChecks {
                    cardinalityConstraints.forEach { assume(it) }
                    evaluatedAssertions.forEach { isTrue(it) }
                    check { "assertions are not true in model" }
                }
            }
        }
    }

    fun <C : KSolverConfiguration> testSolver(
        name: String,
        samplePath: Path,
        solverType: KClass<out KSolver<C>>
    ) {
        ignoreNoTestDataStub(name)
        val ctx = KContext()
        testWorkers.withWorker(ctx) { worker ->
            worker.skipBadTestCases {
                val assertions = worker.parseFile(samplePath)
                val solver = worker.createSolver()
                assertions.forEach {
                    worker.assert(solver, it)
                }

                val expectedStatus = worker.check(solver)
                Assumptions.assumeTrue(expectedStatus != KSolverStatus.UNKNOWN) {
                    "expected status: unknown"
                }

                val ksmtAssertions = worker.convertAssertions(assertions)

                val actualStatus = solverManager.createSolver(ctx, solverType).use { ksmtSolver ->
                    ksmtAssertions.forEach { ksmtSolver.assertAsync(it) }

                    // use greater timeout to reduce false-positive unknowns
                    val status = ksmtSolver.check(timeout = 2.seconds)
                    Assumptions.assumeTrue(status != KSolverStatus.UNKNOWN) {
                        "Actual status is unknown: ${ksmtSolver.reasonOfUnknown()}"
                    }

                    status
                }
                assertEquals(expectedStatus, actualStatus, "solver check-sat mismatch")
            }
        }
    }

    internal fun KsmtWorkerPool<TestProtocolModel>.withWorker(
        ctx: KContext,
        body: suspend (TestRunner) -> Unit
    ) = runBlocking {
        val worker = try {
            getOrCreateFreeWorker()
        } catch (ex: WorkerInitializationFailedException) {
            val testIgnoreReason = "worker initialization failed -- ${ex.message}"
            System.err.println(testIgnoreReason)
            Assumptions.assumeTrue(false, testIgnoreReason)
            error("ignored")
        }
        worker.astSerializationCtx.initCtx(ctx)
        worker.lifetime.onTermination {
            worker.astSerializationCtx.resetCtx()
        }
        try {
            TestRunner(ctx, worker).let {
                try {
                    it.init()
                    body(it)
                } finally {
                    it.delete()
                }
            }
        } catch (ex: TimeoutCancellationException) {
            val testIgnoreReason = "worker timeout -- ${ex.message}"
            System.err.println(testIgnoreReason)
            Assumptions.assumeTrue(false, testIgnoreReason)
        } finally {
            worker.release()
        }
    }

    companion object {
        internal lateinit var solverManager: KSolverRunnerManager
        internal lateinit var testWorkers: KsmtWorkerPool<TestProtocolModel>

        private val testDataChunkSize = System.getenv("benchmarkChunkMaxSize")?.toIntOrNull() ?: Int.MAX_VALUE
        private val testDataChunk = System.getenv("benchmarkChunk")?.toIntOrNull() ?: 0

        private val NO_TEST_DATA = BenchmarkTestArguments("__NO__TEST__DATA__", Path("."))

        private fun testDataLocation(): Path = this::class.java.classLoader
            .getResource("testData")
            ?.toURI()
            ?.let { Paths.get(it) }
            ?: error("No test data")

        private fun prepareTestData(): List<BenchmarkTestArguments> {
            val testDataLocation = testDataLocation()
            return testDataLocation
                .listDirectoryEntries("*.smt2")
                .sorted()
                .drop(testDataChunk * testDataChunkSize)
                .take(testDataChunkSize)
                .map { BenchmarkTestArguments(it.relativeTo(testDataLocation).toString(), it) }
                .skipBadTestCases()
                .ensureNotEmpty()
        }

        val testData by lazy {
            prepareTestData()
        }

        /**
         * Parametrized tests require at least one argument.
         * In some cases, we may filter out all provided test samples,
         * which will cause JUnit failure. To overcome this problem,
         * we use [NO_TEST_DATA] stub, which is handled
         * by [ignoreNoTestDataStub] and results in a single ignored test.
         * */
        fun List<BenchmarkTestArguments>.ensureNotEmpty() = ifEmpty { listOf(NO_TEST_DATA) }

        fun ignoreNoTestDataStub(name: String) {
            Assumptions.assumeTrue(name != NO_TEST_DATA.name)
        }

        private fun List<BenchmarkTestArguments>.skipBadTestCases(): List<BenchmarkTestArguments> =
            /**
             * Contains a declaration with an empty name.
             * Normally, such declarations have special <null> name in Z3,
             * but in this case it is not true. After internalization via API,
             * resulting declaration has <null> name as excepted.
             * Therefore, declarations are not equal, but this is not our issue.
             * */
            filterNot { it.name == "QF_BV_symbols.smt2" }

        @BeforeAll
        @JvmStatic
        fun initWorkerPools() {
            solverManager = KSolverRunnerManager(
                workerPoolSize = 4,
                hardTimeout = 3.seconds,
                workerProcessIdleTimeout = 50.seconds
            )
            testWorkers = KsmtWorkerPool(
                maxWorkerPoolSize = 4,
                workerProcessIdleTimeout = 300.seconds,
                workerFactory = object : KsmtWorkerFactory<TestProtocolModel> {
                    override val childProcessEntrypoint = TestWorkerProcess::class
                    override fun updateArgs(args: KsmtWorkerArgs): KsmtWorkerArgs = args
                    override fun mkWorker(id: Int, process: RdServer) = TestWorker(id, process)
                }
            )
        }

        @AfterAll
        @JvmStatic
        fun closeWorkerPools() {
            solverManager.close()
            testWorkers.terminate()
        }
    }

    data class BenchmarkTestArguments(
        val name: String,
        val samplePath: Path
    ) : Arguments {
        override fun get() = arrayOf(name, samplePath)
    }

    fun checkAsArrayDeclsPresentInModel(ctx: KContext, model: KModel) {
        val checker = AsArrayDeclChecker(ctx, model)
        model.declarations.forEach { decl ->
            model.interpretation(decl)?.let { interpretation ->
                interpretation.entries.forEach { it.value.accept(checker) }
                interpretation.default?.accept(checker)
            }
        }
    }

    class AsArrayDeclChecker(override val ctx: KContext, val model: KModel) : KTransformer {
        override fun <D : KSort, R : KSort> transform(expr: KFunctionAsArray<D, R>): KExpr<KArraySort<D, R>> {
            assertNotNull(model.interpretation(expr.function), "no interpretation for as-array: $expr")
            return expr
        }
    }

    class SortChecker(ctx: KContext) : KNonRecursiveTransformer(ctx) {
        override fun <T : KSort> transformApp(expr: KApp<T, *>): KExpr<T> = with(ctx) {
            // apply internally check arguments sorts
            expr.decl.apply(expr.args)
            return super.transformApp(expr).also {
                check(it.sort == expr.sort) { "sort mismatch" }
            }
        }
    }

    suspend fun TestRunner.performEqualityChecks(checks: suspend EqualityChecker.() -> Unit) {
        val solver = createSolver()
        val checker = EqualityChecker(this, solver)
        checker.checks()
    }

    private data class EqualityCheck(val actual: KExpr<*>, val expected: Long)

    class EqualityChecker(
        private val worker: TestRunner,
        private val solver: Int,
    ) {
        private val equalityChecks = mutableListOf<EqualityCheck>()
        private val workerTrueExpr: Long by lazy { runBlocking { worker.mkTrueExpr() } }

        suspend fun areEqual(actual: KExpr<*>, expected: Long) {
            worker.addEqualityCheck(solver, actual, expected)
            equalityChecks += EqualityCheck(actual, expected)
        }

        suspend fun isTrue(actual: KExpr<*>) = areEqual(actual, workerTrueExpr)

        suspend fun assume(expr: KExpr<KBoolSort>) {
            worker.addEqualityCheckAssumption(solver, expr)
        }

        suspend fun check(message: () -> String) {
            val status = worker.checkEqualities(solver)
            when (status) {
                KSolverStatus.UNSAT -> return
                KSolverStatus.SAT -> {
                    val failedEqualityCheck = worker.findFirstFailedEquality(solver)?.let { equalityChecks[it] }
                    if (failedEqualityCheck != null) {
                        val expected = worker.exprToString(failedEqualityCheck.expected)
                        assertEquals(expected, "${failedEqualityCheck.actual}", message())
                    }
                    assertTrue(false, message())
                }

                KSolverStatus.UNKNOWN -> {
                    val testIgnoreReason = "equality check: unknown -- ${worker.getReasonUnknown(solver)}"
                    System.err.println(testIgnoreReason)
                    Assumptions.assumeTrue(false, testIgnoreReason)
                }
            }
        }
    }

    inline fun TestRunner.skipBadTestCases(body: () -> Unit) = try {
        skipUnsupportedSolverFeatures {
            body()
        }
    } catch (ex: SmtLibParseError) {
        val testIgnoreReason = "parse failed -- ${ex.message}"
        System.err.println(testIgnoreReason)
        Assumptions.assumeTrue(false, testIgnoreReason)
    } catch (ex: TimeoutCancellationException) {
        val testIgnoreReason = "timeout -- ${ex.message}"
        System.err.println(testIgnoreReason)
        Assumptions.assumeTrue(false, testIgnoreReason)
    }

    inline fun TestRunner.skipUnsupportedSolverFeatures(body: () -> Unit) = try {
        handleWrappedSolverException {
            body()
        }
    } catch (ex: NotImplementedError) {
        val reducedStackTrace = ex.stackTrace.take(5).joinToString("\n") { it.toString() }
        val report = "${ex.message}\n$reducedStackTrace"

        System.err.println(report)
        // skip test with not implemented feature

        Assumptions.assumeTrue(false, ex.message)
    } catch (ex: KSolverUnsupportedFeatureException) {
        Assumptions.assumeTrue(false, ex.message)
    }

    inline fun <reified T> TestRunner.handleWrappedSolverException(body: () -> T): T = try {
        body()
    } catch (ex: KSolverException) {
        val unwrappedException = when (val cause = ex.cause) {
            is RdFault -> rdExceptionCause(cause) ?: ex
            is TimeoutCancellationException -> cause
            else -> ex
        }
        throw unwrappedException
    }

}
