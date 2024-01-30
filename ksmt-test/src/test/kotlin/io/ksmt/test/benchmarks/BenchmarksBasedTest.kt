package io.ksmt.test.benchmarks

import com.jetbrains.rd.util.reactive.RdFault
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.Assumptions
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.params.provider.Arguments
import io.ksmt.KContext
import io.ksmt.expr.KApp
import io.ksmt.expr.KDivArithExpr
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFpMaxExpr
import io.ksmt.expr.KFpMinExpr
import io.ksmt.expr.KFpToBvExpr
import io.ksmt.expr.KFpToRealExpr
import io.ksmt.expr.KFpValue
import io.ksmt.expr.KFunctionAsArray
import io.ksmt.expr.KModIntExpr
import io.ksmt.expr.KPowerArithExpr
import io.ksmt.expr.KRemIntExpr
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.expr.transformer.KTransformer
import io.ksmt.runner.core.KsmtWorkerArgs
import io.ksmt.runner.core.KsmtWorkerFactory
import io.ksmt.runner.core.KsmtWorkerPool
import io.ksmt.runner.core.RdServer
import io.ksmt.runner.core.WorkerInitializationFailedException
import io.ksmt.runner.generated.models.TestProtocolModel
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverException
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.KSolverUnsupportedFeatureException
import io.ksmt.solver.async.KAsyncSolver
import io.ksmt.solver.runner.KSolverRunnerManager
import io.ksmt.sort.KArithSort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KSort
import io.ksmt.test.SmtLibParseError
import io.ksmt.test.TestRunner
import io.ksmt.test.TestWorker
import io.ksmt.test.TestWorkerProcess
import io.ksmt.utils.FpUtils.isZero
import java.nio.file.Path
import java.nio.file.Paths
import kotlin.io.path.Path
import kotlin.io.path.listDirectoryEntries
import kotlin.io.path.relativeTo
import kotlin.reflect.KClass
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

@Suppress("UnnecessaryAbstractClass")
abstract class BenchmarksBasedTest {

    fun testConverter(
        name: String,
        samplePath: Path,
        mkKsmtAssertions: suspend TestRunner.(List<KExpr<KBoolSort>>) -> List<KExpr<KBoolSort>>
    ) = handleIgnoredTests("testConverter[$name]") {
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
    ) = testModelConversion(name, samplePath) { ctx ->
        solverManager.createSolver(ctx, solverType)
    }

    fun <C : KSolverConfiguration> testModelConversion(
        name: String,
        samplePath: Path,
        solverProvider: (KContext) -> KAsyncSolver<C>
    ) = handleIgnoredTests("testModelConversion[$name]") {
        ignoreNoTestDataStub(name)
        val ctx = KContext()
        testWorkers.withWorker(ctx) { worker ->
            worker.skipBadTestCases {
                val assertions = worker.parseFile(samplePath)
                val ksmtAssertions = worker.convertAssertions(assertions)

                val model = solverProvider(ctx).use { testSolver ->
                    ksmtAssertions.forEach { testSolver.assertAsync(it) }

                    val status = testSolver.checkAsync(SOLVER_CHECK_SAT_TIMEOUT)
                    if (status != KSolverStatus.SAT) {
                        ignoreTest { "No model to check" }
                    }

                    testSolver.modelAsync()
                }

                checkAsArrayDeclsPresentInModel(ctx, model)

                val evaluatedAssertions = ksmtAssertions.map { model.eval(it, isComplete = true) }

                val cardinalityConstraints = model.uninterpretedSorts.mapNotNull { sort ->
                    model.uninterpretedSortUniverse(sort)?.takeIf { it.isNotEmpty() }?.let { universe ->
                        with(ctx) {
                            val x = mkFreshConst("x", sort)
                            val variants = mkOr(universe.map { x eq it })
                            mkUniversalQuantifier(variants, listOf(x.decl))
                        }
                    }
                }

                /**
                 * Evaluated assertion may contain some underspecified
                 * operations (e.g division by zero). Currently used test oracle (Z3)
                 * can define any interpretation for the underspecified operations
                 * and therefore (not (= a true)) check will always be SAT.
                 * We consider such cases as false positives.
                 * */
                val assertionsToCheck = evaluatedAssertions.filterNot { hasUnderspecifiedOperations(it) }

                worker.performEqualityChecks {
                    cardinalityConstraints.forEach { assume(it) }
                    assertionsToCheck.forEach { isTrue(it) }
                    check { "assertions are not true in model" }
                }
            }
        }
    }

    fun <C : KSolverConfiguration> testSolver(
        name: String,
        samplePath: Path,
        solverType: KClass<out KSolver<C>>
    ) = testSolver(name, samplePath) { ctx ->
        solverManager.createSolver(ctx, solverType)
    }

    fun <C : KSolverConfiguration> testSolver(
        name: String,
        samplePath: Path,
        solverProvider: (KContext) -> KAsyncSolver<C>
    ) = handleIgnoredTests("testSolver[$name]") {
        ignoreNoTestDataStub(name)
        val ctx = KContext()
        testWorkers.withWorker(ctx) { worker ->
            worker.skipBadTestCases {
                val assertions = worker.parseFile(samplePath)
                val solver = worker.createSolver(TEST_WORKER_CHECK_SAT_TIMEOUT)
                assertions.forEach {
                    worker.assert(solver, it)
                }

                val expectedStatus = worker.check(solver)

                if (expectedStatus == KSolverStatus.UNKNOWN) {
                    ignoreTest { "Expected status is unknown" }
                }

                val ksmtAssertions = worker.convertAssertions(assertions)

                val actualStatus = solverProvider(ctx).use { ksmtSolver ->
                    ksmtAssertions.forEach { ksmtSolver.assertAsync(it) }

                    // use greater timeout to reduce false-positive unknowns
                    val status = ksmtSolver.check(SOLVER_CHECK_SAT_TIMEOUT)
                    if (status == KSolverStatus.UNKNOWN) {
                        ignoreTest { "Actual status is unknown: ${ksmtSolver.reasonOfUnknown()}" }
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

    /**
     * Check if expression contains underspecified operations:
     * 1. division by zero
     * 2. integer mod/rem with zero divisor
     * 3. zero to the zero power
     * 4. Fp to number conversions with NaN and Inf
     * 5. Fp max/min with +/- zero
     * */
    private fun hasUnderspecifiedOperations(expr: KExpr<*>): Boolean {
        val detector = UnderspecifiedOperationDetector(expr.ctx)
        detector.apply(expr)
        return detector.hasUnderspecifiedOperation
    }

    private class UnderspecifiedOperationDetector(ctx: KContext) : KNonRecursiveTransformer(ctx) {
        var hasUnderspecifiedOperation = false

        override fun <T : KArithSort> transform(expr: KDivArithExpr<T>): KExpr<T> =
            super.transform(expr).also { checkDivisionByZero(expr.rhs) }

        override fun transform(expr: KModIntExpr): KExpr<KIntSort> =
            super.transform(expr).also { checkDivisionByZero(expr.rhs) }

        override fun transform(expr: KRemIntExpr): KExpr<KIntSort> =
            super.transform(expr).also { checkDivisionByZero(expr.rhs) }

        override fun <T : KArithSort> transform(expr: KPowerArithExpr<T>): KExpr<T> =
            super.transform(expr).also { checkZeroToZeroPower(expr.lhs, expr.rhs) }

        override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort> =
            super.transform(expr).also { checkFpNaNOrInf(expr.value) }

        override fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort>  =
            super.transform(expr).also { checkFpNaNOrInf(expr.value) }

        override fun <T : KFpSort> transform(expr: KFpMinExpr<T>): KExpr<T> =
            super.transform(expr).also { checkFpZeroWithDifferentSign(expr.arg0, expr.arg1) }

        override fun <T : KFpSort> transform(expr: KFpMaxExpr<T>): KExpr<T> =
            super.transform(expr).also { checkFpZeroWithDifferentSign(expr.arg0, expr.arg1) }

        private fun checkDivisionByZero(divisor: KExpr<*>) = with(ctx) {
            if (divisor == 0.expr) {
                hasUnderspecifiedOperation = true
            }
        }

        private fun checkZeroToZeroPower(base: KExpr<*>, power: KExpr<*>) = with(ctx) {
            if (base == 0.expr && power == 0.expr) {
                hasUnderspecifiedOperation = true
            }
        }

        private fun <T: KFpSort> checkFpNaNOrInf(value: KExpr<T>) = with(ctx) {
            val underspecifiedValues = setOf(
                mkFpNaN(value.sort),
                mkFpInf(signBit = true, value.sort),
                mkFpInf(signBit = false, value.sort),
            )
            if (value in underspecifiedValues) {
                hasUnderspecifiedOperation = true
            }
        }

        private fun <T : KFpSort> checkFpZeroWithDifferentSign(lhs: KExpr<T>, rhs: KExpr<T>) {
            if (lhs !is KFpValue<T> || rhs !is KFpValue<T>) return
            if (lhs.isZero() && rhs.isZero() && lhs.signBit != rhs.signBit) {
                hasUnderspecifiedOperation = true
            }
        }
    }

    companion object {
        val TEST_WORKER_SINGLE_OPERATION_TIMEOUT = 10.seconds
        val TEST_WORKER_EQUALITY_CHECK_TIMEOUT = 3.seconds
        val TEST_WORKER_CHECK_SAT_TIMEOUT = 1.seconds

        val SOLVER_SINGLE_OPERATION_TIMEOUT = 10.seconds
        val SOLVER_CHECK_SAT_TIMEOUT = 3.seconds

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

        private fun prepareTestData(filterCondition: (String) -> Boolean): List<BenchmarkTestArguments> {
            val testDataLocation = testDataLocation()
            return testDataLocation
                .listDirectoryEntries("*.smt2")
                .asSequence()
                .filter {
                    val name = it.relativeTo(testDataLocation).toString()
                    filterCondition(name)
                }
                .sorted()
                .drop(testDataChunk * testDataChunkSize)
                .take(testDataChunkSize)
                .map { BenchmarkTestArguments(it.relativeTo(testDataLocation).toString(), it) }
                .toList()
                .skipBadTestCases()
                .ensureNotEmpty()
        }

        fun testData(filterCondition: (String) -> Boolean = { true }) = prepareTestData(filterCondition)

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
                hardTimeout = SOLVER_SINGLE_OPERATION_TIMEOUT,
                workerProcessIdleTimeout = 10.minutes
            )
            testWorkers = KsmtWorkerPool(
                maxWorkerPoolSize = 4,
                workerProcessIdleTimeout = 10.minutes,
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

        // See [handleIgnoredTests]
        inline fun ignoreTest(message: () -> String?): Nothing {
            throw IgnoreTestException(message())
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
        override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KFunctionAsArray<A, R>): KExpr<A> {
            assertNotNull(model.interpretation(expr.function), "no interpretation for as-array: $expr")
            return expr
        }
    }

    class SortChecker(ctx: KContext) : KNonRecursiveTransformer(ctx) {
        override fun <T : KSort, A : KSort> transformApp(expr: KApp<T, A>): KExpr<T> = with(ctx) {
            // apply internally check arguments sorts
            expr.decl.apply(expr.args)
            return super.transformApp(expr).also {
                check(it.sort == expr.sort) { "sort mismatch" }
            }
        }
    }

    suspend fun TestRunner.performEqualityChecks(checks: suspend EqualityChecker.() -> Unit) {
        val solver = createSolver(TEST_WORKER_EQUALITY_CHECK_TIMEOUT)
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

                KSolverStatus.UNKNOWN -> ignoreTest {
                    "equality check: unknown -- ${worker.getReasonUnknown(solver)}"
                }
            }
        }
    }

    inline fun TestRunner.skipBadTestCases(body: () -> Unit) = try {
        skipUnsupportedSolverFeatures {
            body()
        }
    } catch (ex: SmtLibParseError) {
        ignoreTest { "parse failed -- ${ex.message}" }
    } catch (ex: TimeoutCancellationException) {
        ignoreTest { "timeout -- ${ex.message}" }
    }

    inline fun TestRunner.skipUnsupportedSolverFeatures(body: () -> Unit) = try {
        handleWrappedSolverException {
            body()
        }
    } catch (ex: NotImplementedError) {
        // skip test with not implemented feature
        ignoreTest {
            val reducedStackTrace = ex.stackTrace.take(5).joinToString("\n") { it.toString() }
            "${ex.message}\n$reducedStackTrace"
        }
    } catch (ex: KSolverUnsupportedFeatureException) {
        ignoreTest { ex.message }
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

    class IgnoreTestException(message: String?) : Exception(message)

    /**
     * When a test is ignored via JUnit assumption the reason (message)
     * of the ignore is not shown in the test report.
     * To keep some insight on the ignore reasons we use
     * logging to stderr, since it is present in test reports.
     * */
    fun handleIgnoredTests(testName: String, testBody: () -> Unit) {
        try {
            testBody()
        } catch (ignore: IgnoreTestException) {
            val testClassName = javaClass.canonicalName
            System.err.println("IGNORE $testClassName.$testName: ${ignore.message}")
            Assumptions.assumeTrue(false)
        }
    }
}
