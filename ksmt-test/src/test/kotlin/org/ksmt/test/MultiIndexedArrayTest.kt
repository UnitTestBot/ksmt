package org.ksmt.test

import com.microsoft.z3.Context
import io.github.cvc5.Solver
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable
import org.ksmt.KContext
import org.ksmt.KContext.SimplificationMode.NO_SIMPLIFY
import org.ksmt.expr.KExpr
import org.ksmt.expr.rewrite.KExprUninterpretedDeclCollector
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.KSolverUnsupportedFeatureException
import org.ksmt.solver.bitwuzla.KBitwuzlaContext
import org.ksmt.solver.bitwuzla.KBitwuzlaExprConverter
import org.ksmt.solver.bitwuzla.KBitwuzlaExprInternalizer
import org.ksmt.solver.bitwuzla.KBitwuzlaSolver
import org.ksmt.solver.cvc5.KCvc5Context
import org.ksmt.solver.cvc5.KCvc5ExprConverter
import org.ksmt.solver.cvc5.KCvc5ExprInternalizer
import org.ksmt.solver.cvc5.KCvc5Solver
import org.ksmt.solver.runner.KSolverRunnerManager
import org.ksmt.solver.yices.KYicesContext
import org.ksmt.solver.yices.KYicesExprConverter
import org.ksmt.solver.yices.KYicesExprInternalizer
import org.ksmt.solver.yices.KYicesSolver
import org.ksmt.solver.z3.KZ3Context
import org.ksmt.solver.z3.KZ3ExprConverter
import org.ksmt.solver.z3.KZ3ExprInternalizer
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KBv8Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KSort
import org.ksmt.utils.uncheckedCast
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

@EnabledIfEnvironmentVariable(
    named = "runMultiIndexedArrayTest",
    matches = "true",
    disabledReason = "Disable due to a very long runtime (about 6 hours) which is not applicable for usual CI runs"
)
class MultiIndexedArrayTest {

    @Test
    fun testMultiIndexedArraysZ3WithZ3Oracle(): Unit = with(KContext(simplificationMode = NO_SIMPLIFY)) {
        oracleManager.createSolver(this, KZ3Solver::class).use { oracleSolver ->
            mkZ3Context(this).use { z3NativeCtx ->
                runMultiIndexedArraySamples(oracleSolver) { expr ->
                    internalizeAndConvertZ3(z3NativeCtx, expr)
                }
            }
        }
    }

    @Test
    fun testMultiIndexedArraysBitwuzlaWithZ3Oracle(): Unit = with(KContext(simplificationMode = NO_SIMPLIFY)) {
        oracleManager.createSolver(this, KZ3Solver::class).use { oracleSolver ->
            KBitwuzlaContext(this).use { bitwuzlaNativeCtx ->
                runMultiIndexedArraySamples(oracleSolver) { expr ->
                    internalizeAndConvertBitwuzla(bitwuzlaNativeCtx, expr)
                }
            }
        }
    }

    @Test
    fun testMultiIndexedArraysZ3WithBitwuzlaOracle(): Unit = with(KContext(simplificationMode = NO_SIMPLIFY)) {
        oracleManager.createSolver(this, KBitwuzlaSolver::class).use { oracleSolver ->
            mkZ3Context(this).use { z3NativeCtx ->
                runMultiIndexedArraySamples(oracleSolver) { expr ->
                    internalizeAndConvertZ3(z3NativeCtx, expr)
                }
            }
        }
    }

    @Test
    fun testMultiIndexedArraysBitwuzlaWithBitwuzlaOracle(): Unit = with(KContext(simplificationMode = NO_SIMPLIFY)) {
        oracleManager.createSolver(this, KBitwuzlaSolver::class).use { oracleSolver ->
            KBitwuzlaContext(this).use { bitwuzlaNativeCtx ->
                runMultiIndexedArraySamples(oracleSolver) { expr ->
                    internalizeAndConvertBitwuzla(bitwuzlaNativeCtx, expr)
                }
            }
        }
    }

    @Test
    fun testMultiIndexedArraysYicesWithZ3Oracle(): Unit = with(KContext(simplificationMode = NO_SIMPLIFY)) {
        oracleManager.createSolver(this, KZ3Solver::class).use { oracleSolver ->
            KYicesContext().use { yicesNativeCtx ->
                runMultiIndexedArraySamples(oracleSolver) { expr ->
                    internalizeAndConvertYices(yicesNativeCtx, expr)
                }
            }
        }
    }

    @Test
    fun testMultiIndexedArraysZ3WithYicesOracle(): Unit = with(KContext(simplificationMode = NO_SIMPLIFY)) {
        oracleManager.createSolver(this, KYicesSolver::class).use { oracleSolver ->
            mkZ3Context(this).use { z3NativeCtx ->
                runMultiIndexedArraySamples(oracleSolver) { expr ->
                    internalizeAndConvertZ3(z3NativeCtx, expr)
                }
            }
        }
    }

    @Test
    fun testMultiIndexedArraysYicesWithYicesOracle(): Unit = with(KContext(simplificationMode = NO_SIMPLIFY)) {
        oracleManager.createSolver(this, KYicesSolver::class).use { oracleSolver ->
            KYicesContext().use { yicesNativeCtx ->
                runMultiIndexedArraySamples(oracleSolver) { expr ->
                    internalizeAndConvertYices(yicesNativeCtx, expr)
                }
            }
        }
    }

    @Test
    fun testMultiIndexedArraysZ3WithCvc5Oracle(): Unit = with(KContext(simplificationMode = NO_SIMPLIFY)) {
        oracleManager.createSolver(this, KCvc5Solver::class).use { oracleSolver ->

            // Enable HO to test array lambda equalities
            oracleSolver.configure { setCvc5Logic("HO_QF_ALL") }

            mkZ3Context(this).use { z3NativeCtx ->
                runMultiIndexedArraySamples(oracleSolver) { expr ->
                    internalizeAndConvertZ3(z3NativeCtx, expr)
                }
            }
        }
    }

    @Test
    fun testMultiIndexedArraysCvc5WithZ3Oracle(): Unit = with(KContext(simplificationMode = NO_SIMPLIFY)) {
        oracleManager.createSolver(this, KZ3Solver::class).use { oracleSolver ->
            mkCvc5Context(this).use { cvc5NativeCtx ->
                runMultiIndexedArraySamples(oracleSolver) { expr ->
                    internalizeAndConvertCvc5(cvc5NativeCtx, expr)
                }
            }
        }
    }

    private inline fun KContext.runMultiIndexedArraySamples(
        oracle: KSolver<*>,
        process: (KExpr<KSort>) -> KExpr<KSort>
    ) {
        val stats = TestStats()
        val sorts = listOf(
            mkArraySort(bv8Sort, bv8Sort),
            mkArraySort(bv32Sort, bv16Sort, bv8Sort),
            mkArraySort(bv32Sort, bv16Sort, bv8Sort, bv8Sort),
            mkArrayNSort(listOf(bv32Sort, bv16Sort, bv8Sort, bv32Sort, bv8Sort), bv8Sort)
        )

        for (sort in sorts) {
            val expressions = mkArrayExpressions(sort)
            for (expr in expressions) {
                stats.start()
                stats.withErrorHandling {
                    val processed = process(expr)
                    assertEquals(stats, oracle, expr, processed)
                }
            }
        }

        stats.result()
    }

    private fun <A : KArraySortBase<KBv8Sort>> KContext.mkArrayExpressions(sort: A): List<KExpr<KSort>> {
        var arrayExpressions = listOfNotNull(
            mkConst(sort),
//            mkAsArray(sort), // disable as-array because it is too hard to check equality
            mkArrayConst(sort) { mkConst("cv", bv8Sort) },
            mkLambda(sort) { mkConst("lv", bv8Sort) }
        )

        arrayExpressions = arrayExpressions + arrayExpressions.map {
            mkStore(it) { mkConst("v", bv8Sort) }
        }

        arrayExpressions = arrayExpressions + arrayExpressions.flatMap { first ->
            arrayExpressions.map { second ->
                mkIte(mkConst("cond", boolSort), first, second)
            }
        }

        val arrayEq = arrayExpressions.crossProduct { first, second -> first eq second }

        var arraySelects = arrayExpressions.map { mkSelect(it) }

        val arrayValues = arraySelects + listOf(mkConst("x", bv8Sort))

        arrayExpressions = arrayExpressions + arrayExpressions.map { array ->
            mkLambda(sort) { indices -> mkSelect(array) { indices.uncheckedCast() } }
        }

        arrayExpressions = arrayExpressions + arrayValues.flatMap { value ->
            listOf(
                mkArrayConst(sort) { value },
                mkLambda(sort) { value },
            )
        }

        arrayExpressions = arrayExpressions + arrayExpressions.flatMap { array ->
            arrayValues.map { value ->
                mkStore(array) { value }
            }
        }

        arraySelects = arraySelects + arrayExpressions.map { mkSelect(it) }

        return listOf(
            arrayExpressions,
            arraySelects,
            arrayEq
        ).flatten().uncheckedCast()
    }

    private inline fun <T, R> List<T>.crossProduct(transform: (T, T) -> R): List<R> {
        val result = mutableListOf<R>()
        for (i in indices) {
            for (j in i until size) {
                result += transform(get(i), get(j))
            }
        }
        return result
    }

    private fun <A : KArraySortBase<KBv8Sort>> KContext.mkConst(sort: A): KExpr<A> =
        mkFreshConst("c", sort)

    private fun <A : KArraySortBase<KBv8Sort>> KContext.mkArrayConst(
        sort: A,
        value: () -> KExpr<KBv8Sort>
    ): KExpr<A> = mkArrayConst(sort, value())

    private fun <A : KArraySortBase<KBv8Sort>> KContext.mkAsArray(sort: A): KExpr<A> {
        val function = mkFreshFuncDecl("f", sort.range, sort.domainSorts)
        return mkFunctionAsArray(sort, function)
    }

    private fun <A : KArraySortBase<KBv8Sort>> KContext.mkLambda(
        sort: A,
        mkBody: (List<KExpr<KBvSort>>) -> KExpr<KBv8Sort>
    ): KExpr<A> {
        val indices = sort.domainSorts.map { mkFreshConst("i", it) }
        val body = mkBody(indices.uncheckedCast())
        return when (indices.size) {
            KArraySort.DOMAIN_SIZE -> mkArrayLambda(indices.single().decl, body)
            KArray2Sort.DOMAIN_SIZE -> mkArrayLambda(indices.first().decl, indices.last().decl, body)
            KArray3Sort.DOMAIN_SIZE -> {
                val (i0, i1, i2) = indices
                mkArrayLambda(i0.decl, i1.decl, i2.decl, body)
            }

            else -> mkArrayNLambda(indices.map { it.decl }, body)
        }.uncheckedCast()
    }

    private fun <A : KArraySortBase<KBv8Sort>> KContext.mkStore(
        array: KExpr<A>,
        mkValue: () -> KExpr<KBv8Sort>
    ): KExpr<A> {
        val indices = array.sort.domainSorts.map { mkFreshConst("i", it) }
        val value = mkValue()
        return when (indices.size) {
            KArraySort.DOMAIN_SIZE -> mkArrayStore(array.uncheckedCast(), indices.single(), value)
            KArray2Sort.DOMAIN_SIZE -> mkArrayStore(array.uncheckedCast(), indices.first(), indices.last(), value)
            KArray3Sort.DOMAIN_SIZE -> {
                val (i0, i1, i2) = indices
                mkArrayStore(array.uncheckedCast(), i0, i1, i2, value)
            }

            else -> mkArrayNStore(array.uncheckedCast(), indices, value)
        }.uncheckedCast()
    }

    private fun <A : KArraySortBase<KBv8Sort>> KContext.mkSelect(
        array: KExpr<A>,
        mkIndices: KContext.(A) -> List<KExpr<KSort>> = { sort ->
            sort.domainSorts.map { mkFreshConst("i", it) }
        }
    ): KExpr<KBv8Sort> {
        val indices = mkIndices(array.sort)
        return when (indices.size) {
            KArraySort.DOMAIN_SIZE -> mkArraySelect(array.uncheckedCast(), indices.single())
            KArray2Sort.DOMAIN_SIZE -> mkArraySelect(array.uncheckedCast(), indices.first(), indices.last())
            KArray3Sort.DOMAIN_SIZE -> {
                val (i0, i1, i2) = indices
                mkArraySelect(array.uncheckedCast(), i0, i1, i2)
            }

            else -> mkArrayNSelect(array.uncheckedCast(), indices)
        }
    }

    private fun <T : KSort> KContext.internalizeAndConvertBitwuzla(
        nativeCtx: KBitwuzlaContext, expr: KExpr<T>
    ): KExpr<T> {
        val internalized = with(KBitwuzlaExprInternalizer(nativeCtx)) {
            expr.internalizeExpr()
        }

        val converted = with(KBitwuzlaExprConverter(this, nativeCtx)) {
            internalized.convertExpr(expr.sort)
        }

        return converted
    }

    private fun <T : KSort> KContext.internalizeAndConvertYices(
        nativeCtx: KYicesContext, expr: KExpr<T>
    ): KExpr<T> {
        val internalized = with(KYicesExprInternalizer(nativeCtx)) {
            expr.internalizeExpr()
        }

        val converted = with(KYicesExprConverter(this, nativeCtx)) {
            internalized.convert(expr.sort)
        }

        return converted
    }

    private fun <T : KSort> KContext.internalizeAndConvertZ3(nativeCtx: Context, expr: KExpr<T>): KExpr<T> {
        val z3InternCtx = KZ3Context(this, nativeCtx)
        val z3ConvertCtx = KZ3Context(this, nativeCtx)

        val internalized = with(KZ3ExprInternalizer(this, z3InternCtx)) {
            expr.internalizeExpr()
        }

        // Copy declarations since we have fresh decls
        val declarations = KExprUninterpretedDeclCollector.collectUninterpretedDeclarations(expr)
        declarations.forEach {
            val nativeDecl = z3InternCtx.findInternalizedDecl(it)
            z3ConvertCtx.saveConvertedDecl(nativeDecl, it)
        }

        val converted = with(KZ3ExprConverter(this, z3ConvertCtx)) {
            internalized.convertExpr<T>()
        }

        return converted
    }

    private fun <T : KSort> KContext.internalizeAndConvertCvc5(nativeCtx: Solver, expr: KExpr<T>): KExpr<T> {
        val internalizationCtx = KCvc5Context(nativeCtx, this)
        val conversionCtx = KCvc5Context(nativeCtx, this)

        val internalizer = KCvc5ExprInternalizer(internalizationCtx)
        val converter = KCvc5ExprConverter(this, conversionCtx)

        val internalized = with(internalizer) {
            expr.internalizeExpr()
        }

        // Copy declarations since we have fresh decls
        val declarations = KExprUninterpretedDeclCollector.collectUninterpretedDeclarations(expr)
        declarations.forEach { decl ->
            internalizationCtx.findInternalizedDecl(decl)?.also {
                conversionCtx.saveConvertedDecl(it, decl)
            }
        }

        val converted = with(converter) {
            internalized.convertExpr<T>()
        }

        return converted
    }

    private fun <T : KSort> KContext.assertEquals(
        stats: TestStats,
        oracle: KSolver<*>,
        expected: KExpr<T>,
        actual: KExpr<T>
    ) {
        if (expected == actual) {
            stats.passedFast()
            return
        }

        val satIsPossiblePassed = oracle.scoped {
            assertPossibleToBeEqual(oracle, expected, actual)
            oracle.checkSatAndReport(stats, expected, actual, KSolverStatus.SAT, "SAT is possible")
        }

        val expressionEqualityPassed = oracle.scoped {
            assertNotEqual(oracle, expected, actual)

            oracle.checkSatAndReport(
                stats, expected, actual, KSolverStatus.UNSAT, "Expressions equal"
            ) {
                val model = oracle.model()
                val actualValue = model.eval(actual, isComplete = false)
                val expectedValue = model.eval(expected, isComplete = false)

                if (expectedValue == actualValue) {
                    stats.ignore(expected, actual, "Expressions equal: check incorrect")
                    return
                }
            }
        }

        if (satIsPossiblePassed && expressionEqualityPassed) {
            stats.passed()
        }
    }

    private fun <T : KSort> KContext.assertNotEqual(
        oracle: KSolver<*>,
        expected: KExpr<T>,
        actual: KExpr<T>
    ) {
        val sort = expected.sort
        if (sort !is KArraySortBase<*>) {
            oracle.assert(expected neq actual)
            return
        }

        val expectedArray: KExpr<KArraySortBase<KBv8Sort>> = expected.uncheckedCast()
        val actualArray: KExpr<KArraySortBase<KBv8Sort>> = actual.uncheckedCast()

        // (exists (i) (/= (select expected i) (select actual i))
        val indices = sort.domainSorts.mapIndexed { i, srt -> mkFreshConst("i_${i}", srt) }
        oracle.assert(mkSelect(expectedArray) { indices } neq mkSelect(actualArray) { indices })
    }

    private fun <T : KSort> KContext.assertPossibleToBeEqual(
        oracle: KSolver<*>,
        expected: KExpr<T>,
        actual: KExpr<T>
    ) {
        val sort = expected.sort
        if (sort !is KArraySortBase<*>) {
            oracle.assert(expected eq actual)
            return
        }

        val expectedArray: KExpr<KArraySortBase<KBv8Sort>> = expected.uncheckedCast()
        val actualArray: KExpr<KArraySortBase<KBv8Sort>> = actual.uncheckedCast()

        // (exists (i) (= (select expected i) (select actual i))
        val indices = sort.domainSorts.mapIndexed { i, srt -> mkFreshConst("i_${i}", srt) }
        oracle.assert(mkSelect(expectedArray) { indices } eq mkSelect(actualArray) { indices })
    }

    private inline fun KSolver<*>.checkSatAndReport(
        stats: TestStats,
        expected: KExpr<*>,
        actual: KExpr<*>,
        expectedStatus: KSolverStatus,
        prefix: String,
        onFailure: () -> Unit = {}
    ): Boolean = when (check(CHECK_TIMEOUT)) {
        expectedStatus -> true
        KSolverStatus.UNKNOWN -> {
            val message = "$prefix: ${reasonOfUnknown()}"
            stats.ignore(expected, actual, message)
            false
        }

        else -> {
            onFailure()
            stats.fail(expected, actual, prefix)
            false
        }
    }

    private inline fun <T> KSolver<*>.scoped(block: () -> T): T {
        push()
        try {
            return block()
        } finally {
            pop()
        }
    }

    private fun mkZ3Context(ctx: KContext): Context {
        KZ3Solver(ctx).close()
        return Context()
    }

    private fun mkCvc5Context(ctx: KContext): Solver {
        KCvc5Solver(ctx).close()
        return Solver()
    }

    private inline fun TestStats.withErrorHandling(body: () -> Unit) = try {
        body()
    } catch (ex: KSolverUnsupportedFeatureException) {
        ignore(ex)
    } catch (ex: Throwable) {
        fail(ex)
    }

    private data class TestCase(
        val id: Int,
        val message: String,
        val expected: KExpr<*>?,
        val actual: KExpr<*>?
    )

    private class TestStats(
        private var total: Int = 0,
        private var passed: Int = 0,
        private var passedFast: Int = 0,
        val ignored: MutableList<TestCase> = mutableListOf(),
        val failed: MutableList<TestCase> = mutableListOf()
    ) {
        var testId = 0
            private set

        fun start() {
            testId = total++
        }

        fun passed() {
            passed++
        }

        fun passedFast() {
            passedFast++
            passed()
        }

        fun ignore(expected: KExpr<*>, actual: KExpr<*>, message: String) {
            System.err.println("IGNORED ${testId}: $message")
            ignored += TestCase(testId, message, expected, actual)
        }

        fun fail(expected: KExpr<*>, actual: KExpr<*>, message: String) {
            System.err.println("FAILED ${testId}: $message")
            failed += TestCase(testId, message, expected, actual)
        }

        fun fail(ex: Throwable) {
            System.err.println("FAILED ${testId}: $ex")
            failed += TestCase(testId, message = "$ex", expected = null, actual = null)
        }

        fun ignore(ex: Throwable) {
            System.err.println("IGNORED ${testId}: $ex")
            ignored += TestCase(testId, message = "$ex", expected = null, actual = null)
        }

        fun result() {
            val stats = listOf(
                "total=${total}",
                "failed=${failed.size}",
                "ignored=${ignored.size}",
                "passed=${passed}",
                "passedFast=${passedFast}"
            )
            System.err.println("STATS: $stats")

            assertTrue(failed.isEmpty(), "Some tests failed")
        }
    }

    companion object {
        private val CHECK_TIMEOUT = 10.seconds

        private val oracleManager = KSolverRunnerManager(
            workerPoolSize = 2,
            hardTimeout = 1.minutes
        )

        @AfterAll
        @JvmStatic
        fun cleanup() {
            oracleManager.close()
        }
    }
}
