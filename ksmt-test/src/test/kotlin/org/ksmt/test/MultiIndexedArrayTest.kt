package org.ksmt.test

import com.microsoft.z3.Context
import org.junit.jupiter.api.AfterAll
import org.ksmt.KContext
import org.ksmt.KContext.SimplificationMode.NO_SIMPLIFY
import org.ksmt.expr.KExpr
import org.ksmt.expr.rewrite.KExprUninterpretedDeclCollector
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.bitwuzla.KBitwuzlaContext
import org.ksmt.solver.bitwuzla.KBitwuzlaExprConverter
import org.ksmt.solver.bitwuzla.KBitwuzlaExprInternalizer
import org.ksmt.solver.bitwuzla.KBitwuzlaSolver
import org.ksmt.solver.runner.KSolverRunnerManager
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

//@Disabled
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
            KBitwuzlaContext(this).use { z3NativeCtx ->
                runMultiIndexedArraySamples(oracleSolver) { expr ->
                    internalizeAndConvertBitwuzla(z3NativeCtx, expr)
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
            KBitwuzlaContext(this).use { z3NativeCtx ->
                runMultiIndexedArraySamples(oracleSolver) { expr ->
                    internalizeAndConvertBitwuzla(z3NativeCtx, expr)
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
                try {
                    val processed = process(expr)
                    assertEquals(stats, oracle, expr, processed)
                } catch (ex: Throwable) {
                    stats.fail(ex)
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

        val arrayEq = arrayExpressions.zipWithNext().map { (first, second) -> first eq second }

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

    private fun <T : KSort> KContext.internalizeAndConvertZ3(nativeCtx: Context, expr: KExpr<T>): KExpr<T> {
        val z3InternCtx = KZ3Context(nativeCtx)
        val z3ConvertCtx = KZ3Context(nativeCtx)

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

        oracle.scoped {
            assertNotEqual(oracle, expected, actual)

            oracle.checkSatAndReport(stats, expected, actual, KSolverStatus.UNSAT, "Expressions equal") {
                val model = oracle.model()
                val actualValue = model.eval(actual, isComplete = false)
                val expectedValue = model.eval(expected, isComplete = false)

                if (expectedValue == actualValue) {
                    stats.ignore(expected, actual, "Expressions equal: check incorrect")
                    return
                }
            } ?: return
        }

        stats.passed()
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

    private inline fun KSolver<*>.checkSatAndReport(
        stats: TestStats,
        expected: KExpr<*>,
        actual: KExpr<*>,
        expectedStatus: KSolverStatus,
        prefix: String,
        onFailure: () -> Unit = {}
    ): Unit? = when (check()) {
        expectedStatus -> Unit
        KSolverStatus.UNKNOWN -> {
            val message = "$prefix: ${reasonOfUnknown()}"
            stats.ignore(expected, actual, message)
            null
        }

        else -> {
            onFailure()
            stats.fail(expected, actual, prefix)
            null
        }
    }

    private inline fun KSolver<*>.scoped(block: () -> Unit) {
        push()
        try {
            block()
        } finally {
            pop()
        }
    }

    private fun mkZ3Context(ctx: KContext): Context {
        KZ3Solver(ctx).close()
        return Context()
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
            failed += TestCase(testId, "$ex", null, null)
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
