package org.ksmt.test

import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable
import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.bitwuzla.KBitwuzlaSolver
import org.ksmt.solver.cvc5.KCvc5Solver
import org.ksmt.solver.runner.KSolverRunnerManager
import org.ksmt.solver.yices.KYicesSolver
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.utils.uncheckedCast
import kotlin.random.Random
import kotlin.random.nextInt
import kotlin.reflect.KClass
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds

@EnabledIfEnvironmentVariable(
    named = "runUninterpretedSortValuesTest",
    matches = "true",
    disabledReason = "Not suitable for usual CI runs"
)
class UninterpretedSortValuesTest {

    @Test
    fun test(): Unit = with(KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY)) {
        val testData = generateTestData()
        KSolverRunnerManager(workerPoolSize = 3).use { solverManager ->
            solverManager.createSolver(this, KZ3Solver::class).use { oracleSover ->
                val z3Results = solverManager.testSolver(this, KZ3Solver::class, testData)
                validateResults(oracleSover, z3Results)

                val yicesResults = solverManager.testSolver(this, KYicesSolver::class, testData)
                validateResults(oracleSover, yicesResults)

                val bitwuzlaResults = solverManager.testSolver(this, KBitwuzlaSolver::class, testData)
                validateResults(oracleSover, bitwuzlaResults)

                val cvcResults = solverManager.testSolver(this, KCvc5Solver::class, testData)
                validateResults(oracleSover, cvcResults)
            }
        }
    }

    @Test
    fun testIncremental(): Unit = with(KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY)) {
        val testData = generateTestData()
        KSolverRunnerManager(workerPoolSize = 3).use { solverManager ->
            solverManager.createSolver(this, KZ3Solver::class).use { oracleSover ->
                val z3Results = solverManager.testSolverIncremental(this, KZ3Solver::class, testData)
                validateResults(oracleSover, z3Results)

                val yicesResults = solverManager.testSolverIncremental(this, KYicesSolver::class, testData)
                validateResults(oracleSover, yicesResults)

                val bitwuzlaResults = solverManager.testSolverIncremental(this, KBitwuzlaSolver::class, testData)
                validateResults(oracleSover, bitwuzlaResults)

                val cvcResults = solverManager.testSolverIncremental(this, KCvc5Solver::class, testData)
                validateResults(oracleSover, cvcResults)
            }
        }
    }

    private fun <C : KSolverConfiguration> KSolverRunnerManager.testSolver(
        ctx: KContext,
        solverType: KClass<out KSolver<C>>,
        data: List<KExpr<KBoolSort>>
    ) = createSolver(ctx, solverType).use { solver ->
        val results = arrayListOf<TestResult>()

        for ((i, sample) in data.withIndex()) {
            solver.push()

            solver.assert(sample)
            val status = solver.check(SINGLE_CHECK_TIMEOUT)

            val model = if (status == KSolverStatus.SAT) solver.model() else null

            results += TestResult(
                testId = i,
                solverName = solverType.simpleName!!,
                testSample = sample,
                status = status,
                model = model?.detach()
            )

            solver.pop()
        }

        results
    }

    private data class SolverTestDataSample(
        val index: Int,
        val expr: KExpr<KBoolSort>,
        val variable: KExpr<KBoolSort>
    )

    private fun <C : KSolverConfiguration> KSolverRunnerManager.testSolverIncremental(
        ctx: KContext,
        solverType: KClass<out KSolver<C>>,
        data: List<KExpr<KBoolSort>>
    ) = createSolver(ctx, solverType).use { solver ->
        val results = arrayListOf<TestResult>()

        val chunks = data.mapIndexed { index, sample ->
            val variable = ctx.mkFreshConst("t:$index", ctx.boolSort)
            SolverTestDataSample(index, sample, variable)
        }.chunked(TEST_SET_INCREMENTAL_CHUNK)

        ctx.pushAssertChunk(solver, chunks.first())

        for (chunk in chunks.drop(1)) {
            ctx.pushAssertChunk(solver, chunk)
            popCheckSatChunk(solver, solverType.simpleName!!, chunk, results)
        }

        popCheckSatChunk(solver, solverType.simpleName!!, chunks.first(), results)

        results
    }

    private fun KContext.pushAssertChunk(
        solver: KSolver<*>,
        chunk: List<SolverTestDataSample>
    ) {
        for (sample in chunk) {
            solver.push()
            solver.assert(sample.variable implies sample.expr)
        }
    }

    private fun popCheckSatChunk(
        solver: KSolver<*>,
        solverName: String,
        chunk: List<SolverTestDataSample>,
        results: MutableList<TestResult>
    ) {
        for (sample in chunk.asReversed()) {
            solver.push()
            solver.assert(sample.variable)
            val status = solver.check(SINGLE_CHECK_TIMEOUT)

            val model = if (status == KSolverStatus.SAT) solver.model() else null

            results += TestResult(
                testId = sample.index,
                solverName = solverName,
                testSample = sample.expr,
                status = status,
                model = model?.detach()
            )

            solver.pop()

            // pop sample assertion
            solver.pop()
        }
    }

    private fun KContext.validateResults(oracle: KSolver<*>, results: List<TestResult>) {
        validateResultStatus(results)

        for (res in results) {
            if (res.model == null) continue
            validateModel(oracle, res, res.model)
        }
    }

    private fun KContext.validateModel(oracle: KSolver<*>, result: TestResult, model: KModel) {
        val actualValue = model.eval(result.testSample, isComplete = false)
        if (actualValue == trueExpr) return

        oracle.push()

        model.uninterpretedSorts.forEach { sort ->
            val universe = model.uninterpretedSortUniverse(sort)
            if (!universe.isNullOrEmpty()) {
                val x = mkFreshConst("x", sort)
                val constraint = mkOr(universe.map { x eq it })

                oracle.assert(mkUniversalQuantifier(constraint, listOf(x.decl)))
            }
        }

        oracle.assert(actualValue neq trueExpr)

        val status = oracle.check()
        assertEquals(KSolverStatus.UNSAT, status, result.solverName)

        oracle.pop()
    }

    private fun validateResultStatus(results: List<TestResult>) {
        val statuses = results
            .groupBy({ it.testId }, { it.status })
            .mapValues { (_, status) -> status.filterNot { it == KSolverStatus.UNKNOWN }.toSet() }

        assertTrue(statuses.all { it.value.size <= 1 }, results.first().solverName)
    }

    private fun KContext.generateTestData(): List<KExpr<KBoolSort>> {
        val sorts = (0 until NUMBER_OF_UNINTERPRETED_SORTS).map {
            mkUninterpretedSort("T${it}")
        }
        val generationContext = GenerationContext(this, sorts)
        return (0 until TEST_SET_SIZE).map {
            generationContext.generate()
        }
    }

    private class GenerationContext(
        val ctx: KContext,
        val sorts: List<KUninterpretedSort>
    ) {
        private val random = Random(RANDOM_SEED)

        private val sortMapOperations = sorts.associateWith { s1 ->
            sorts.associateWith { s2 ->
                ctx.mkFuncDecl("${s1}_${s2}", s2, listOf(s1))
            }
        }

        private val basicExpressions = sorts.associateWithTo(hashMapOf<KSort, MutableList<KExpr<*>>>()) {
            arrayListOf(ctx.mkFreshConst("x", it))
        }

        private val expressions = hashMapOf<KSort, MutableList<KExpr<*>>>()

        private fun newExpr() = when (random.nextDouble()) {
            in 0.0..0.1 -> newArray()
            in 0.1..0.3 -> newVar()
            else -> newValue()
        }

        private fun newArray() = with(ctx) {
            val sort = mkArraySort(sorts.random(random), sorts.random(random))
            mkFreshConst("array", sort)
        }

        private fun newVar() = with(ctx) {
            mkFreshConst("v", sorts.random(random))
        }

        private fun newValue() = with(ctx) {
            mkUninterpretedSortValue(sorts.random(random), random.nextInt(USORT_VALUE_RANGE))
        }

        private fun findExpr(sort: KSort): KExpr<KSort> {
            if (random.nextDouble() > 0.5) {
                newExpr().also {
                    basicExpressions.getOrPut(it.sort) { arrayListOf() }.add(it)
                }
            }

            val expressionSet = when (random.nextDouble()) {
                in 0.0..0.4 -> basicExpressions
                else -> expressions
            }

            val variants = expressionSet.getOrPut(sort) {
                arrayListOf(ctx.mkFreshConst("stub", sort))
            }

            return variants.random(random).uncheckedCast()
        }

        fun generate(): KExpr<KBoolSort> {
            expressions.clear()

            val first = generateDeepExpr()
            val second = generateDeepExpr()

            val transformer = sortMapOperations[first.sort]?.get(second.sort)
                ?: error("Unexpected sorts: ${first.sort}, ${second.sort}")
            val firstAsSecond = transformer.apply(listOf(first))

            return ctx.mkEq(firstAsSecond, second.uncheckedCast())
        }

        private fun generateDeepExpr(): KExpr<KSort> {
            var current = findExpr(sorts.random(random))
            repeat(EXPRESSION_DEPTH) {
                current = if (current.sort is KArraySort<*, *>) {
                    generateArrayOperation(current.uncheckedCast())
                } else {
                    generateOperation(current)
                }
                expressions.getOrPut(current.sort) { arrayListOf() }.add(current)
            }

            val sort = current.sort
            if (sort is KArraySort<*, *>) {
                current = ctx.mkArraySelect(current.uncheckedCast(), findExpr(sort.domain))
            }

            return current
        }

        private fun generateArrayOperation(expr: KExpr<KArraySort<KSort, KSort>>): KExpr<KSort> = with(ctx) {
            when (random.nextDouble()) {
                in 0.0..0.3 -> mkArraySelect(expr, findExpr(expr.sort.domain))
                else -> mkArrayStore(expr, findExpr(expr.sort.domain), findExpr(expr.sort.range)).uncheckedCast()
            }
        }

        private fun generateOperation(expr: KExpr<KSort>): KExpr<KSort> = with(ctx) {
            when (random.nextDouble()) {
                in 0.0..0.5 -> {
                    val otherSort = sorts.random(random)
                    val transformer = sortMapOperations[expr.sort]?.get(otherSort)
                        ?: error("Unexpected expr sort: ${expr.sort}")
                    transformer.apply(listOf(expr)).uncheckedCast()
                }

                else -> {
                    val sort = mkArraySort(expr.sort, sorts.random(random))
                    val array: KExpr<KArraySort<KSort, KSort>> = findExpr(sort).uncheckedCast()
                    mkArrayStore(array, expr, findExpr(sort.range)).uncheckedCast()
                }
            }
        }
    }

    private data class TestResult(
        val testId: Int,
        val solverName: String,
        val testSample: KExpr<KBoolSort>,
        val status: KSolverStatus,
        val model: KModel?
    )

    companion object {
        private const val NUMBER_OF_UNINTERPRETED_SORTS = 3
        private const val RANDOM_SEED = 42
        private const val EXPRESSION_DEPTH = 30
        private const val TEST_SET_SIZE = 500
        private const val TEST_SET_INCREMENTAL_CHUNK = TEST_SET_SIZE / 50
        private val USORT_VALUE_RANGE = 0 until 25
        private val SINGLE_CHECK_TIMEOUT = 1.seconds
    }
}
