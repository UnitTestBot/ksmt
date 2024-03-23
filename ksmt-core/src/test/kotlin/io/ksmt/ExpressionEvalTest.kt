package io.ksmt

import io.ksmt.KContext.SimplificationMode.SIMPLIFY
import io.ksmt.expr.KApp
import io.ksmt.expr.KBitVecValue
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFpRoundingMode
import io.ksmt.expr.KFpRoundingModeExpr
import io.ksmt.expr.KFpValue
import io.ksmt.expr.KInt32NumExpr
import io.ksmt.expr.KInterpretedValue
import io.ksmt.expr.KRealNumExpr
import io.ksmt.expr.rewrite.simplify.KExprSimplifier
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KSort
import io.ksmt.utils.BvUtils
import io.ksmt.utils.FpUtils.mkFpMaxValue
import io.ksmt.utils.uncheckedCast
import kotlin.random.Random
import kotlin.random.nextInt
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.asserter

open class ExpressionEvalTest {

    fun <S : KBvSort> KContext.randomBvValues(sort: S) = sequence<KBitVecValue<S>> {
        // special values
        with(BvUtils) {
            yield(bvMaxValueSigned(sort.sizeBits))
            yield(bvMaxValueUnsigned(sort.sizeBits))
            yield(bvMinValueSigned(sort.sizeBits))
            yield(bvZero(sort.sizeBits))
            yield(bvOne(sort.sizeBits))
        }

        // small positive values
        repeat(5) {
            val value = random.nextInt(1..20)
            yield(mkBv(value, sort))
        }

        // small negative values
        repeat(5) {
            val value = random.nextInt(1..20)
            yield(mkBv(-value, sort))
        }

        // random values
        repeat(50) {
            val binaryValue = String(CharArray(sort.sizeBits.toInt()) {
                if (random.nextBoolean()) '1' else '0'
            })
            yield(mkBv(binaryValue, sort.sizeBits).uncheckedCast())
        }
    }

    fun <S : KBvSort> KContext.randomBvNonZeroValues(sort: S): Sequence<KBitVecValue<S>> {
        val zero = mkBv(0, sort)
        return randomBvValues(sort).filter { it != zero }
    }

    fun KContext.roundingModeValues(): Sequence<KFpRoundingModeExpr> =
        KFpRoundingMode.values().asSequence().map { mkFpRoundingModeExpr(it) }

    fun <S : KFpSort> KContext.randomFpValues(sort: S) = sequence<KFpValue<S>> {
        // special values
        yield(mkFpZero(sort = sort, signBit = true))
        yield(mkFpZero(sort = sort, signBit = false))
        yield(mkFpInf(sort = sort, signBit = true))
        yield(mkFpInf(sort = sort, signBit = false))
        yield(mkFpNaN(sort))
        yield(mkFpMaxValue(sort = sort, signBit = false))
        yield(mkFpMaxValue(sort = sort, signBit = true))

        // small positive values
        repeat(5) {
            val value = random.nextDouble()
            yield(mkFp(value, sort))
        }

        // small negative values
        repeat(5) {
            val value = random.nextDouble()
            yield(mkFp(-value, sort))
        }

        // random values
        val exponentBvSort = mkBvSort(sort.exponentBits)
        val significandBvSort = mkBvSort(sort.significandBits - 1u)
        randomBvValues(exponentBvSort).shuffled(random).forEach { exponent ->
            randomBvValues(significandBvSort).shuffled(random).forEach { significand ->
                val sign = random.nextBoolean()
                val value = mkFpBiased(significand, exponent, sign, sort)
                yield(value)
            }
        }
    }

    fun KContext.randomRealValues() = sequence<KRealNumExpr> {
        randomIntValues().filter { it.value != 0 }.forEach { denominator ->
            randomIntValues().forEach { numerator ->
                yield(mkRealNum(numerator, denominator))
            }
        }
    }

    fun KContext.randomIntValues() = sequence<KInt32NumExpr> {
        repeat(1000) {
            yield(mkIntNum(random.nextInt()) as KInt32NumExpr)
        }
    }

    internal fun <S : KSort> runTest(
        mkSort: KContext.() -> S,
        test: KContext.(S, TestRunner) -> Unit
    ) {
        val ctx = KContext(simplificationMode = SIMPLIFY)
        val sort = ctx.mkSort()
        val checker = TestRunner(ctx)
        ctx.test(sort, checker)
        checker.runTests()
    }

    private class TestCase(
        val unsimplifiedExpr: KExpr<*>,
        val simplifiedExpr: KExpr<*>,
        val printArgs: () -> String
    ) {
        lateinit var solverValue: KExpr<*>
    }

    internal class TestRunner(private val ctx: KContext) {
        private val testCases = arrayListOf<TestCase>()

        fun <T : KSort> check(unsimplifiedExpr: KExpr<T>, simplifiedExpr: KExpr<T>, printArgs: () -> String) {
            testCases += TestCase(unsimplifiedExpr, simplifiedExpr, printArgs)

            if (testCases.size > 1000) {
                runTests()
            }
        }

        fun runTests() {
            KZ3Solver(ctx).use {
                computeSolverValues(it)
            }

            testCases.forEach {
                val expectedValue = it.solverValue
                val simplifierValueValue = evalExpr(it.unsimplifiedExpr)

                assertEquals(expectedValue, it.simplifiedExpr) { it.printArgs() }
                assertEquals(expectedValue, simplifierValueValue) { it.printArgs() }

                val decl = (expectedValue as KApp<*, *>).decl
                val declValue = decl.apply(emptyList())
                assertEquals(expectedValue, declValue) { it.printArgs() }
            }

            testCases.clear()
        }

        private fun computeSolverValues(solver: KSolver<*>) = with(ctx) {
            val testCaseVars = testCases.map {
                val valueVar = mkFreshConst("v", it.unsimplifiedExpr.sort)
                solver.assert(valueVar eq it.unsimplifiedExpr.uncheckedCast())
                valueVar
            }
            assertEquals(KSolverStatus.SAT, solver.check())

            val model = solver.model()
            testCases.zip(testCaseVars).forEach { (testCase, tcVar) ->
                val value = model.eval(tcVar, isComplete = false)
                assertTrue(value is KInterpretedValue<*>)
                testCase.solverValue = value
            }
        }

        private fun <T : KSort> evalExpr(expr: KExpr<T>): KExpr<T> {
            val evaluator = KExprSimplifier(ctx)
            return evaluator.apply(expr)
        }
    }

    companion object {
        val random = Random(42)
    }
}

private fun assertEquals(expected: Any?, actual: Any?, lazyMessage: () -> String) {
    asserter.assertTrue(
        lazyMessage = {
            val message = lazyMessage()
            "$message\nExpected <$expected>, actual <$actual>."
        },
        expected == actual
    )
}
