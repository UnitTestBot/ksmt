package org.ksmt

import org.ksmt.expr.KApp
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.expr.KFpRoundingModeExpr
import org.ksmt.expr.KFpValue
import org.ksmt.expr.KInt32NumExpr
import org.ksmt.expr.KInterpretedValue
import org.ksmt.expr.KRealNumExpr
import org.ksmt.expr.rewrite.simplify.KExprSimplifier
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.utils.BvUtils
import org.ksmt.utils.FpUtils.mkFpMaxValue
import org.ksmt.utils.uncheckedCast
import kotlin.random.Random
import kotlin.random.nextInt
import kotlin.test.assertEquals
import kotlin.test.assertTrue

open class ExpressionEvalTest {

    fun <S : KBvSort> KContext.randomBvValues(sort: S) = sequence<KBitVecValue<S>> {
        // special values
        with(BvUtils) {
            yield(bvMaxValueSigned(sort.sizeBits).uncheckedCast())
            yield(bvMaxValueUnsigned(sort.sizeBits).uncheckedCast())
            yield(bvMinValueSigned(sort.sizeBits).uncheckedCast())
            yield(bvZero(sort.sizeBits).uncheckedCast())
            yield(bvOne(sort.sizeBits).uncheckedCast())
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
        repeat(30) {
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
        val ctx = KContext()
        val sort = ctx.mkSort()
        KZ3Solver(ctx).use { solver ->
            val checker = TestRunner(ctx, solver)
            ctx.test(sort, checker)
        }
    }

    internal class TestRunner(
        private val ctx: KContext,
        private val solver: KSolver<*>
    ) {

        fun <T : KSort> check(expr: KExpr<T>, printArgs: () -> String) {
            val expectedValue = solverValue(expr)
            val actualValue = evalExpr(expr)

            assertEquals(expectedValue, actualValue, printArgs())

            val decl = (expectedValue as KApp<*, *>).decl
            val declValue = decl.apply(emptyList())
            assertEquals(expectedValue, declValue, printArgs())
        }

        private fun <T : KSort> solverValue(expr: KExpr<T>): KExpr<T> =
            withSolverScope { solver ->
                with(ctx) {
                    val valueVar = mkFreshConst("v", expr.sort)
                    solver.assert(valueVar eq expr)
                    assertEquals(KSolverStatus.SAT, solver.check())
                    val value = solver.model().eval(valueVar)
                    assertTrue(value is KInterpretedValue<*>)
                    value
                }
            }


        private fun <T : KSort> evalExpr(expr: KExpr<T>): KExpr<T> {
            val evaluator = KExprSimplifier(ctx)
            return evaluator.apply(expr)
        }

        private fun <T> withSolverScope(block: (KSolver<*>) -> T): T = try {
            solver.push()
            block(solver)
        } finally {
            solver.pop()
        }
    }

    companion object {
        val random = Random(42)
    }
}
