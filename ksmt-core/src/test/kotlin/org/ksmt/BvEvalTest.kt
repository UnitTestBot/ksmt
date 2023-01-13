package org.ksmt

import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KExpr
import org.ksmt.expr.KInterpretedConstant
import org.ksmt.expr.rewrite.simplify.KExprSimplifier
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KSort
import org.ksmt.utils.BvUtils
import org.ksmt.utils.uncheckedCast
import kotlin.random.Random
import kotlin.random.nextInt
import kotlin.test.assertEquals
import kotlin.test.assertTrue

@Execution(ExecutionMode.CONCURRENT)
class BvEvalTest {

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvAdd(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvAddExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvAddNoOverflow(sizeBits: Int) {
        testOperation(sizeBits) { a, b -> mkBvAddNoOverflowExpr(a, b, isSigned = true) }
        testOperation(sizeBits) { a, b -> mkBvAddNoOverflowExpr(a, b, isSigned = false) }
    }

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvAddNoUnderflow(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvAddNoUnderflowExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvAnd(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvAndExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvArithShiftRight(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvArithShiftRightExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvConcat(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvConcatExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvDivNoOverflow(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvDivNoOverflowExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvExtract(sizeBits: Int) {
        repeat(5) {
            val high = random.nextInt(0 until sizeBits)
            repeat(5) {
                val low = random.nextInt(0..high)
                testOperation(sizeBits) { value -> mkBvExtractExpr(high, low, value) }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvLogicalShiftRight(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvLogicalShiftRightExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvMul(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvMulExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvMulNoOverflow(sizeBits: Int) {
        testOperation(sizeBits) { a, b -> mkBvMulNoOverflowExpr(a, b, isSigned = true) }
        testOperation(sizeBits) { a, b -> mkBvMulNoOverflowExpr(a, b, isSigned = false) }
    }

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvMulNoUnderflow(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvMulNoUnderflowExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvNAnd(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvNAndExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvNegation(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvNegationExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvNegationNoOverflow(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvNegationNoOverflowExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvNor(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvNorExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvNot(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvNotExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvOr(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvOrExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvReductionAnd(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvReductionAndExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvReductionOr(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvReductionOrExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvRepeat(sizeBits: Int) {
        repeat(5) {
            val repetitions = random.nextInt(1..10)
            testOperation(sizeBits) { value -> mkBvRepeatExpr(repetitions, value) }
        }
    }

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvRotateLeft(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvRotateLeftExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvRotateLeftIndexed(sizeBits: Int) {
        repeat(5) {
            val rotation = random.nextInt(0..1024)
            testOperation(sizeBits) { value -> mkBvRotateLeftIndexedExpr(rotation, value) }
        }
    }

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvRotateRight(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvRotateRightExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvRotateRightIndexed(sizeBits: Int) {
        repeat(5) {
            val rotation = random.nextInt(0..1024)
            testOperation(sizeBits) { value -> mkBvRotateRightIndexedExpr(rotation, value) }
        }
    }

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvShiftLeft(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvShiftLeftExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedDiv(sizeBits: Int) = testOperationNonZeroSecondArg(sizeBits, KContext::mkBvSignedDivExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedGreater(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvSignedGreaterExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedGreaterOrEqual(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvSignedGreaterOrEqualExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedLess(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvSignedLessExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedLessOrEqual(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvSignedLessOrEqualExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedMod(sizeBits: Int) = testOperationNonZeroSecondArg(sizeBits, KContext::mkBvSignedModExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedRem(sizeBits: Int) = testOperationNonZeroSecondArg(sizeBits, KContext::mkBvSignedRemExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignExtension(sizeBits: Int) {
        repeat(5) {
            val extension = random.nextInt(0..100)
            testOperation(sizeBits) { value -> mkBvSignExtensionExpr(extension, value) }
        }
    }

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSub(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvSubExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSubNoOverflow(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvSubNoOverflowExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSubNoUnderflow(sizeBits: Int) {
        testOperation(sizeBits) { a, b -> mkBvSubNoUnderflowExpr(a, b, isSigned = true) }
        testOperation(sizeBits) { a, b -> mkBvSubNoUnderflowExpr(a, b, isSigned = false) }
    }

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvUnsignedDiv(sizeBits: Int) = testOperationNonZeroSecondArg(sizeBits, KContext::mkBvUnsignedDivExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvUnsignedGreater(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvUnsignedGreaterExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvUnsignedGreaterOrEqual(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvUnsignedGreaterOrEqualExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvUnsignedLess(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvUnsignedLessExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvUnsignedLessOrEqual(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvUnsignedLessOrEqualExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvUnsignedRem(sizeBits: Int) = testOperationNonZeroSecondArg(sizeBits, KContext::mkBvUnsignedRemExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvXNor(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvXNorExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvXor(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvXorExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvZeroExtension(sizeBits: Int) {
        repeat(5) {
            val extension = random.nextInt(0..100)
            testOperation(sizeBits) { value -> mkBvZeroExtensionExpr(extension, value) }
        }
    }

    private fun <S : KBvSort> KContext.randomBvValues(sort: S) = sequence<KBitVecValue<S>> {
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

    private fun <S : KBvSort> KContext.nonZeroRandomValues(sort: S): Sequence<KBitVecValue<S>> {
        val zero = mkBv(0, sort)
        return randomBvValues(sort).filter { it != zero }
    }

    private fun <S : KBvSort, T : KSort> testOperation(
        size: Int,
        operation: KContext.(KExpr<S>) -> KExpr<T>
    ) = runTest(size) { sort: S, checker ->
        randomBvValues(sort).forEach { value ->
            val expr = operation(value)
            checker.check(expr)
        }
    }

    private fun <S : KBvSort, T : KSort> testOperation(
        size: Int,
        operation: KContext.(KExpr<S>, KExpr<S>) -> KExpr<T>
    ) = runTest(size) { sort: S, checker ->
        randomBvValues(sort).forEach { a ->
            randomBvValues(sort).forEach { b ->
                val expr = operation(a, b)
                checker.check(expr)
            }
        }
    }

    private fun <S : KBvSort, T : KSort> testOperationNonZeroSecondArg(
        size: Int,
        operation: KContext.(KExpr<S>, KExpr<S>) -> KExpr<T>
    ) = runTest(size) { sort: S, checker ->
        randomBvValues(sort).forEach { a ->
            nonZeroRandomValues(sort).forEach { b ->
                val expr = operation(a, b)
                checker.check(expr)
            }
        }
    }

    private fun <S : KBvSort> runTest(size: Int, test: KContext.(S, TestRunner) -> Unit) {
        val ctx = KContext()
        val sort: S = ctx.mkBvSort(size.toUInt()).uncheckedCast()
        KZ3Solver(ctx).use { solver ->
            val checker = TestRunner(ctx, solver)
            ctx.test(sort, checker)
        }
    }

    private class TestRunner(
        private val ctx: KContext,
        private val solver: KSolver<*>
    ) {

        fun <T : KSort> check(expr: KExpr<T>) {
            val expectedValue = solverValue(expr)
            val actualValue = evalBvExpr(expr)
            assertEquals(expectedValue, actualValue)
        }

        private fun <T : KSort> solverValue(expr: KExpr<T>): KExpr<T> =
            withSolverScope { solver ->
                with(ctx) {
                    val valueVar = mkFreshConst("v", expr.sort)
                    solver.assert(valueVar eq expr)
                    assertEquals(KSolverStatus.SAT, solver.check())
                    val value = solver.model().eval(valueVar)
                    assertTrue(value is KInterpretedConstant)
                    value
                }
            }


        private fun <T : KSort> evalBvExpr(expr: KExpr<T>): KExpr<T> {
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
        private val random = Random(42)

        private val bvSizesToTest by lazy {
            val context = KContext()
            val smallCustomBv = context.mkBvSort(17u)
            val largeCustomBv = context.mkBvSort(111u)

            listOf(
                context.bv1Sort,
                context.bv8Sort,
                context.bv16Sort,
                context.bv32Sort,
                context.bv64Sort,
                smallCustomBv,
                largeCustomBv
            ).map { it.sizeBits.toInt() }
        }

        @JvmStatic
        fun bvSizes() = bvSizesToTest.map { Arguments.of(it) }
    }
}
