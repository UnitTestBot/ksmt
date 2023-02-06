package org.ksmt

import org.junit.jupiter.api.Test
import org.ksmt.expr.KExpr
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KSort
import org.ksmt.utils.BvUtils.bvZero
import org.ksmt.utils.uncheckedCast
import kotlin.random.Random
import kotlin.random.nextInt
import kotlin.test.assertEquals

class BvSimplifyTest {

    @Test
    fun testBvAdd() =
        testOperation(KContext::mkBvAddExpr, KContext::mkBvAddExprNoSimplify)

    @Test
    fun testBvAddNoOverflow() =
        testOperation(KContext::mkBvAddNoOverflowExpr, KContext::mkBvAddNoOverflowExprNoSimplify)

    @Test
    fun testBvAddNoUnderflow() =
        testOperation(KContext::mkBvAddNoUnderflowExpr, KContext::mkBvAddNoUnderflowExprNoSimplify)

    @Test
    fun testBvAnd() =
        testOperation(KContext::mkBvAndExpr, KContext::mkBvAndExprNoSimplify)

    @Test
    fun testBvArithShiftRight() =
        testOperation(KContext::mkBvArithShiftRightExpr, KContext::mkBvArithShiftRightExprNoSimplify)

    @Test
    fun testBvConcat() =
        testOperation(KContext::mkBvConcatExpr, KContext::mkBvConcatExprNoSimplify)

    @Test
    fun testBvDivNoOverflow() =
        testOperation(KContext::mkBvDivNoOverflowExpr, KContext::mkBvDivNoOverflowExprNoSimplify)

    @Test
    fun testBvExtract() {
        repeat(5) {
            val high = random.nextInt(0 until BV_SIZE.toInt())
            repeat(5) {
                val low = random.nextInt(0..high)
                testOperation(
                    { value: KExpr<KBvSort> -> mkBvExtractExpr(high, low, value) },
                    { value: KExpr<KBvSort> -> mkBvExtractExprNoSimplify(high, low, value) }
                )
            }
        }
    }

    @Test
    fun testBvLogicalShiftRight() =
        testOperation(KContext::mkBvLogicalShiftRightExpr, KContext::mkBvLogicalShiftRightExprNoSimplify)

    @Test
    fun testBvMul() = testOperation(KContext::mkBvMulExpr, KContext::mkBvMulExprNoSimplify)

    @Test
    fun testBvMulNoOverflow() =
        testOperation(KContext::mkBvMulNoOverflowExpr, KContext::mkBvMulNoOverflowExprNoSimplify)

    @Test
    fun testBvMulNoUnderflow() =
        testOperation(KContext::mkBvMulNoUnderflowExpr, KContext::mkBvMulNoUnderflowExprNoSimplify)

    @Test
    fun testBvNAnd() =
        testOperation(KContext::mkBvNAndExpr, KContext::mkBvNAndExprNoSimplify)

    @Test
    fun testBvNegation() =
        testOperation(KContext::mkBvNegationExpr, KContext::mkBvNegationExprNoSimplify)

    @Test
    fun testBvNegationNoOverflow() =
        testOperation(KContext::mkBvNegationNoOverflowExpr, KContext::mkBvNegationNoOverflowExprNoSimplify)

    @Test
    fun testBvNor() =
        testOperation(KContext::mkBvNorExpr, KContext::mkBvNorExprNoSimplify)

    @Test
    fun testBvNot() =
        testOperation(KContext::mkBvNotExpr, KContext::mkBvNotExprNoSimplify)

    @Test
    fun testBvOr() =
        testOperation(KContext::mkBvOrExpr, KContext::mkBvOrExprNoSimplify)

    @Test
    fun testBvReductionAnd() =
        testOperation(KContext::mkBvReductionAndExpr, KContext::mkBvReductionAndExprNoSimplify)

    @Test
    fun testBvReductionOr() =
        testOperation(KContext::mkBvReductionOrExpr, KContext::mkBvReductionOrExprNoSimplify)

    @Test
    fun testBvRepeat() {
        repeat(5) {
            val repetitions = random.nextInt(1..10)
            testOperation(
                { value: KExpr<KBvSort> -> mkBvRepeatExpr(repetitions, value) },
                { value: KExpr<KBvSort> -> mkBvRepeatExprNoSimplify(repetitions, value) }
            )
        }
    }

    @Test
    fun testBvRotateLeft() =
        testOperation(KContext::mkBvRotateLeftExpr, KContext::mkBvRotateLeftExprNoSimplify)

    @Test
    fun testBvRotateLeftIndexed() {
        repeat(5) {
            val rotation = random.nextInt(0..1024)
            testOperation(
                { value: KExpr<KBvSort> -> mkBvRotateLeftIndexedExpr(rotation, value) },
                { value: KExpr<KBvSort> -> mkBvRotateLeftIndexedExprNoSimplify(rotation, value) }
            )
        }
    }

    @Test
    fun testBvRotateRight() =
        testOperation(KContext::mkBvRotateRightExpr, KContext::mkBvRotateRightExprNoSimplify)

    @Test
    fun testBvRotateRightIndexed() {
        repeat(5) {
            val rotation = random.nextInt(0..1024)
            testOperation(
                { value: KExpr<KBvSort> -> mkBvRotateRightIndexedExpr(rotation, value) },
                { value: KExpr<KBvSort> -> mkBvRotateRightIndexedExprNoSimplify(rotation, value) }
            )
        }
    }

    @Test
    fun testBvShiftLeft() =
        testOperation(KContext::mkBvShiftLeftExpr, KContext::mkBvShiftLeftExprNoSimplify)

    @Test
    fun testBvSignedDiv() =
        testOperationNonZeroSecondArg(KContext::mkBvSignedDivExpr, KContext::mkBvSignedDivExprNoSimplify)

    @Test
    fun testBvSignedGreater() =
        testOperation(KContext::mkBvSignedGreaterExpr, KContext::mkBvSignedGreaterExprNoSimplify)

    @Test
    fun testBvSignedGreaterOrEqual() =
        testOperation(
            KContext::mkBvSignedGreaterOrEqualExpr,
            KContext::mkBvSignedGreaterOrEqualExprNoSimplify
        )

    @Test
    fun testBvSignedLess() =
        testOperation(KContext::mkBvSignedLessExpr, KContext::mkBvSignedLessExprNoSimplify)

    @Test
    fun testBvSignedLessOrEqual() =
        testOperation(KContext::mkBvSignedLessOrEqualExpr, KContext::mkBvSignedLessOrEqualExprNoSimplify)

    @Test
    fun testBvSignedMod() =
        testOperationNonZeroSecondArg(KContext::mkBvSignedModExpr, KContext::mkBvSignedModExprNoSimplify)

    @Test
    fun testBvSignedRem() =
        testOperationNonZeroSecondArg(KContext::mkBvSignedRemExpr, KContext::mkBvSignedRemExprNoSimplify)

    @Test
    fun testBvSignExtension() {
        repeat(5) {
            val extension = random.nextInt(0..100)
            testOperation(
                { value: KExpr<KBvSort> -> mkBvSignExtensionExpr(extension, value) },
                { value: KExpr<KBvSort> -> mkBvSignExtensionExprNoSimplify(extension, value) }
            )
        }
    }

    @Test
    fun testBvSub() =
        testOperation(KContext::mkBvSubExpr, KContext::mkBvSubExprNoSimplify)

    @Test
    fun testBvSubNoOverflow() =
        testOperation(KContext::mkBvSubNoOverflowExpr, KContext::mkBvSubNoOverflowExprNoSimplify)

    @Test
    fun testBvSubNoUnderflow() =
        testOperation(KContext::mkBvSubNoUnderflowExpr, KContext::mkBvSubNoUnderflowExprNoSimplify)

    @Test
    fun testBvUnsignedDiv() =
        testOperationNonZeroSecondArg(KContext::mkBvUnsignedDivExpr, KContext::mkBvUnsignedDivExprNoSimplify)

    @Test
    fun testBvUnsignedGreater() =
        testOperation(KContext::mkBvUnsignedGreaterExpr, KContext::mkBvUnsignedGreaterExprNoSimplify)

    @Test
    fun testBvUnsignedGreaterOrEqual() =
        testOperation(
            KContext::mkBvUnsignedGreaterOrEqualExpr,
            KContext::mkBvUnsignedGreaterOrEqualExprNoSimplify
        )

    @Test
    fun testBvUnsignedLess() =
        testOperation(KContext::mkBvUnsignedLessExpr, KContext::mkBvUnsignedLessExprNoSimplify)

    @Test
    fun testBvUnsignedLessOrEqual() =
        testOperation(KContext::mkBvUnsignedLessOrEqualExpr, KContext::mkBvUnsignedLessOrEqualExprNoSimplify)

    @Test
    fun testBvUnsignedRem() =
        testOperationNonZeroSecondArg(KContext::mkBvUnsignedRemExpr, KContext::mkBvUnsignedRemExprNoSimplify)

    @Test
    fun testBvXNor() =
        testOperation(KContext::mkBvXNorExpr, KContext::mkBvXNorExprNoSimplify)

    @Test
    fun testBvXor() =
        testOperation(KContext::mkBvXorExpr, KContext::mkBvXorExprNoSimplify)

    @Test
    fun testBvZeroExtension() {
        repeat(5) {
            val extension = random.nextInt(0..100)
            testOperation(
                { value: KExpr<KBvSort> -> mkBvZeroExtensionExpr(extension, value) },
                { value: KExpr<KBvSort> -> mkBvZeroExtensionExprNoSimplify(extension, value) }
            )
        }
    }

    private fun <S : KBvSort, T : KSort> testOperation(
        operation: KContext.(KExpr<S>) -> KExpr<T>,
        operationNoSimplify: KContext.(KExpr<S>) -> KExpr<T>,
        mkValue: KContext.(S) -> KExpr<S> = { mkConst("x", it) }
    ) = runTest { sort: S, checker ->
        val value = mkValue(sort)
        val unsimplifiedExpr = operationNoSimplify(value)
        val simplifiedExpr = operation(value)
        checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) { "$value" }
    }

    private fun <S : KBvSort, T : KSort> testOperation(
        operation: KContext.(KExpr<S>, KExpr<S>) -> KExpr<T>,
        operationNoSimplify: KContext.(KExpr<S>, KExpr<S>) -> KExpr<T>,
        mkLhs: KContext.(S) -> KExpr<S> = { mkConst("a", it) },
        mkRhs: KContext.(S) -> KExpr<S> = { mkConst("b", it) },
    ) = runTest { sort: S, checker ->
        val a = mkLhs(sort)
        val b = mkRhs(sort)
        val unsimplifiedExpr = operationNoSimplify(a, b)
        val simplifiedExpr = operation(a, b)
        checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) { "$a, $b" }
    }

    private fun <S : KBvSort, T : KSort> testOperationNonZeroSecondArg(
        operation: KContext.(KExpr<S>, KExpr<S>) -> KExpr<T>,
        operationNoSimplify: KContext.(KExpr<S>, KExpr<S>) -> KExpr<T>,
        mkLhs: KContext.(S) -> KExpr<S> = { mkConst("a", it) },
        mkRhs: KContext.(S) -> KExpr<S> = { mkConst("b", it) },
    ) = runTest { sort: S, checker ->
        val a = mkLhs(sort)
        val b = mkRhs(sort)
        val unsimplifiedExpr = operationNoSimplify(a, b)
        val simplifiedExpr = operation(a, b)
        val zero: KExpr<S> = bvZero(sort.sizeBits).uncheckedCast()
        val assumptions = listOf(!(b eq zero))
        checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr, assumptions) { "$a, $b" }
    }

    private fun <S : KBvSort, T : KSort> testOperation(
        operation: KContext.(KExpr<S>, KExpr<S>, Boolean) -> KExpr<T>,
        operationNoSimplify: KContext.(KExpr<S>, KExpr<S>, Boolean) -> KExpr<T>,
        mkLhs: KContext.(S) -> KExpr<S> = { mkConst("a", it) },
        mkRhs: KContext.(S) -> KExpr<S> = { mkConst("b", it) },
    ) = runTest { sort: S, checker ->
        val a = mkLhs(sort)
        val b = mkRhs(sort)
        listOf(true, false).forEach { c ->
            val unsimplifiedExpr = operationNoSimplify(a, b, c)
            val simplifiedExpr = operation(a, b, c)
            checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) {
                "$a, $b, $c"
            }
        }
    }

    private fun <S : KBvSort> runTest(test: KContext.(S, TestRunner) -> Unit) {
        val ctx = KContext(simplificationMode = KContext.SimplificationMode.SIMPLIFY)
        val sort: S = ctx.mkBvSort(BV_SIZE).uncheckedCast()
        val checker = TestRunner(ctx)
        ctx.test(sort, checker)
    }

    internal class TestRunner(private val ctx: KContext) {
        fun <T : KSort> check(
            unsimplifiedExpr: KExpr<T>,
            simplifiedExpr: KExpr<T>,
            assumptions: List<KExpr<KBoolSort>> = emptyList(),
            printArgs: () -> String
        ) = KZ3Solver(ctx).use { solver ->
            assumptions.forEach { solver.assert(it) }

            val equivalenceCheck = ctx.mkEq(simplifiedExpr, unsimplifiedExpr)
            solver.assert(ctx.mkNot(equivalenceCheck))

            val status = solver.check()
            assertEquals(KSolverStatus.UNSAT, status, printArgs())
        }
    }

    companion object {
        val random = Random(42)
        val BV_SIZE = 77u
    }
}
