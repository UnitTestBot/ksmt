package io.ksmt

import io.ksmt.expr.KExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvMulNoOverflowExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvMulNoUnderflowExpr
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KSort
import io.ksmt.utils.BvUtils.bvMaxValueUnsigned
import io.ksmt.utils.BvUtils.bvOne
import io.ksmt.utils.BvUtils.bvValue
import io.ksmt.utils.BvUtils.bvZero
import io.ksmt.utils.uncheckedCast
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import kotlin.random.nextInt
import kotlin.test.Ignore

@Execution(ExecutionMode.CONCURRENT)
class BvSimplifyTest: ExpressionSimplifyTest() {

    @Test
    fun testBvAdd() = testOperation(
        KContext::mkBvAddExpr,
        KContext::mkBvAddExprNoSimplify,
        mkSpecialValues = {
            listOf(
                bvZero(it.sizeBits).uncheckedCast(),
                bvOne(it.sizeBits).uncheckedCast()
            )
        }
    )

    @Test
    fun testBvAddNoOverflow() =
        testOperation(KContext::mkBvAddNoOverflowExpr, KContext::mkBvAddNoOverflowExprNoSimplify)

    @Test
    fun testBvAddNoUnderflow() =
        testOperation(KContext::mkBvAddNoUnderflowExpr, KContext::mkBvAddNoUnderflowExprNoSimplify)

    @Test
    fun testBvAnd() = testOperation(
        KContext::mkBvAndExpr,
        KContext::mkBvAndExprNoSimplify,
        mkSpecialValues = {
            listOf(
                bvZero(it.sizeBits).uncheckedCast(),
                bvMaxValueUnsigned(it.sizeBits).uncheckedCast(),
                bvOne(it.sizeBits).uncheckedCast()
            )
        }
    )

    @Test
    fun testBvArithShiftRight() = testOperation(
        KContext::mkBvArithShiftRightExpr,
        KContext::mkBvArithShiftRightExprNoSimplify,
        mkSpecialValues = {
            listOf(
                bvValue(it.sizeBits, 3).uncheckedCast(),
                bvValue(it.sizeBits, BV_SIZE.toInt() + 5).uncheckedCast(),
                bvZero(it.sizeBits).uncheckedCast(),
                mkConst("a", it)  // same as lhs
            )
        }
    )

    @Test
    @Suppress("UNCHECKED_CAST")
    fun testBvConcat() = testOperation(
        KContext::mkBvConcatExpr,
        KContext::mkBvConcatExprNoSimplify,
        mkSpecialValues = {
            listOf(
                bvZero(BV_SIZE / 2u) as KExpr<KBvSort>,
                mkBvConcatExprNoSimplify(
                    bvZero(BV_SIZE / 2u),
                    mkConst("c", mkBvSort(BV_SIZE / 2u))
                ),
                mkBvConcatExprNoSimplify(
                    mkConst("d", mkBvSort(BV_SIZE / 2u)),
                    bvZero(BV_SIZE / 2u)
                )
            )
        }
    )

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
                    { value: KExpr<KBvSort> -> mkBvExtractExprNoSimplify(high, low, value) },
                    mkSpecialValues = {
                        listOf(
                            mkBvExtractExprNoSimplify(
                                high = BV_SIZE.toInt() + 5,
                                low = 5,
                                mkConst("y", mkBvSort(BV_SIZE * 2u))
                            )
                        )
                    }
                )
            }
        }
    }

    @Test
    fun testBvLogicalShiftRight() = testOperation(
        KContext::mkBvLogicalShiftRightExpr,
        KContext::mkBvLogicalShiftRightExprNoSimplify,
        mkSpecialValues = {
            listOf(
                bvValue(it.sizeBits, 3).uncheckedCast(),
                bvValue(it.sizeBits, BV_SIZE.toInt() + 5).uncheckedCast(),
                bvZero(it.sizeBits).uncheckedCast(),
                mkConst("a", it)  // same as lhs
            )
        }
    )

    @Test
    fun testBvMul() = testOperation(
        KContext::mkBvMulExpr,
        KContext::mkBvMulExprNoSimplify,
        mkSpecialValues = {
            listOf(
                bvZero(it.sizeBits).uncheckedCast(),
                bvOne(it.sizeBits).uncheckedCast(),
                bvValue(it.sizeBits, 2).uncheckedCast(),
                bvValue(it.sizeBits, -1).uncheckedCast(),
            )
        }
    )

    @Test
    fun testBvMulNoOverflow() =
        testOperation(KContext::mkBvMulNoOverflowExpr, KContext::mkBvMulNoOverflowExprNoSimplify)

    @Test
    fun testBvMulNoUnderflow() =
        testOperation(KContext::mkBvMulNoUnderflowExpr, KContext::mkBvMulNoUnderflowExprNoSimplify)

    @Test
    fun testBvMulNoOverflowUnsignedRewrite() =
        testOperation(
            { l, r -> rewriteBvMulNoOverflowExpr(l, r, isSigned = false) },
            KContext::mkBvMulNoOverflowUnsignedExprNoSimplify
        )

    @Ignore // Slow on Z3 solver. Check manually with Bitwuzla.
    @Test
    fun testBvMulNoOverflowSignedRewrite() =
        testOperation(
            { l, r -> rewriteBvMulNoOverflowExpr(l, r, isSigned = true) },
            KContext::mkBvMulNoOverflowSignedExprNoSimplify
        )

    @Ignore // Slow on Z3 solver. Check manually with Bitwuzla.
    @Test
    fun testBvMulNoUnderflowRewrite() =
        testOperation(
            { l, r -> rewriteBvMulNoUnderflowExpr(l, r) },
            KContext::mkBvMulNoUnderflowExprNoSimplify
        )

    @Test
    fun testBvNAnd() =
        testOperation(KContext::mkBvNAndExpr, KContext::mkBvNAndExprNoSimplify)

    @Test
    fun testBvNegation() = testOperation(
        KContext::mkBvNegationExpr,
        KContext::mkBvNegationExprNoSimplify,
        mkSpecialValues = {
            listOf(
                mkBvAddExprNoSimplify(
                    mkConst("y", it),
                    bvOne(it.sizeBits).uncheckedCast()
                ),
                mkBvAddExprNoSimplify(
                    mkBvNegationExprNoSimplify(mkConst("y", it)),
                    mkConst("z", it),
                )
            )
        }
    )

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
    fun testBvOr() = testOperation(
        KContext::mkBvOrExpr,
        KContext::mkBvOrExprNoSimplify,
        mkSpecialValues = {
            listOf(
                bvZero(it.sizeBits).uncheckedCast(),
                bvMaxValueUnsigned(it.sizeBits).uncheckedCast(),
                bvOne(it.sizeBits).uncheckedCast()
            )
        }
    )

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
    fun testBvShiftLeft() = testOperation(
        KContext::mkBvShiftLeftExpr,
        KContext::mkBvShiftLeftExprNoSimplify,
        mkSpecialValues = {
            listOf(
                bvValue(it.sizeBits, 3).uncheckedCast(),
                bvValue(it.sizeBits, BV_SIZE.toInt() + 5).uncheckedCast(),
                bvZero(it.sizeBits).uncheckedCast(),
                mkConst("a", it)  // same as lhs
            )
        }
    )

    @Test
    fun testBvSignedDiv() = testOperation(
        KContext::mkBvSignedDivExpr,
        KContext::mkBvSignedDivExprNoSimplify,
        mkSpecialValues = {
            listOf(
                bvZero(it.sizeBits).uncheckedCast(),
                bvOne(it.sizeBits).uncheckedCast(),
                bvValue(it.sizeBits, -1).uncheckedCast(),
                bvValue(it.sizeBits, 1024).uncheckedCast(),
            )
        }
    )

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
    fun testBvSignedMod() = testOperation(
        KContext::mkBvSignedModExpr,
        KContext::mkBvSignedModExprNoSimplify,
        mkSpecialValues = {
            listOf(
                bvZero(it.sizeBits).uncheckedCast(),
                bvOne(it.sizeBits).uncheckedCast(),
                bvValue(it.sizeBits, -1).uncheckedCast(),
                bvValue(it.sizeBits, 1024).uncheckedCast(),
            )
        }
    )

    @Test
    fun testBvSignedRem() = testOperation(
        KContext::mkBvSignedRemExpr,
        KContext::mkBvSignedRemExprNoSimplify,
        mkSpecialValues = {
            listOf(
                bvZero(it.sizeBits).uncheckedCast(),
                bvOne(it.sizeBits).uncheckedCast(),
                bvValue(it.sizeBits, -1).uncheckedCast(),
                bvValue(it.sizeBits, 1024).uncheckedCast(),
            )
        }
    )

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
    fun testBvUnsignedDiv() = testOperation(
        KContext::mkBvUnsignedDivExpr,
        KContext::mkBvUnsignedDivExprNoSimplify,
        mkSpecialValues = {
            listOf(
                bvZero(it.sizeBits).uncheckedCast(),
                bvOne(it.sizeBits).uncheckedCast(),
                bvValue(it.sizeBits, -1).uncheckedCast(),
                bvValue(it.sizeBits, 1024).uncheckedCast(),
            )
        }
    )

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
    fun testBvUnsignedRem() = testOperation(
        KContext::mkBvUnsignedRemExpr,
        KContext::mkBvUnsignedRemExprNoSimplify,
        mkSpecialValues = {
            listOf(
                bvZero(it.sizeBits).uncheckedCast(),
                bvOne(it.sizeBits).uncheckedCast(),
                bvValue(it.sizeBits, -1).uncheckedCast(),
                bvValue(it.sizeBits, 1024).uncheckedCast(),
            )
        }
    )

    @Test
    fun testBvXNor() =
        testOperation(KContext::mkBvXNorExpr, KContext::mkBvXNorExprNoSimplify)

    @Test
    fun testBvXor() = testOperation(
        KContext::mkBvXorExpr,
        KContext::mkBvXorExprNoSimplify,
        mkSpecialValues = {
            listOf(
                bvZero(it.sizeBits).uncheckedCast(),
                bvMaxValueUnsigned(it.sizeBits).uncheckedCast(),
                bvOne(it.sizeBits).uncheckedCast()
            )
        }
    )

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

    @JvmName("testUnary")
    private fun <S : KBvSort, T : KSort> testOperation(
        operation: KContext.(KExpr<S>) -> KExpr<T>,
        operationNoSimplify: KContext.(KExpr<S>) -> KExpr<T>,
        mkSpecialValues: KContext.(S) -> List<KExpr<S>> = { emptyList() },
    ) = runTest { sort: S, checker ->
        val x = mkConst("x", sort)
        val specialValues = mkSpecialValues(sort)

        (listOf(x) + specialValues).forEach { value ->
            val unsimplifiedExpr = operationNoSimplify(value)
            val simplifiedExpr = operation(value)
            checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) { "$value" }
        }
    }

    @JvmName("testBinaryNested")
    private fun <S : KBvSort> testOperation(
        operation: KContext.(KExpr<S>, KExpr<S>) -> KExpr<S>,
        operationNoSimplify: KContext.(KExpr<S>, KExpr<S>) -> KExpr<S>,
        mkSpecialValues: KContext.(S) -> List<KExpr<S>> = { emptyList() }
    ) = runTest { sort: S, checker ->
        val a = mkConst("a", sort)
        val b = mkConst("b", sort)
        val specialValues = mkSpecialValues(sort)

        (listOf(a) + specialValues).forEach { lhs ->
            (listOf(b) + specialValues).forEach { rhs ->
                run {
                    val unsimplifiedExpr = operationNoSimplify(lhs, rhs)
                    val simplifiedExpr = operation(lhs, rhs)
                    checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) {
                        "$lhs, $rhs"
                    }
                }
                run {
                    val someValue: KExpr<S> = bvValue(sort.sizeBits, 17).uncheckedCast()
                    val nestedOperation = operationNoSimplify(someValue, rhs)
                    val unsimplifiedExpr = operationNoSimplify(lhs, nestedOperation)
                    val simplifiedExpr = operation(lhs, nestedOperation)
                    checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) {
                        "$lhs, $nestedOperation"
                    }
                }
            }
        }
    }

    @JvmName("testBinaryNotNested")
    private fun <S : KBvSort, T : KSort> testOperation(
        operation: KContext.(KExpr<S>, KExpr<S>) -> KExpr<T>,
        operationNoSimplify: KContext.(KExpr<S>, KExpr<S>) -> KExpr<T>,
        mkSpecialValues: KContext.(S) -> List<KExpr<S>> = { emptyList() }
    ) = runTest { sort: S, checker ->
        val a = mkConst("a", sort)
        val b = mkConst("b", sort)
        val specialValues = mkSpecialValues(sort)

        (listOf(a) + specialValues).forEach { lhs ->
            (listOf(b) + specialValues).forEach { rhs ->
                val unsimplifiedExpr = operationNoSimplify(lhs, rhs)
                val simplifiedExpr = operation(lhs, rhs)
                checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) {
                    "$lhs, $rhs"
                }
            }
        }
    }

    @JvmName("testTernary")
    private fun <S : KBvSort, T : KSort> testOperation(
        operation: KContext.(KExpr<S>, KExpr<S>, Boolean) -> KExpr<T>,
        operationNoSimplify: KContext.(KExpr<S>, KExpr<S>, Boolean) -> KExpr<T>
    ) = runTest { sort: S, checker ->
        val a = mkConst("a", sort)
        val b = mkConst("b", sort)
        listOf(true, false).forEach { c ->
            val unsimplifiedExpr = operationNoSimplify(a, b, c)
            val simplifiedExpr = operation(a, b, c)
            checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) {
                "$a, $b, $c"
            }
        }
    }

    private fun <S : KBvSort> runTest(test: KContext.(S, TestRunner) -> Unit) = runTest(
        mkSort = { mkBvSort(BV_SIZE).uncheckedCast() },
        test = test
    )

    companion object {
        const val BV_SIZE = 77u
    }
}

private fun <T : KBvSort> KContext.mkBvMulNoOverflowSignedExprNoSimplify(
    arg0: KExpr<T>,
    arg1: KExpr<T>
) = mkBvMulNoOverflowExprNoSimplify(arg0, arg1, isSigned = true)

private fun <T : KBvSort> KContext.mkBvMulNoOverflowUnsignedExprNoSimplify(
    arg0: KExpr<T>,
    arg1: KExpr<T>
) = mkBvMulNoOverflowExprNoSimplify(arg0, arg1, isSigned = false)
