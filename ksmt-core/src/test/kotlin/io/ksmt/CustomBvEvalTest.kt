package io.ksmt

import io.ksmt.KContext.SimplificationMode.SIMPLIFY
import io.ksmt.decl.KDecl
import io.ksmt.expr.KBitVec1Value
import io.ksmt.expr.KBitVecNumberValue
import io.ksmt.expr.KBitVecValue
import io.ksmt.expr.KExpr
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KSort
import io.ksmt.utils.uncheckedCast
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import kotlin.random.nextInt
import kotlin.test.assertEquals

@Execution(ExecutionMode.CONCURRENT)
class CustomBvEvalTest : ExpressionEvalTest() {

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvAdd(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvAddExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvAnd(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvAndExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvArithShiftRight(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvArithShiftRightExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvConcat(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvConcatExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvExtract(sizeBits: Int) {
        repeat(5) {
            val high = random.nextInt(0 until sizeBits)
            repeat(5) {
                val low = random.nextInt(0..high)
                testOperation(
                    sizeBits
                ) { value: KExpr<KBvSort> -> mkBvExtractExpr(high, low, value) }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvLogicalShiftRight(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvLogicalShiftRightExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvMul(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvMulExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvNAnd(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvNAndExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvNegation(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvNegationExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvNor(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvNorExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvNot(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvNotExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvOr(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvOrExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvReductionAnd(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvReductionAndExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvReductionOr(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvReductionOrExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvRepeat(sizeBits: Int) {
        repeat(5) {
            val repetitions = random.nextInt(1..10)
            testOperation(
                sizeBits
            ) { value: KExpr<KBvSort> -> mkBvRepeatExpr(repetitions, value) }
        }
    }

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvRotateLeft(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvRotateLeftExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvRotateLeftIndexed(sizeBits: Int) {
        repeat(5) {
            val rotation = random.nextInt(0..1024)
            testOperation(
                sizeBits
            ) { value: KExpr<KBvSort> -> mkBvRotateLeftIndexedExpr(rotation, value) }
        }
    }

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvRotateRight(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvRotateRightExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvRotateRightIndexed(sizeBits: Int) {
        repeat(5) {
            val rotation = random.nextInt(0..1024)
            testOperation(
                sizeBits
            ) { value: KExpr<KBvSort> -> mkBvRotateRightIndexedExpr(rotation, value) }
        }
    }

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvShiftLeft(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvShiftLeftExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedDiv(sizeBits: Int) =
        testOperationNonZeroSecondArg(sizeBits, KContext::mkBvSignedDivExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedGreater(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvSignedGreaterExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedGreaterOrEqual(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvSignedGreaterOrEqualExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedLess(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvSignedLessExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedLessOrEqual(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvSignedLessOrEqualExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedMod(sizeBits: Int) =
        testOperationNonZeroSecondArg(sizeBits, KContext::mkBvSignedModExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedRem(sizeBits: Int) =
        testOperationNonZeroSecondArg(sizeBits, KContext::mkBvSignedRemExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignExtension(sizeBits: Int) {
        repeat(5) {
            val extension = random.nextInt(0..100)
            testOperation(
                sizeBits
            ) { value: KExpr<KBvSort> -> mkBvSignExtensionExpr(extension, value) }
        }
    }

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSub(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvSubExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvUnsignedDiv(sizeBits: Int) =
        testOperationNonZeroSecondArg(sizeBits, KContext::mkBvUnsignedDivExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvUnsignedGreater(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvUnsignedGreaterExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvUnsignedGreaterOrEqual(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvUnsignedGreaterOrEqualExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvUnsignedLess(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvUnsignedLessExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvUnsignedLessOrEqual(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvUnsignedLessOrEqualExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvUnsignedRem(sizeBits: Int) =
        testOperationNonZeroSecondArg(sizeBits, KContext::mkBvUnsignedRemExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvXNor(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvXNorExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvXor(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvXorExpr)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvZeroExtension(sizeBits: Int) {
        repeat(5) {
            val extension = random.nextInt(0..100)
            testOperation(
                sizeBits
            ) { value: KExpr<KBvSort> -> mkBvZeroExtensionExpr(extension, value) }
        }
    }

    private fun <S : KBvSort, T : KSort> testOperation(
        size: Int,
        operation: KContext.(KExpr<S>) -> KExpr<T>
    ) = runTest(size) { sort: S ->
        randomBvValues(sort).forEach { value ->
            check(value, operation)
        }
    }

    private fun <S : KBvSort, T : KSort> testOperation(
        size: Int,
        operation: KContext.(KExpr<S>, KExpr<S>) -> KExpr<T>
    ) = runTest(size) { sort: S ->
        randomBvValues(sort).forEach { a ->
            randomBvValues(sort).forEach { b ->
                check(a, b, operation)
            }
        }
    }

    private fun <S : KBvSort, T : KSort> testOperation(
        size: Int,
        operation: KContext.(KExpr<S>, KExpr<S>, Boolean) -> KExpr<T>
    ) = runTest(size) { sort: S ->
        val boolValues = listOf(true, false)
        randomBvValues(sort).forEach { a ->
            randomBvValues(sort).forEach { b ->
                boolValues.forEach { c ->
                    check(a, b) { x, y -> operation(x, y, c) }
                }
            }
        }
    }

    private fun <S : KBvSort, T : KSort> testOperationNonZeroSecondArg(
        size: Int,
        operation: KContext.(KExpr<S>, KExpr<S>) -> KExpr<T>
    ) = runTest(size) { sort: S ->
        randomBvValues(sort).forEach { a ->
            randomBvNonZeroValues(sort).forEach { b ->
                check(a, b, operation)
            }
        }
    }

    private fun <S : KBvSort, T : KSort> KContext.check(
        value: KBitVecValue<S>,
        op: KContext.(KExpr<S>) -> KExpr<T>
    ) {
        val expected = op(value)
        value.customValues().forEach {
            checkValue(expected, op(it), expected != value)
        }
    }

    private fun <S : KBvSort, T : KSort> KContext.check(
        a: KBitVecValue<S>,
        b: KBitVecValue<S>,
        op: KContext.(KExpr<S>, KExpr<S>) -> KExpr<T>
    ) {
        val expected = op(a, b)
        a.customValues().forEach { ac ->
            b.customValues().forEach { bc ->
                checkValue(expected, op(a, bc), expected != a && expected != b)
                checkValue(expected, op(ac, b), expected != a && expected != b)
                checkValue(expected, op(ac, bc), expected != a && expected != b)
            }
        }
    }

    private fun <T : KSort> checkValue(expected: KExpr<T>, actual: KExpr<T>, reportError: Boolean) {
        if (reportError) {
            assertEquals(expected, actual)
        }
    }

    private fun <S : KBvSort> runTest(size: Int, test: KContext.(S) -> Unit) {
        val ctx = KContext(simplificationMode = SIMPLIFY)
        val sort = ctx.mkBvSort(size.toUInt())
        test(ctx, sort.uncheckedCast())
    }

    private fun <S : KBvSort> KBitVecValue<S>.customValues(): List<KBitVecValue<S>> = buildList {
        if (this@customValues is KBitVec1Value) {
            add(CustomBv1Value(ctx, value).uncheckedCast())
            return@buildList
        }

        add(CustomBvValue(ctx, sort, stringValue))

        if (this@customValues is KBitVecNumberValue<S, *>) {
            add(CustomBvNumericValue(ctx, sort, numberValue))
        }
    }

    private class CustomBvValue<S : KBvSort>(
        ctx: KContext,
        override val sort: S,
        override val stringValue: String,
    ) : KBitVecValue<S>(ctx) {
        override val decl: KDecl<S>
            get() = error("Should not be executed")

        override fun accept(transformer: KTransformerBase): KExpr<S> {
            error("Should not be executed")
        }
    }

    private class CustomBv1Value(ctx: KContext, val value: Boolean) : KBitVecValue<KBv1Sort>(ctx) {
        override val stringValue: String
            get() = ctx.mkBv(value).stringValue

        override val decl: KDecl<KBv1Sort>
            get() = error("Should not be executed")

        override val sort: KBv1Sort
            get() = ctx.bv1Sort

        override fun accept(transformer: KTransformerBase): KExpr<KBv1Sort> {
            error("Should not be executed")
        }
    }

    private class CustomBvNumericValue<S : KBvSort, V : Number>(
        ctx: KContext,
        override val sort: S,
        override val numberValue: V
    ) : KBitVecNumberValue<S, V>(ctx) {
        override val decl: KDecl<S>
            get() = error("Should not be executed")

        override fun accept(transformer: KTransformerBase): KExpr<S> {
            error("Should not be executed")
        }
    }

    companion object {
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
