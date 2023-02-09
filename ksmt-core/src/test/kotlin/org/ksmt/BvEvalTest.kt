package org.ksmt

import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KSort
import org.ksmt.utils.uncheckedCast
import kotlin.random.nextInt
import kotlin.test.Test
import kotlin.test.assertEquals

@Execution(ExecutionMode.CONCURRENT)
class BvEvalTest : ExpressionEvalTest() {

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

    @Test
    fun testBvCreateFromNumber() {
        testBvCreate<Byte>(8u, { random.nextInt().toByte() }, KContext::mkBv, KContext::mkBv)
        testBvCreate<Short>(16u, { random.nextInt().toShort() }, KContext::mkBv, KContext::mkBv)
        testBvCreate<Int>(32u, { random.nextInt() }, KContext::mkBv, KContext::mkBv)
        testBvCreate<Long>(64u, { random.nextLong() }, KContext::mkBv, KContext::mkBv)
    }

    private inline fun <reified T : Number> testBvCreate(
        size: UInt,
        valueGen: () -> T,
        specializedBuilder: KContext.(T) -> KExpr<*>,
        genericBuilder: KContext.(T, UInt) -> KExpr<*>
    ) = KContext().run {
        repeat(100) {
            val value = valueGen()
            val createSpecialized = specializedBuilder(value)
            val createGeneric = genericBuilder(value, size)
            assertEquals(createSpecialized, createGeneric)
        }
    }

    private fun <S : KBvSort, T : KSort> testOperation(
        size: Int,
        operation: KContext.(KExpr<S>) -> KExpr<T>
    ) = runTest(size) { sort: S, checker ->
        randomBvValues(sort).forEach { value ->
            val expr = operation(value)
            checker.check(expr){ "$value" }
        }
    }

    private fun <S : KBvSort, T : KSort> testOperation(
        size: Int,
        operation: KContext.(KExpr<S>, KExpr<S>) -> KExpr<T>
    ) = runTest(size) { sort: S, checker ->
        randomBvValues(sort).forEach { a ->
            randomBvValues(sort).forEach { b ->
                val expr = operation(a, b)
                checker.check(expr){ "$a, $b" }
            }
        }
    }

    private fun <S : KBvSort, T : KSort> testOperationNonZeroSecondArg(
        size: Int,
        operation: KContext.(KExpr<S>, KExpr<S>) -> KExpr<T>
    ) = runTest(size) { sort: S, checker ->
        randomBvValues(sort).forEach { a ->
            randomBvNonZeroValues(sort).forEach { b ->
                val expr = operation(a, b)
                checker.check(expr) { "$a, $b" }
            }
        }
    }

    private fun <S : KBvSort> runTest(size: Int, test: KContext.(S, TestRunner) -> Unit) = runTest(
        mkSort = { mkBvSort(size.toUInt()).uncheckedCast() },
        test = test.uncheckedCast()
    )

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
