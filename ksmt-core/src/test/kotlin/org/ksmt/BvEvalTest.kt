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
    fun testBvAdd(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvAddExpr, KContext::mkBvAddExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvAddNoOverflow(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvAddNoOverflowExpr, KContext::mkBvAddNoOverflowExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvAddNoUnderflow(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvAddNoUnderflowExpr, KContext::mkBvAddNoUnderflowExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvAnd(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvAndExpr, KContext::mkBvAndExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvArithShiftRight(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvArithShiftRightExpr, KContext::mkBvArithShiftRightExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvConcat(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvConcatExpr, KContext::mkBvConcatExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvDivNoOverflow(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvDivNoOverflowExpr, KContext::mkBvDivNoOverflowExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvExtract(sizeBits: Int) {
        repeat(5) {
            val high = random.nextInt(0 until sizeBits)
            repeat(5) {
                val low = random.nextInt(0..high)
                testOperation(
                    sizeBits,
                    { value: KExpr<KBvSort> -> mkBvExtractExpr(high, low, value) },
                    { value: KExpr<KBvSort> -> mkBvExtractExprNoSimplify(high, low, value) }
                )
            }
        }
    }

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvLogicalShiftRight(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvLogicalShiftRightExpr, KContext::mkBvLogicalShiftRightExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvMul(sizeBits: Int) = testOperation(sizeBits, KContext::mkBvMulExpr, KContext::mkBvMulExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvMulNoOverflow(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvMulNoOverflowExpr, KContext::mkBvMulNoOverflowExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvMulNoUnderflow(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvMulNoUnderflowExpr, KContext::mkBvMulNoUnderflowExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvNAnd(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvNAndExpr, KContext::mkBvNAndExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvNegation(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvNegationExpr, KContext::mkBvNegationExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvNegationNoOverflow(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvNegationNoOverflowExpr, KContext::mkBvNegationNoOverflowExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvNor(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvNorExpr, KContext::mkBvNorExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvNot(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvNotExpr, KContext::mkBvNotExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvOr(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvOrExpr, KContext::mkBvOrExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvReductionAnd(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvReductionAndExpr, KContext::mkBvReductionAndExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvReductionOr(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvReductionOrExpr, KContext::mkBvReductionOrExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvRepeat(sizeBits: Int) {
        repeat(5) {
            val repetitions = random.nextInt(1..10)
            testOperation(
                sizeBits,
                { value: KExpr<KBvSort> -> mkBvRepeatExpr(repetitions, value) },
                { value: KExpr<KBvSort> -> mkBvRepeatExprNoSimplify(repetitions, value) }
            )
        }
    }

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvRotateLeft(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvRotateLeftExpr, KContext::mkBvRotateLeftExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvRotateLeftIndexed(sizeBits: Int) {
        repeat(5) {
            val rotation = random.nextInt(0..1024)
            testOperation(
                sizeBits,
                { value: KExpr<KBvSort> -> mkBvRotateLeftIndexedExpr(rotation, value) },
                { value: KExpr<KBvSort> -> mkBvRotateLeftIndexedExprNoSimplify(rotation, value) }
            )
        }
    }

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvRotateRight(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvRotateRightExpr, KContext::mkBvRotateRightExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvRotateRightIndexed(sizeBits: Int) {
        repeat(5) {
            val rotation = random.nextInt(0..1024)
            testOperation(
                sizeBits,
                { value: KExpr<KBvSort> -> mkBvRotateRightIndexedExpr(rotation, value) },
                { value: KExpr<KBvSort> -> mkBvRotateRightIndexedExprNoSimplify(rotation, value) }
            )
        }
    }

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvShiftLeft(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvShiftLeftExpr, KContext::mkBvShiftLeftExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedDiv(sizeBits: Int) =
        testOperationNonZeroSecondArg(sizeBits, KContext::mkBvSignedDivExpr, KContext::mkBvSignedDivExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedGreater(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvSignedGreaterExpr, KContext::mkBvSignedGreaterExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedGreaterOrEqual(sizeBits: Int) =
        testOperation(
            sizeBits,
            KContext::mkBvSignedGreaterOrEqualExpr,
            KContext::mkBvSignedGreaterOrEqualExprNoSimplify
        )

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedLess(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvSignedLessExpr, KContext::mkBvSignedLessExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedLessOrEqual(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvSignedLessOrEqualExpr, KContext::mkBvSignedLessOrEqualExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedMod(sizeBits: Int) =
        testOperationNonZeroSecondArg(sizeBits, KContext::mkBvSignedModExpr, KContext::mkBvSignedModExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignedRem(sizeBits: Int) =
        testOperationNonZeroSecondArg(sizeBits, KContext::mkBvSignedRemExpr, KContext::mkBvSignedRemExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSignExtension(sizeBits: Int) {
        repeat(5) {
            val extension = random.nextInt(0..100)
            testOperation(
                sizeBits,
                { value: KExpr<KBvSort> -> mkBvSignExtensionExpr(extension, value) },
                { value: KExpr<KBvSort> -> mkBvSignExtensionExprNoSimplify(extension, value) }
            )
        }
    }

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSub(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvSubExpr, KContext::mkBvSubExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSubNoOverflow(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvSubNoOverflowExpr, KContext::mkBvSubNoOverflowExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvSubNoUnderflow(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvSubNoUnderflowExpr, KContext::mkBvSubNoUnderflowExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvUnsignedDiv(sizeBits: Int) =
        testOperationNonZeroSecondArg(sizeBits, KContext::mkBvUnsignedDivExpr, KContext::mkBvUnsignedDivExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvUnsignedGreater(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvUnsignedGreaterExpr, KContext::mkBvUnsignedGreaterExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvUnsignedGreaterOrEqual(sizeBits: Int) =
        testOperation(
            sizeBits,
            KContext::mkBvUnsignedGreaterOrEqualExpr,
            KContext::mkBvUnsignedGreaterOrEqualExprNoSimplify
        )

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvUnsignedLess(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvUnsignedLessExpr, KContext::mkBvUnsignedLessExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvUnsignedLessOrEqual(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvUnsignedLessOrEqualExpr, KContext::mkBvUnsignedLessOrEqualExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvUnsignedRem(sizeBits: Int) =
        testOperationNonZeroSecondArg(sizeBits, KContext::mkBvUnsignedRemExpr, KContext::mkBvUnsignedRemExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvXNor(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvXNorExpr, KContext::mkBvXNorExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvXor(sizeBits: Int) =
        testOperation(sizeBits, KContext::mkBvXorExpr, KContext::mkBvXorExprNoSimplify)

    @ParameterizedTest
    @MethodSource("bvSizes")
    fun testBvZeroExtension(sizeBits: Int) {
        repeat(5) {
            val extension = random.nextInt(0..100)
            testOperation(
                sizeBits,
                { value: KExpr<KBvSort> -> mkBvZeroExtensionExpr(extension, value) },
                { value: KExpr<KBvSort> -> mkBvZeroExtensionExprNoSimplify(extension, value) }
            )
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
        operation: KContext.(KExpr<S>) -> KExpr<T>,
        operationNoSimplify: KContext.(KExpr<S>) -> KExpr<T>
    ) = runTest(size) { sort: S, checker ->
        randomBvValues(sort).forEach { value ->
            val unsimplifiedExpr = operationNoSimplify(value)
            val simplifiedExpr = operation(value)
            checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) { "$value" }
        }
    }

    private fun <S : KBvSort, T : KSort> testOperation(
        size: Int,
        operation: KContext.(KExpr<S>, KExpr<S>) -> KExpr<T>,
        operationNoSimplify: KContext.(KExpr<S>, KExpr<S>) -> KExpr<T>
    ) = runTest(size) { sort: S, checker ->
        randomBvValues(sort).forEach { a ->
            randomBvValues(sort).forEach { b ->
                val unsimplifiedExpr = operationNoSimplify(a, b)
                val simplifiedExpr = operation(a, b)
                checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) { "$a, $b" }
            }
        }
    }

    private fun <S : KBvSort, T : KSort> testOperation(
        size: Int,
        operation: KContext.(KExpr<S>, KExpr<S>, Boolean) -> KExpr<T>,
        operationNoSimplify: KContext.(KExpr<S>, KExpr<S>, Boolean) -> KExpr<T>
    ) = runTest(size) { sort: S, checker ->
        val boolValues = listOf(true, false)
        randomBvValues(sort).forEach { a ->
            randomBvValues(sort).forEach { b ->
                boolValues.forEach { c ->
                    val unsimplifiedExpr = operationNoSimplify(a, b, c)
                    val simplifiedExpr = operation(a, b, c)
                    checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) {
                        "$a, $b, $c"
                    }
                }
            }
        }
    }

    private fun <S : KBvSort, T : KSort> testOperationNonZeroSecondArg(
        size: Int,
        operation: KContext.(KExpr<S>, KExpr<S>) -> KExpr<T>,
        operationNoSimplify: KContext.(KExpr<S>, KExpr<S>) -> KExpr<T>
    ) = runTest(size) { sort: S, checker ->
        randomBvValues(sort).forEach { a ->
            randomBvNonZeroValues(sort).forEach { b ->
                val unsimplifiedExpr = operationNoSimplify(a, b)
                val simplifiedExpr = operation(a, b)
                checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) { "$a, $b" }
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
