package io.ksmt.solver.z3

import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import io.ksmt.KContext
import io.ksmt.expr.KBitVec1Value
import io.ksmt.expr.KBitVec32Value
import io.ksmt.expr.KBitVec64Value
import io.ksmt.expr.KBitVec8Value
import io.ksmt.expr.KBitVecCustomValue
import io.ksmt.expr.KBitVecValue
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFalse
import io.ksmt.expr.KInt64NumExpr
import io.ksmt.expr.KIntBigNumExpr
import io.ksmt.expr.KIntNumExpr
import io.ksmt.expr.KTrue
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBv16Sort
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBv32Sort
import io.ksmt.sort.KBv64Sort
import io.ksmt.sort.KBv8Sort
import io.ksmt.utils.mkConst
import io.ksmt.utils.toBinary
import kotlin.random.Random
import kotlin.random.nextUInt
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class BitVecTest {
    private var context = KContext()
    private var solver = KZ3Solver(context)

    @BeforeEach
    fun createNewEnvironment() {
        context = KContext()
        solver = KZ3Solver(context)
    }

    @AfterEach
    fun clearResources() {
        solver.close()
    }

    // bv creation
    @Test
    fun testCreateBv1() = with(context) {
        val trueBv = mkBv(value = true)
        val falseBv = mkBv(value = false)
        val trueBvFromString = mkBv(value = "1", sizeBits = 1u)
        val falseBvFromString = mkBv(value = "0", sizeBits = 1u)

        assertTrue(trueBv as KBitVecValue<KBv1Sort> === trueBvFromString)
        assertTrue(falseBv as KBitVecValue<KBv1Sort> === falseBvFromString)

        assertEquals(expected = "1", trueBv.stringValue)
        assertEquals(expected = "0", falseBv.stringValue)

        assertEquals(trueBvFromString.sort, falseBvFromString.sort)
    }

    @Test
    fun testCreateBv8() = with(context) {
        val positiveValue = Random.nextInt(from = 1, until = Byte.MAX_VALUE.toInt()).toByte()
        val negativeValue = Random.nextInt(from = Byte.MIN_VALUE.toInt(), until = 0).toByte()

        val positiveBv = mkBv(positiveValue)
        val negativeBv = mkBv(negativeValue)

        val positiveStringValue = positiveValue.toBinary()
        val negativeStringValue = negativeValue.toBinary()

        val positiveBvFromString = mkBv(value = positiveStringValue, sizeBits = Byte.SIZE_BITS.toUInt())
        val negativeBvFromString = mkBv(value = negativeStringValue, sizeBits = Byte.SIZE_BITS.toUInt())

        assertTrue(positiveBv as KBitVecValue<KBv8Sort> === positiveBvFromString)
        assertTrue(negativeBv as KBitVecValue<KBv8Sort> === negativeBvFromString)

        assertEquals(expected = positiveStringValue, positiveBv.stringValue)
        assertEquals(expected = negativeStringValue, negativeBv.stringValue)

        assertEquals(positiveBvFromString.sort, negativeBvFromString.sort)
    }

    @Test
    fun testCreateBv16() = with(context) {
        val positiveValue = Random.nextInt(from = 1, until = Short.MAX_VALUE.toInt()).toShort()
        val negativeValue = Random.nextInt(from = Short.MIN_VALUE.toInt(), until = 0).toShort()

        val positiveBv = mkBv(positiveValue)
        val negativeBv = mkBv(negativeValue)

        val positiveStringValue = positiveValue.toBinary()
        val negativeStringValue = negativeValue.toBinary()

        val positiveBvFromString = mkBv(value = positiveValue.toBinary(), sizeBits = Short.SIZE_BITS.toUInt())
        val negativeBvFromString = mkBv(value = negativeValue.toBinary(), sizeBits = Short.SIZE_BITS.toUInt())

        assertTrue(positiveBv as KBitVecValue<KBv16Sort> === positiveBvFromString)
        assertTrue(negativeBv as KBitVecValue<KBv16Sort> === negativeBvFromString)

        assertEquals(expected = positiveStringValue, positiveBv.stringValue)
        assertEquals(expected = negativeStringValue, negativeBv.stringValue)

        assertEquals(positiveBvFromString.sort, negativeBvFromString.sort)
    }

    @Test
    fun testCreateBv32() = with(context) {
        val positiveValue = Random.nextInt(from = 1, until = Byte.MAX_VALUE.toInt())
        val negativeValue = Random.nextInt(from = Byte.MIN_VALUE.toInt(), until = 0)

        val positiveBv = mkBv(positiveValue)
        val negativeBv = mkBv(negativeValue)

        val positiveStringValue = positiveValue.toBinary()
        val negativeStringValue = negativeValue.toBinary()

        val positiveBvFromString = mkBv(value = positiveValue.toBinary(), sizeBits = Int.SIZE_BITS.toUInt())
        val negativeBvFromString = mkBv(value = negativeValue.toBinary(), sizeBits = Int.SIZE_BITS.toUInt())

        assertTrue(positiveBv as KBitVecValue<KBv32Sort> === positiveBvFromString)
        assertTrue(negativeBv as KBitVecValue<KBv32Sort> === negativeBvFromString)

        assertEquals(expected = positiveStringValue, positiveBv.stringValue)
        assertEquals(expected = negativeStringValue, negativeBv.stringValue)

        assertEquals(positiveBvFromString.sort, negativeBvFromString.sort)
    }

    @Test
    fun testCreateBv64() = with(context) {
        val (negativeValue, positiveValue) = createTwoRandomLongValues()

        val positiveBv = mkBv(positiveValue)
        val negativeBv = mkBv(negativeValue)

        val positiveStringValue = positiveValue.toBinary()
        val negativeStringValue = negativeValue.toBinary()

        val positiveBvFromString = mkBv(value = positiveValue.toBinary(), sizeBits = Long.SIZE_BITS.toUInt())
        val negativeBvFromString = mkBv(value = negativeValue.toBinary(), sizeBits = Long.SIZE_BITS.toUInt())

        assertTrue(positiveBv as KBitVecValue<KBv64Sort> === positiveBvFromString)
        assertTrue(negativeBv as KBitVecValue<KBv64Sort> === negativeBvFromString)

        assertEquals(expected = positiveStringValue, positiveBv.stringValue)
        assertEquals(expected = negativeStringValue, negativeBv.stringValue)

        assertEquals(positiveBvFromString.sort, negativeBvFromString.sort)
    }

    @Test
    fun testCreateBvCustomSize() = with(context) {
        val sizeBits = Random.nextUInt(from = Long.SIZE_BITS.toUInt() + 1u, until = Long.SIZE_BITS.toUInt() * 2u)

        val nextChar = { if (Random.nextBoolean()) '1' else '0' }
        val positiveValue = "0" + CharArray(sizeBits.toInt() - 1) { nextChar() }.joinToString("")
        val negativeValue = "1" + CharArray(sizeBits.toInt() - 1) { nextChar() }.joinToString("")

        val positiveBvFromString = mkBv(positiveValue, sizeBits) as KBitVecCustomValue
        val negativeBvFromString = mkBv(negativeValue, sizeBits) as KBitVecCustomValue

        assertEquals(positiveBvFromString.stringValue, positiveValue)
        assertEquals(negativeBvFromString.stringValue, negativeValue)

        assertEquals(positiveBvFromString.sort.sizeBits, sizeBits)
        assertEquals(negativeBvFromString.sort.sizeBits, sizeBits)

        assertEquals(expected = positiveValue, positiveBvFromString.stringValue)
        assertEquals(expected = negativeValue, negativeBvFromString.stringValue)

        assertEquals(positiveBvFromString.sort, negativeBvFromString.sort)
    }

    @Test
    fun testCreateBvNarrowingTransformation(): Unit = with(context) {
        val sizeBits = 42u
        val bitvector = mkBv(Long.MAX_VALUE, sizeBits) as KBitVecCustomValue

        assertEquals(bitvector.stringValue, Long.MAX_VALUE.toBinary().takeLast(sizeBits.toInt()))
    }

    @Test
    fun testCreateBvWithGreaterSize(): Unit = with(context) {
        val sizeBits = 100u
        val positiveBv = mkBv(Long.MAX_VALUE, sizeBits) as KBitVecCustomValue
        val negativeBv = mkBv(Long.MIN_VALUE, sizeBits) as KBitVecCustomValue

        val sizeDifference = sizeBits.toInt() - Long.SIZE_BITS

        assertEquals(positiveBv.stringValue.take(sizeDifference), "0".repeat(sizeDifference))
        assertEquals(negativeBv.stringValue.take(sizeDifference), "1".repeat(sizeDifference))

        assertEquals(positiveBv.stringValue.takeLast(Long.SIZE_BITS), Long.MAX_VALUE.toBinary())
        assertEquals(negativeBv.stringValue.takeLast(Long.SIZE_BITS), Long.MIN_VALUE.toBinary())
    }

    @Test
    fun testNotExpr(): Unit = with(context) {
        val (negativeValue, positiveValue) = createTwoRandomLongValues().let {
            it.first.toBinary() to it.second.toBinary()
        }
        val negativeSizeBits = negativeValue.length.toUInt()
        val positiveSizeBits = positiveValue.length.toUInt()

        val negativeBv = mkBv(negativeValue, negativeSizeBits)
        val positiveBv = mkBv(positiveValue, positiveSizeBits)

        val negativeSymbolicValue = negativeBv.sort.mkConst("negative_symbolic_variable")
        val positiveSymbolicValue = positiveBv.sort.mkConst("positive_symbolic_variable")

        solver.assert(mkBvNotExpr(mkBvNotExpr(negativeBv)) eq negativeBv)
        solver.assert(mkBvNotExpr(negativeBv) eq negativeSymbolicValue)

        solver.assert(mkBvNotExpr(mkBvNotExpr(positiveBv)) eq positiveBv)
        solver.assert(mkBvNotExpr(positiveBv) eq positiveSymbolicValue)

        solver.check()

        val actualNegativeValue = (solver.model().eval(negativeSymbolicValue) as KBitVec64Value).numberValue.toBinary()
        val actualPositiveValue = (solver.model().eval(positiveSymbolicValue) as KBitVec64Value).numberValue.toBinary()
        val sizeBits = negativeBv.sort.sizeBits

        val expectedValueTransformation = { stringValue: String ->
            stringValue
                .padStart(sizeBits.toInt(), if (stringValue[0] == '1') '1' else '0')
                .map { if (it == '1') '0' else '1' }
                .joinToString("")
        }

        val expectedNegativeValue = expectedValueTransformation(negativeValue)
        val expectedPositiveValue = expectedValueTransformation(positiveValue)

        assertEquals(
            expectedNegativeValue,
            actualNegativeValue,
            message = "Size bits: $sizeBits, negativeValue: $negativeValue"
        )
        assertEquals(
            expectedPositiveValue,
            actualPositiveValue,
            message = "Size bits: $sizeBits, positiveValue: $positiveValue"
        )
    }

    @Test
    fun testBvReductionAndExpr(): Unit = with(context) {
        val longValue = Random.nextLong()
        val value = longValue.toBv(sizeBits = Random.nextUInt(from = 64u, until = 1000u))
        val symbolicResult = mkBv1Sort().mkConst("symbolicValue")

        solver.assert(symbolicResult eq value.reductionAnd())
        solver.check()

        val actualValue = (solver.model().eval(symbolicResult) as KBitVec1Value).value
        val expectedValue = longValue.toBinary().none { it == '0' }

        assertEquals(expectedValue, actualValue)
    }

    @Test
    fun testBvReductionOrExpr(): Unit = with(context) {
        val longValue = Random.nextLong()
        val value = longValue.toBv(sizeBits = Random.nextUInt(from = 64u, until = 1000u))
        val symbolicResult = mkBv1Sort().mkConst("symbolicValue")

        solver.assert(symbolicResult eq value.reductionOr())
        solver.check()

        val actualValue = (solver.model().eval(symbolicResult) as KBitVec1Value).value
        val expectedValue = longValue.toBinary().any { it == '1' }

        assertEquals(expectedValue, actualValue)
    }

    @Test
    fun testAndExpr(): Unit = with(context) {
        val value = Random.nextLong()

        val bv = value.toBv()
        val anotherBv = Random.nextLong().toBv()
        val zero = 0L.toBv()

        val conjunctionWithItself = mkBvAndExpr(bv, bv)
        val conjunctionWithZero = mkBvAndExpr(bv, zero)
        val conjunctionWithOnes = mkBvAndExpr(bv, mkBvNotExpr(zero))

        val conjunctionResult = mkBv64Sort().mkConst("symbolicVariable")

        solver.assert(conjunctionWithItself eq bv)
        solver.assert(conjunctionWithZero eq zero)
        solver.assert(conjunctionWithOnes eq bv)
        solver.assert(mkBvAndExpr(bv, anotherBv) eq conjunctionResult)

        solver.check()

        val actualValue = (solver.model().eval(conjunctionResult) as KBitVec64Value).numberValue
        val expectedValue = value and anotherBv.numberValue

        assertEquals(expectedValue, actualValue)
    }

    @Test
    fun testOrExpr(): Unit = with(context) {
        val value = Random.nextLong()

        val bv = value.toBv()
        val anotherBv = Random.nextLong().toBv()
        val zero = 0L.toBv()

        val disjunctionWithItself = mkBvOrExpr(bv, bv)
        val disjunctionWithZero = mkBvOrExpr(bv, zero)
        val disjunctionWithOnes = mkBvOrExpr(bv, mkBvNotExpr(zero))

        val disjunctionResult = mkBv64Sort().mkConst("symbolicVariable")

        solver.assert(disjunctionWithItself eq bv)
        solver.assert(disjunctionWithZero eq bv)
        solver.assert(disjunctionWithOnes eq mkBvNotExpr(zero))
        solver.assert(mkBvOrExpr(bv, anotherBv) eq disjunctionResult)

        solver.check()

        val actualValue = (solver.model().eval(disjunctionResult) as KBitVec64Value).numberValue
        val expectedValue = value or anotherBv.numberValue

        assertEquals(expectedValue, actualValue)
    }

    private fun testBinaryOperation(
        symbolicOperation: (KExpr<KBv64Sort>, KExpr<KBv64Sort>) -> KExpr<KBv64Sort>,
        concreteOperation: (Long, Long) -> Long
    ): Unit = with(context) {
        val (negativeValue, positiveValue) = createTwoRandomLongValues()

        val negativeBv = negativeValue.toBv()
        val positiveBv = positiveValue.toBv()

        val firstResult = mkBv64Sort().mkConst("symbolicVariable")
        val secondResult = mkBv64Sort().mkConst("anotherSymbolicVariable")

        solver.assert(symbolicOperation(negativeBv, positiveBv) eq firstResult)
        solver.assert(symbolicOperation(positiveBv, negativeBv) eq secondResult)
        solver.check()

        val firstActualValue = (solver.model().eval(firstResult) as KBitVec64Value).numberValue
        val secondActualValue = (solver.model().eval(secondResult) as KBitVec64Value).numberValue

        val firstExpectedValue = concreteOperation(negativeValue, positiveValue)
        val secondExpectedValue = concreteOperation(positiveValue, negativeValue)

        assertEquals(firstExpectedValue, firstActualValue)
        assertEquals(secondExpectedValue, secondActualValue)
    }

    private fun testLogicalOperation(
        symbolicOperation: (KExpr<KBv64Sort>, KExpr<KBv64Sort>) -> KExpr<KBoolSort>,
        concreteOperation: (Long, Long) -> Boolean
    ): Unit = with(context) {
        val values = (0 until 2).map { Random.nextLong() }.sorted()
        val bvValues = values.map { it.toBv() }

        val withItselfConst = mkBoolSort().mkConst("withItself")
        val firstWithSecondConst = mkBoolSort().mkConst("firstWithSecond")
        val secondWithFirstConst = mkBoolSort().mkConst("secondWithFirst")

        val withItself = symbolicOperation(bvValues[0], bvValues[0]) eq withItselfConst
        val firstWithSecond = symbolicOperation(bvValues[0], bvValues[1]) eq firstWithSecondConst
        val secondWithFirst = symbolicOperation(bvValues[1], bvValues[0]) eq secondWithFirstConst

        solver.assert(withItself)
        solver.assert(firstWithSecond)
        solver.assert(secondWithFirst)

        solver.check()
        val model = solver.model()

        val expectedValues = listOf(
            concreteOperation(values[0], values[0]),
            concreteOperation(values[0], values[1]),
            concreteOperation(values[1], values[0])
        )

        val actualValues = listOf(
            model.eval(withItselfConst),
            model.eval(firstWithSecondConst),
            model.eval(secondWithFirstConst)
        )

        assertFalse("Values: $values") { expectedValues[0] xor (actualValues[0] is KTrue) }
        assertFalse("Values: $values") { expectedValues[1] xor (actualValues[1] is KTrue) }
        assertFalse("Values: $values") { expectedValues[2] xor (actualValues[2] is KTrue) }
    }

    @Test
    fun testXorExpr(): Unit = testBinaryOperation(context::mkBvXorExpr, Long::xor)

    @Test
    fun testNAndExpr(): Unit = testBinaryOperation(context::mkBvNAndExpr) { arg0: Long, arg1: Long ->
        (arg0 and arg1).inv()
    }

    @Test
    fun testNorExpr(): Unit = testBinaryOperation(context::mkBvNorExpr) { arg0: Long, arg1: Long ->
        (arg0 or arg1).inv()
    }

    @Test
    fun testXNorExpr(): Unit = testBinaryOperation(context::mkBvXNorExpr) { arg0: Long, arg1: Long ->
        (arg0 xor arg1).inv()
    }

    @Test
    fun testNegationExpr(): Unit = with(context) {
        val (negativeValue, positiveValue) = createTwoRandomLongValues()

        val negativeBv = negativeValue.toBv()
        val positiveBv = positiveValue.toBv()
        val zero = 0.toBv()

        val negNegativeValue = mkBv64Sort().mkConst("neg_negative_value")
        val negPositiveValue = mkBv64Sort().mkConst("neg_positive_value")
        val zeroValue = mkBv32Sort().mkConst("zero_value")

        solver.assert(mkBvNegationExpr(negativeBv) eq negNegativeValue)
        solver.assert(mkBvNegationExpr(positiveBv) eq negPositiveValue)
        solver.assert(mkBvNegationExpr(zero) eq zeroValue)

        solver.check()
        val model = solver.model()

        val evaluatedNegNegativeBv = model.eval(negNegativeValue) as KBitVec64Value
        val evaluatedNegPositiveBv = model.eval(negPositiveValue) as KBitVec64Value
        val evaluatedZeroValue = model.eval(zeroValue) as KBitVec32Value

        val message = "NegativeValue: $negativeValue, positiveValue: $positiveValue"

        assertEquals(-negativeValue, evaluatedNegNegativeBv.numberValue, message)
        assertEquals(-positiveValue, evaluatedNegPositiveBv.numberValue, message)
        assertEquals(expected = 0, evaluatedZeroValue.numberValue, message)
    }

    @Test
    fun testAddExpr(): Unit = testBinaryOperation(context::mkBvAddExpr, Long::plus)

    @Test
    fun testSubExpr(): Unit = testBinaryOperation(context::mkBvSubExpr, Long::minus)

    @Test
    fun testMulExpr(): Unit = testBinaryOperation(context::mkBvMulExpr, Long::times)

    @Test
    fun testUnsignedDivExpr(): Unit = testBinaryOperation(context::mkBvUnsignedDivExpr) { arg0: Long, arg1: Long ->
        (arg0.toULong() / arg1.toULong()).toLong()
    }

    @Test
    fun testSignedDivExpr(): Unit = testBinaryOperation(context::mkBvSignedDivExpr, Long::div)

    @Test
    fun testUnsignedRemExpr(): Unit = testBinaryOperation(context::mkBvUnsignedRemExpr) { arg0: Long, arg1: Long ->
        arg0.toULong().rem(arg1.toULong()).toLong()
    }

    @Test
    fun testSignedReminderExpr(): Unit = testBinaryOperation(context::mkBvSignedRemExpr, Long::rem)

    @Test
    fun testSignedModExpr(): Unit = testBinaryOperation(context::mkBvSignedModExpr, Long::mod)

    @Test
    fun testUnsignedLessExpr(): Unit = testLogicalOperation(context::mkBvUnsignedLessExpr) { arg0: Long, arg1: Long ->
        arg0.toULong() < arg1.toULong()
    }

    @Test
    fun testSignedLessExpr(): Unit = testLogicalOperation(context::mkBvSignedLessExpr) { arg0: Long, arg1: Long ->
        arg0 < arg1
    }

    @Test
    fun testUnsignedLessOrEqualExpr(): Unit =
        testLogicalOperation(context::mkBvUnsignedLessOrEqualExpr) { arg0: Long, arg1: Long ->
            arg0.toULong() <= arg1.toULong()
        }

    @Test
    fun testSignedLessOrEqualExpr(): Unit =
        testLogicalOperation(context::mkBvSignedLessOrEqualExpr) { arg0: Long, arg1: Long ->
            arg0 <= arg1
        }

    @Test
    fun testUnsignedGreaterOrEqualExpr(): Unit =
        testLogicalOperation(context::mkBvUnsignedGreaterOrEqualExpr) { arg0: Long, arg1: Long ->
            arg0.toULong() >= arg1.toULong()
        }

    @Test
    fun testSignedGreaterOrEqualExpr(): Unit =
        testLogicalOperation(context::mkBvSignedGreaterOrEqualExpr) { arg0: Long, arg1: Long ->
            arg0 >= arg1
        }

    @Test
    fun testUnsignedGreaterExpr(): Unit =
        testLogicalOperation(context::mkBvUnsignedGreaterExpr) { arg0: Long, arg1: Long ->
            arg0.toULong() > arg1.toULong()
        }

    @Test
    fun testSignedGreaterExpr(): Unit = testLogicalOperation(context::mkBvSignedGreaterExpr) { arg0: Long, arg1: Long ->
        arg0 > arg1
    }

    @Test
    fun testConcatExpr(): Unit = with(context) {
        val firstBv = Random.nextLong().toBv()
        val secondBv = Random.nextInt().toBv()

        val sizeBits = firstBv.sort.sizeBits + secondBv.sort.sizeBits
        val symbolicConst = mkBvSort(sizeBits).mkConst("symbolicConst")

        solver.assert(symbolicConst eq mkBvConcatExpr(firstBv, secondBv))
        solver.check()

        val resultValue = solver.model().eval(symbolicConst) as KBitVecCustomValue
        val expectedResult = firstBv.numberValue.toBinary() + secondBv.numberValue.toBinary()

        assertEquals(expectedResult, resultValue.stringValue)
    }

    @Test
    fun testBvExtractExpr(): Unit = with(context) {
        val value = Random.nextLong().toBv()
        val high = Random.nextInt(from = 32, until = 64)
        val low = Random.nextInt(from = 5, until = 32)

        val symbolicValue = mkBvSort(high.toUInt() - low.toUInt() + 1u).mkConst("symbolicVariable")

        solver.assert(symbolicValue eq mkBvExtractExpr(high, low, value))
        solver.check()

        val result = solver.model().eval(symbolicValue) as KBitVecValue<*>
        val sizeBits = value.sort.sizeBits.toInt()
        val expectedResult = value.numberValue.toBinary().substring(sizeBits - high - 1, sizeBits - low)

        assertEquals(expectedResult, result.stringValue)
    }

    @Test
    fun testBvSignExtensionExpr(): Unit = with(context) {
        val negativeBv = Random.nextInt(from = Int.MIN_VALUE, until = 0).toBv()
        val positiveBv = Random.nextInt(from = 1, until = Int.MAX_VALUE).toBv()

        val positiveSymbolicVariable = mkBvSort(Long.SIZE_BITS.toUInt()).mkConst("positiveSymbolicVariable")
        val negativeSymbolicVariable = mkBvSort(Long.SIZE_BITS.toUInt()).mkConst("negativeSymbolicVariable")

        solver.assert(positiveSymbolicVariable eq mkBvSignExtensionExpr(Int.SIZE_BITS, positiveBv))
        solver.assert(negativeSymbolicVariable eq mkBvSignExtensionExpr(Int.SIZE_BITS, negativeBv))
        solver.check()

        val positiveResult = (solver.model().eval(positiveSymbolicVariable) as KBitVec64Value).numberValue.toBinary()
        val negativeResult = (solver.model().eval(negativeSymbolicVariable) as KBitVec64Value).numberValue.toBinary()

        val expectedPositiveResult = positiveBv.numberValue.toBinary().padStart(Long.SIZE_BITS, '0')
        val expectedNegativeResult = negativeBv.numberValue.toBinary().padStart(Long.SIZE_BITS, '1')

        assertEquals(expectedPositiveResult, positiveResult)
        assertEquals(expectedNegativeResult, negativeResult)
    }

    @Test
    fun testBvZeroExtensionExpr(): Unit = with(context) {
        val negativeBv = Random.nextInt(from = Int.MIN_VALUE, until = 0).toBv()
        val positiveBv = Random.nextInt(from = 1, until = Int.MAX_VALUE).toBv()

        val positiveSymbolicVariable = mkBvSort(Long.SIZE_BITS.toUInt()).mkConst("positiveSymbolicVariable")
        val negativeSymbolicVariable = mkBvSort(Long.SIZE_BITS.toUInt()).mkConst("negativeSymbolicVariable")

        solver.assert(positiveSymbolicVariable eq mkBvZeroExtensionExpr(Int.SIZE_BITS, positiveBv))
        solver.assert(negativeSymbolicVariable eq mkBvZeroExtensionExpr(Int.SIZE_BITS, negativeBv))
        solver.check()

        val positiveResult = (solver.model().eval(positiveSymbolicVariable) as KBitVec64Value).numberValue.toBinary()
        val negativeResult = (solver.model().eval(negativeSymbolicVariable) as KBitVec64Value).numberValue.toBinary()

        val expectedPositiveResult = positiveBv.numberValue.toBinary().padStart(Long.SIZE_BITS, '0')
        val expectedNegativeResult = negativeBv.numberValue.toBinary().padStart(Long.SIZE_BITS, '0')

        assertEquals(expectedPositiveResult, positiveResult)
        assertEquals(expectedNegativeResult, negativeResult)
    }

    @Test
    fun testBvRepeatExpr(): Unit = with(context) {
        val bv = Random.nextInt().toShort().toBv()
        val numberOfRepetitions = 4u

        val symbolicVariable = mkBvSort(bv.sort.sizeBits * numberOfRepetitions).mkConst("symbolicVariable")

        solver.assert(symbolicVariable eq mkBvRepeatExpr(numberOfRepetitions.toInt(), bv))
        solver.check()

        val result = (solver.model().eval(symbolicVariable) as KBitVec64Value).numberValue.toBinary()
        val expectedValue = bv.numberValue.toBinary().repeat(numberOfRepetitions.toInt())

        assertEquals(expectedValue, result)
    }

    private fun testShift(
        symbolicOperation: (KExpr<KBv64Sort>, KExpr<KBv64Sort>) -> KExpr<KBv64Sort>,
        concreteOperation: (Long, Int) -> Long
    ) = with(context) {
        val value = Random.nextLong().toBv()
        val shiftSize = Random.nextInt(from = 1, until = 50).toLong().toBv()

        val symbolicVariable = value.sort.mkConst("symbolicVariable")

        solver.assert(symbolicVariable eq symbolicOperation(value, shiftSize))
        solver.check()

        val expectedResult = concreteOperation(value.numberValue, shiftSize.numberValue.toInt())
        val result = (solver.model().eval(symbolicVariable) as KBitVec64Value).numberValue

        assertEquals(expectedResult, result)
    }

    @Test
    fun testBvShiftLeftExpr(): Unit = testShift(context::mkBvShiftLeftExpr, Long::shl)

    @Test
    fun testBvLogicalShiftRightExpr(): Unit = testShift(context::mkBvLogicalShiftRightExpr, Long::ushr)

    @Test
    fun testBvArithShiftRightExpr(): Unit = testShift(context::mkBvArithShiftRightExpr, Long::shr)

    @Test
    fun testRotateLeft(): Unit = with(context) {
        val bv = Random.nextLong().toBv()
        val rotateSize = Random.nextLong(from = 1, until = 4).toBv()

        val symbolicVariable = bv.sort.mkConst("symbolicVariable")

        solver.assert(symbolicVariable eq mkBvRotateLeftExpr(bv, rotateSize))
        solver.check()

        val expectedResult = bv.numberValue.toBinary().let {
            it.substring(rotateSize.numberValue.toInt(), it.length) + it.substring(0, rotateSize.numberValue.toInt())
        }
        val actualResult = (solver.model().eval(symbolicVariable) as KBitVec64Value).numberValue.toBinary()

        assertEquals(expectedResult, actualResult)
    }

    @Test
    fun testIndexedRotateLeft(): Unit = with(context) {
        val bv = Random.nextLong().toBv()
        val rotateSize = Random.nextInt(from = 1, until = 4)

        val symbolicVariable = bv.sort.mkConst("symbolicVariable")

        solver.assert(symbolicVariable eq mkBvRotateLeftIndexedExpr(rotateSize, bv))
        solver.check()

        val expectedResult = bv.numberValue.toBinary().let {
            it.substring(rotateSize, it.length) + it.substring(0, rotateSize)
        }
        val actualResult = (solver.model().eval(symbolicVariable) as KBitVec64Value).numberValue.toBinary()

        assertEquals(expectedResult, actualResult)
    }

    @Test
    fun testRotateRight(): Unit = with(context) {
        val bv = Random.nextLong().toBv()
        val rotateSize = Random.nextLong(from = 1, until = 4).toBv()

        val symbolicVariable = bv.sort.mkConst("symbolicVariable")

        solver.assert(symbolicVariable eq mkBvRotateRightExpr(bv, rotateSize))
        solver.check()

        val expectedResult = bv.numberValue.toBinary().let {
            val firstPart = it.substring(it.length - rotateSize.numberValue.toInt(), it.length)
            val secondPart = it.substring(0, it.length - rotateSize.numberValue.toInt())
            firstPart + secondPart
        }
        val actualResult = (solver.model().eval(symbolicVariable) as KBitVec64Value).numberValue.toBinary()

        assertEquals(expectedResult, actualResult)
    }

    @Test
    fun testIndexedRotateRight(): Unit = with(context) {
        val bv = Random.nextLong().toBv()
        val rotateSize = Random.nextInt(from = 1, until = 4)

        val symbolicVariable = bv.sort.mkConst("symbolicVariable")

        solver.assert(symbolicVariable eq mkBvRotateRightIndexedExpr(rotateSize, bv))
        solver.check()

        val expectedResult = bv.numberValue.toBinary().let {
            val firstPart = it.substring(it.length - rotateSize, it.length)
            val secondPart = it.substring(0, it.length - rotateSize)
            firstPart + secondPart
        }
        val actualResult = (solver.model().eval(symbolicVariable) as KBitVec64Value).numberValue.toBinary()

        assertEquals(expectedResult, actualResult)
    }

    @Test
    fun testKBv2Int(): Unit = with(context) {
        val value = Random.nextLong(from = Long.MIN_VALUE, until = 0).toBv()
        val signedSymbolicValue = mkIntSort().mkConst("signedSymbolicValue")
        val unsignedSymbolicValue = mkIntSort().mkConst("unsignedSymbolicValue")

        solver.assert(signedSymbolicValue eq mkBv2IntExpr(value, isSigned = true))
        solver.assert(unsignedSymbolicValue eq mkBv2IntExpr(value, isSigned = false))
        solver.check()

        val expectedSignedResult = value.numberValue
        val expectedUnsignedResult = value.numberValue.toULong()

        val actualSignedResult = (solver.model().eval(signedSymbolicValue) as KIntNumExpr)
        val actualUnsignedResult = (solver.model().eval(unsignedSymbolicValue) as KIntNumExpr)

        assertEquals(expectedSignedResult, (actualSignedResult as KInt64NumExpr).value)
        assertEquals(expectedUnsignedResult.toString(), (actualUnsignedResult as KIntBigNumExpr).value.toString())
    }

    private fun testOverUnderflowWithSign(
        initialSignedValue: Byte,
        initialUnsignedValue: UByte,
        operation: (KBitVec8Value, KBitVec8Value, Boolean) -> KExpr<KBoolSort>,
        signedErrorValue: Byte,
        unsignedErrorValue: UByte,
        signedNoErrorValue: Byte,
        unsignedNoErrorValue: UByte
    ): Unit = with(context) {
        val signedValue = initialSignedValue.toBv()
        val unsignedValue = initialUnsignedValue.toBv()

        val symbolicSignedOverflow = mkBoolSort().mkConst("signedOverflow")
        val symbolicUnsignedOverflow = mkBoolSort().mkConst("unsignedOverflow")
        val symbolicSignedNoOverflow = mkBoolSort().mkConst("signedNoOverflow")
        val symbolicUnsignedNoOverflow = mkBoolSort().mkConst("unsignedNoOverflow")

        val signedOverflow = operation(signedValue, signedErrorValue.toBv(), true)
        val unsignedOverflow = operation(unsignedValue, unsignedErrorValue.toBv(), false)
        val signedNoOverflow = operation(signedValue, signedNoErrorValue.toBv(), true)
        val unsignedNoOverflow = operation(unsignedValue, unsignedNoErrorValue.toBv(), false)

        solver.assert(symbolicSignedOverflow eq signedOverflow)
        solver.assert(symbolicUnsignedOverflow eq unsignedOverflow)
        solver.assert(symbolicSignedNoOverflow eq signedNoOverflow)
        solver.assert(symbolicUnsignedNoOverflow eq unsignedNoOverflow)
        solver.check()

        val actualSignedOverflowResult = (solver.model().eval(symbolicSignedOverflow))
        val actualUnsignedOverflowResult = (solver.model().eval(symbolicUnsignedOverflow))
        val actualSignedNoOverflowResult = (solver.model().eval(symbolicSignedNoOverflow))
        val actualUnsignedNoOverflowResult = (solver.model().eval(symbolicUnsignedNoOverflow))

        assertTrue(actualSignedOverflowResult is KFalse)
        assertTrue(actualUnsignedOverflowResult is KFalse)
        assertTrue(actualSignedNoOverflowResult is KTrue)
        assertTrue(actualUnsignedNoOverflowResult is KTrue)
    }

    private fun testOverUnderflowWithoutSign(
        initialValue: Byte,
        operation: (KBitVec8Value, KBitVec8Value) -> KExpr<KBoolSort>,
        errorValue: Byte,
        noErrorValue: Byte,
    ): Unit = with(context) {
        val value = initialValue.toBv()

        val symbolicOverflow = mkBoolSort().mkConst("overflow")
        val symbolicNoOverflow = mkBoolSort().mkConst("noOverflow")

        val overflow = operation(value, errorValue.toBv())
        val noOverflow = operation(value, noErrorValue.toBv())

        solver.assert(symbolicOverflow eq overflow)
        solver.assert(symbolicNoOverflow eq noOverflow)
        solver.check()

        val actualOverflowResult = (solver.model().eval(symbolicOverflow))
        val actualNoOverflowResult = (solver.model().eval(symbolicNoOverflow))

        assertTrue(actualOverflowResult is KFalse)
        assertTrue(actualNoOverflowResult is KTrue)
    }

    @Test
    fun testKBvAddNoOverflow(): Unit = testOverUnderflowWithSign(
        initialSignedValue = 100.toByte(),
        initialUnsignedValue = 100.toUByte(),
        operation = context::mkBvAddNoOverflowExpr,
        signedErrorValue = 100.toByte(),
        unsignedErrorValue = 200.toUByte(),
        signedNoErrorValue = 10.toByte(),
        unsignedNoErrorValue = 100.toUByte()
    )

    @Test
    fun testKBvAddNoUnderflow(): Unit = testOverUnderflowWithoutSign(
        initialValue = (-100).toByte(),
        operation = context::mkBvAddNoUnderflowExpr,
        errorValue = (-120).toByte(),
        noErrorValue = (-5).toByte(),
    )

    @Test
    fun testKBvSubNoOverflow(): Unit = testOverUnderflowWithoutSign(
        initialValue = 100.toByte(),
        operation = context::mkBvSubNoOverflowExpr,
        errorValue = (-100).toByte(),
        noErrorValue = 10.toByte()
    )

    @Test
    fun testKBvSubNoUnderflow(): Unit = testOverUnderflowWithSign(
        initialSignedValue = (-100).toByte(),
        initialUnsignedValue = 10.toUByte(),
        operation = context::mkBvSubNoUnderflowExpr,
        signedErrorValue = Byte.MAX_VALUE,
        unsignedErrorValue = 100.toUByte(),
        signedNoErrorValue = 11.toByte(),
        unsignedNoErrorValue = 10.toUByte()
    )

    @Test
    fun testKBvDivNoOverflow(): Unit = testOverUnderflowWithoutSign(
        initialValue = (-128).toByte(),
        operation = context::mkBvDivNoOverflowExpr,
        errorValue = (-1).toByte(),
        noErrorValue = 1.toByte()
    )

    @Test
    fun testKBvNegNoOverflow(): Unit = with(context) {
        val errorValue = (-128).toByte()
        val noErrorValue = 127.toByte()

        val symbolicOverflow = mkBoolSort().mkConst("overflow")
        val symbolicNoOverflow = mkBoolSort().mkConst("noOverflow")

        val overflow = mkBvNegationNoOverflowExpr(errorValue.toBv())
        val noOverflow = mkBvNegationNoOverflowExpr(noErrorValue.toBv())

        solver.assert(symbolicOverflow eq overflow)
        solver.assert(symbolicNoOverflow eq noOverflow)
        solver.check()

        val actualOverflowResult = (solver.model().eval(symbolicOverflow))
        val actualNoOverflowResult = (solver.model().eval(symbolicNoOverflow))

        assertTrue(actualOverflowResult is KFalse)
        assertTrue(actualNoOverflowResult is KTrue)
    }

    @Test
    fun testKBvMulNoOverflow(): Unit = testOverUnderflowWithSign(
        initialSignedValue = 100.toByte(),
        initialUnsignedValue = 100.toUByte(),
        operation = context::mkBvMulNoOverflowExpr,
        signedErrorValue = 2.toByte(),
        unsignedErrorValue = 3.toUByte(),
        signedNoErrorValue = 1.toByte(),
        unsignedNoErrorValue = 2.toUByte()
    )

    private fun createTwoRandomLongValues(): Pair<NegativeLong, PositiveLong> {
        val negativeValue = Random.nextLong(from = Int.MIN_VALUE.toLong(), until = 0L)
        val positiveValue = Random.nextLong(from = 1L, until = Int.MAX_VALUE.toLong())

        return negativeValue to positiveValue
    }
}

typealias PositiveLong = Long
typealias NegativeLong = Long
