package org.ksmt.solver.z3

import kotlin.random.Random
import kotlin.random.nextUInt
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import org.ksmt.KContext
import org.ksmt.decl.KBitVecValueDecl
import org.ksmt.expr.KBitVec1Value
import org.ksmt.expr.KBitVec32Value
import org.ksmt.expr.KBitVec64Value
import org.ksmt.expr.KBitVecCustomValue
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KExpr
import org.ksmt.expr.KInt64NumExpr
import org.ksmt.expr.KIntBigNumExpr
import org.ksmt.expr.KIntNumExpr
import org.ksmt.expr.KTrue
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv16Sort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBv64Sort
import org.ksmt.sort.KBv8Sort
import org.ksmt.sort.KBvSort

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

        assertEquals(expected = "#b1", trueBv.decl.name)
        assertEquals(expected = "#b0", falseBv.decl.name)

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

        assertEquals(expected = "#b${positiveStringValue}", positiveBv.decl.name)
        assertEquals(expected = "#b${negativeStringValue}", negativeBv.decl.name)

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

        assertEquals(expected = "#b${positiveStringValue}", positiveBv.decl.name)
        assertEquals(expected = "#b${negativeStringValue}", negativeBv.decl.name)

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

        assertEquals(expected = "#b${positiveStringValue}", positiveBv.decl.name)
        assertEquals(expected = "#b${negativeStringValue}", negativeBv.decl.name)

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

        assertEquals(expected = "#b${positiveStringValue}", positiveBv.decl.name)
        assertEquals(expected = "#b${negativeStringValue}", negativeBv.decl.name)

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

        assertEquals(positiveBvFromString.decimalStringValue, positiveValue)
        assertEquals(negativeBvFromString.decimalStringValue, negativeValue)

        assertEquals(positiveBvFromString.sort.sizeBits, sizeBits)
        assertEquals(negativeBvFromString.sort.sizeBits, sizeBits)

        assertEquals(expected = "#b$positiveValue", positiveBvFromString.decl.name)
        assertEquals(expected = "#b$negativeValue", negativeBvFromString.decl.name)

        assertEquals(positiveBvFromString.sort, negativeBvFromString.sort)
    }

    @Test
    @Disabled("TODO add check for the size")
    fun testCreateIllegalBv(): Unit = with(context) {
        val sizeBits = 42u
        val stringValue = "0".repeat(sizeBits.toInt() - 1)

        assertFailsWith(IllegalArgumentException::class) { mkBv(stringValue, sizeBits) }
    }

    @Test
    fun testNotExpr(): Unit = with(context) {
        val sizeBits = Random.nextInt(from = 65, until = 100).toUInt()
        val (negativeValue, positiveValue) = createTwoRandomLongValues().let { it.first.toString() to it.second.toString() }

        val negativeBv = mkBv(negativeValue, sizeBits)
        val positiveBv = mkBv(positiveValue, sizeBits)

        val negativeSymbolicValue = mkBvSort(sizeBits).mkConst("negative_symbolic_variable")
        val positiveSymbolicValue = mkBvSort(sizeBits).mkConst("positive_symbolic_variable")

        solver.assert(mkBvNotExpr(mkBvNotExpr(negativeBv)) eq negativeBv)
        solver.assert(mkBvNotExpr(negativeBv) eq negativeSymbolicValue)

        solver.assert(mkBvNotExpr(mkBvNotExpr(positiveBv)) eq positiveBv)
        solver.assert(mkBvNotExpr(positiveBv) eq positiveSymbolicValue)

        solver.check()

        val actualNegativeValue = (solver.model().eval(negativeSymbolicValue) as KBitVecCustomValue).decimalStringValue
        val actualPositiveValue = (solver.model().eval(positiveSymbolicValue) as KBitVecCustomValue).decimalStringValue

        val expectedValueTransformation = { stringValue: String ->
            stringValue
                .toLong()
                .toBinary()
                .padStart(sizeBits.toInt(), if (stringValue.toLong() < 0) '1' else '0')
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
        val value = longValue.toBv(sizeBits = Random.nextUInt(from = 0u, until = 1000u))
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
        val value = longValue.toBv(sizeBits = Random.nextUInt(from = 0u, until = 1000u))
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
        symbolicOperation: (KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBvSort>,
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
        symbolicOperation: (KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBoolSort>,
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

        assertFalse { expectedValues[0] xor (actualValues[0] is KTrue) }
        assertFalse { expectedValues[1] xor (actualValues[1] is KTrue) }
        assertFalse { expectedValues[2] xor (actualValues[2] is KTrue) }
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

        assertEquals(-negativeValue, evaluatedNegNegativeBv.numberValue)
        assertEquals(-positiveValue, evaluatedNegPositiveBv.numberValue)
        assertEquals(expected = 0, evaluatedZeroValue.numberValue)
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
//    @Disabled("Doesn't work yet")
    fun testUnsignedRemExpr(): Unit = testBinaryOperation(context::mkBvUnsignedRemExpr) { arg0: Long, arg1: Long ->
        arg0 - (arg0.toULong() / arg1.toULong()).toLong() * arg1
    }

    @Test
    fun testSignedReminderExpr(): Unit = testBinaryOperation(context::mkBvSignedRemExpr) { arg0: Long, arg1: Long ->
        arg0 - (arg0 / arg1) * arg1
    }

    @Test
    fun testSignedModExpr(): Unit = testBinaryOperation(context::mkBvSignedModExpr) { arg0: Long, arg1: Long ->
        TODO()
    }

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
            arg0 >= arg1
        }

    @Test
    fun testSignedGreaterOrEqualExpr(): Unit =
        testLogicalOperation(context::mkBvSignedGreaterOrEqualExpr) { arg0: Long, arg1: Long ->
            arg0 >= arg1
        }

    @Test
    fun testUnsignedGreaterExpr(): Unit =
        testLogicalOperation(context::mkBvUnsignedGreaterExpr) { arg0: Long, arg1: Long ->
            arg0.toULong() >= arg1.toULong()
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

        // TODO a bug with decimalStringValue and binary representation, run a debug to understand what I mean
        assertEquals(expectedResult, resultValue.decimalStringValue)
    }

    @Test
    fun testBvExtractExpr() : Unit = with(context) {
        val value = Random.nextLong().toBv()
        val high = Random.nextInt(from = 32, until = 64)
        val low = Random.nextInt(from = 5, until = 32)

        val symbolicValue = mkBvSort(high.toUInt() - low.toUInt() + 1u).mkConst("symbolicVariable")

        solver.assert(symbolicValue eq mkBvExtractExpr(high, low, value))
        solver.check()

        val result = solver.model().eval(symbolicValue) as KBitVecValue<*>
        val sizeBits = value.sort.sizeBits.toInt()
        val expectedResult = value.numberValue.toBinary().substring(sizeBits - high - 1, sizeBits - low)

        assertEquals(expectedResult, (result.decl as KBitVecValueDecl).value)
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
        symbolicOperation: (KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBvSort>,
        concreteOperation: (Long, Int) -> Long
    ) = with(context) {
        val value = Random.nextLong().toBv()
        // TODO add restriction for positive shiftSize or check that it is not required
        val shiftSize = Random.nextInt(from = 1, until = 50).toLong().toBv()

        val symbolicVariable = value.sort().mkConst("symbolicVariable")

        solver.assert(symbolicVariable eq symbolicOperation(value, shiftSize))
        solver.check()

        val expectedResult = concreteOperation(value.numberValue, shiftSize.numberValue.toInt())
        val result = (solver.model().eval(symbolicVariable) as KBitVec64Value).numberValue

        assertEquals(expectedResult, result)
    }

    // получается не строгая типизация, на самом деле. можно передать вот в таком виде функцию, а внутри testShift
    // вызвать от двух разных векторов, и тогда всё умрёт в рантайме
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

        val symbolicVariable = bv.sort().mkConst("symbolicVariable")

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

        val symbolicVariable = bv.sort().mkConst("symbolicVariable")

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

        val symbolicVariable = bv.sort().mkConst("symbolicVariable")

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

        val symbolicVariable = bv.sort().mkConst("symbolicVariable")

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

    // TODO add tests for tests with overloads

    private fun createTwoRandomLongValues(): Pair<NegativeLong, PositiveLong> {
        val negativeValue = Random.nextLong(from = Int.MIN_VALUE.toLong(), until = 0L)
        val positiveValue = Random.nextLong(from = 1L, until = Int.MAX_VALUE.toLong())

        return negativeValue to positiveValue
    }

    // TODO extract into a common module
    private fun Number.toBinary(): String = when (this) {
        is Byte -> toUByte().toString(radix = 2).padStart(Byte.SIZE_BITS, '0')
        is Short -> toUShort().toString(radix = 2).padStart(Short.SIZE_BITS, '0')
        is Int -> toUInt().toString(radix = 2).padStart(Int.SIZE_BITS, '0')
        is Long -> toULong().toString(radix = 2).padStart(Long.SIZE_BITS, '0')
        else -> error("Unsupported type for transformation into a binary string: ${this::class.simpleName}")
    }

}

typealias PositiveLong = Long
typealias NegativeLong = Long