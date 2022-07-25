package org.ksmt.solver.z3

import kotlin.random.Random
import kotlin.random.nextUInt
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.ksmt.KContext
import org.ksmt.expr.KBitVec32Value
import org.ksmt.expr.KBitVecCustomValue
import org.ksmt.expr.KBitVecValue
import org.ksmt.sort.KBv16Sort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBv64Sort
import org.ksmt.sort.KBv8Sort

class BvExamples {
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
        val positiveValue = Random.nextLong(from = 1L, until = Long.MAX_VALUE)
        val negativeValue = Random.nextLong(from = Long.MIN_VALUE, until = 0L)

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
    fun testCreateIllegalBv(): Unit = with(context) {
        val sizeBits = 42u
        val stringValue = "0".repeat(sizeBits.toInt() - 1)

        assertFailsWith(IllegalArgumentException::class) { mkBv(stringValue, sizeBits) }
    }

    @Test
    fun testNotExpr(): Unit = with(context) {
        val stringValue = generateBinaryString(Random.nextInt(from = 65, until = 100))
        val sizeBits = stringValue.length.toUInt()

        val bv = mkBv(stringValue, sizeBits)
        val symbolicBv = mkBvSort(sizeBits).mkConst("symbolic_variable")

        solver.assert(mkEq(mkBvNotExpr(mkBvNotExpr(bv)), bv))
        solver.assert(mkEq(mkBvNotExpr(bv), symbolicBv))
        solver.check()

        val actualValue = (solver.model().eval(symbolicBv) as KBitVecCustomValue).decimalStringValue
        val expectedValue = stringValue.map { if (it == '1') '0' else '1' }.joinToString("")

        assertEquals(expectedValue, actualValue)
    }

    @Test
    fun testNegationExpr(): Unit = with(context) {
        val negativeValue = Random.nextInt(from = Int.MIN_VALUE, until = 0)
        val positiveValue = Random.nextInt(from = 1, until = Int.MAX_VALUE)

        val negativeBv = negativeValue.toBv()
        val positiveBv = positiveValue.toBv()
        val zero = 0.toBv()

        val negNegativeValue = mkBv32Sort().mkConst("neg_negative_value")
        val negPositiveValue = mkBv32Sort().mkConst("neg_positive_value")
        val zeroValue = mkBv32Sort().mkConst("zero_value")

        solver.assert(mkEq(mkBvNegationExpr(negativeBv), negNegativeValue))
        solver.assert(mkEq(mkBvNegationExpr(positiveBv), negPositiveValue))
        solver.assert(mkEq(mkBvNegationExpr(zero), zeroValue))

        solver.check()
        val model = solver.model()

        val evaluatedNegNegativeBv = model.eval(negNegativeValue) as KBitVec32Value
        val evaluatedNegPositiveBv = model.eval(negPositiveValue) as KBitVec32Value
        val evaluatedZeroValue = model.eval(zeroValue) as KBitVec32Value

        assertEquals(-negativeValue, evaluatedNegNegativeBv.numberValue)
        assertEquals(-positiveValue, evaluatedNegPositiveBv.numberValue)
        assertEquals(expected = 0, evaluatedZeroValue.numberValue)
    }

    private fun generateBinaryString(length: Int): String =
        CharArray(length) { if (Random.nextInt() % 2 == 0) '1' else '0' }.joinToString("")

    // TODO extract into a common module
    private fun Number.toBinary(): String = when (this) {
        is Byte -> toUByte().toString(radix = 2).padStart(Byte.SIZE_BITS, '0')
        is Short -> toUShort().toString(radix = 2).padStart(Short.SIZE_BITS, '0')
        is Int -> toUInt().toString(radix = 2).padStart(Int.SIZE_BITS, '0')
        is Long -> toULong().toString(radix = 2).padStart(Long.SIZE_BITS, '0')
        else -> error("Unsupported type for transformation into a binary string: ${this::class.simpleName}")
    }
}