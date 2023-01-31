package org.ksmt.utils

import org.ksmt.KContext
import org.ksmt.expr.KBitVec16Value
import org.ksmt.expr.KBitVec1Value
import org.ksmt.expr.KBitVec32Value
import org.ksmt.expr.KBitVec64Value
import org.ksmt.expr.KBitVec8Value
import org.ksmt.expr.KBitVecCustomValue
import org.ksmt.expr.KBitVecNumberValue
import org.ksmt.expr.KBitVecValue
import java.math.BigInteger
import kotlin.experimental.inv
import kotlin.experimental.or
import kotlin.experimental.xor

object BvUtils {

    private val bvMinValueSigned = object : BvSpecialValueSource {
        override val bv1: Boolean = true // sign bit 1
        override val bv8: Byte = Byte.MIN_VALUE
        override val bv16: Short = Short.MIN_VALUE
        override val bv32: Int = Int.MIN_VALUE
        override val bv64: Long = Long.MIN_VALUE

        override fun bvDefault(size: UInt): BigInteger =
            powerOfTwo(size - 1u)
    }

    fun KContext.bvMinValueSigned(size: UInt): KBitVecValue<*> = mkBvSpecialValue(size, bvMinValueSigned)
    fun KBitVecValue<*>.isBvMinValueSigned(): Boolean = isBvSpecialValue(bvMinValueSigned)

    private val bvMaxValueSigned = object : BvSpecialValueSource {
        override val bv1: Boolean = false // sign bit 0
        override val bv8: Byte = Byte.MAX_VALUE
        override val bv16: Short = Short.MAX_VALUE
        override val bv32: Int = Int.MAX_VALUE
        override val bv64: Long = Long.MAX_VALUE

        override fun bvDefault(size: UInt): BigInteger =
            powerOfTwo(size - 1u) - BigInteger.ONE
    }

    fun KContext.bvMaxValueSigned(size: UInt): KBitVecValue<*> = mkBvSpecialValue(size, bvMaxValueSigned)
    fun KBitVecValue<*>.isBvMaxValueSigned(): Boolean = isBvSpecialValue(bvMaxValueSigned)

    private val bvMaxValueUnsigned = object : BvSpecialValueSource {
        override val bv1: Boolean = true
        override val bv8: Byte = (-1).toByte()
        override val bv16: Short = (-1).toShort()
        override val bv32: Int = -1
        override val bv64: Long = -1L

        override fun bvDefault(size: UInt): BigInteger =
            powerOfTwo(size) - BigInteger.ONE
    }

    fun KContext.bvMaxValueUnsigned(size: UInt): KBitVecValue<*> = mkBvSpecialValue(size, bvMaxValueUnsigned)
    fun KBitVecValue<*>.isBvMaxValueUnsigned(): Boolean = isBvSpecialValue(bvMaxValueUnsigned)

    private val bvZeroValue = object : BvSpecialValueSource {
        override val bv1: Boolean = false
        override val bv8: Byte = 0
        override val bv16: Short = 0
        override val bv32: Int = 0
        override val bv64: Long = 0

        override fun bvDefault(size: UInt): BigInteger = BigInteger.ZERO
    }

    fun KContext.bvZero(size: UInt): KBitVecValue<*> = mkBvSpecialValue(size, bvZeroValue)
    fun KBitVecValue<*>.isBvZero(): Boolean = isBvSpecialValue(bvZeroValue)

    private val bvOneValue = object : BvSpecialValueSource {
        override val bv1: Boolean = true
        override val bv8: Byte = 1
        override val bv16: Short = 1
        override val bv32: Int = 1
        override val bv64: Long = 1L

        override fun bvDefault(size: UInt): BigInteger = BigInteger.ONE
    }

    fun KContext.bvOne(size: UInt): KBitVecValue<*> = mkBvSpecialValue(size, bvOneValue)
    fun KBitVecValue<*>.isBvOne(): Boolean = isBvSpecialValue(bvOneValue)

    private class BvIntValue(val value: Int) : BvSpecialValueSource {
        override val bv1: Boolean = value != 0
        override val bv8: Byte = value.toByte()
        override val bv16: Short = value.toShort()
        override val bv32: Int = value
        override val bv64: Long = value.toLong()

        override fun bvDefault(size: UInt): BigInteger =
            value.toBigInteger().normalizeValue(size)
    }

    fun KContext.bvValue(size: UInt, value: Int): KBitVecValue<*> = mkBvSpecialValue(size, BvIntValue(value))
    fun KBitVecValue<*>.bvValueIs(value: Int): Boolean = isBvSpecialValue(BvIntValue(value))

    fun KBitVecValue<*>.getBit(bit: UInt): Boolean {
        check(bit < sort.sizeBits) { "Requested bit is out of bounds for $sort" }
        return when (this) {
            is KBitVec1Value -> value
            is KBitVecNumberValue<*, *> -> ((numberValue.toLong() shr bit.toInt()) and 0x1L) == 0x1L
            is KBitVecCustomValue -> value.testBit(bit.toInt())
            else -> stringValue.let { it[it.lastIndex - bit.toInt()] == '1' }
        }
    }

    fun KBitVecValue<*>.signBit(): Boolean = getBit(sort.sizeBits - 1u)

    fun KBitVecValue<*>.signedGreaterOrEqual(other: Int): Boolean = when (this) {
        is KBitVec1Value -> if (value) {
            // 1 >= 1 -> true
            // 1 >= 0 -> false
            other == 1
        } else {
            // 0 >= 1 -> true
            // 0 >= 0 -> true
            true
        }

        is KBitVec8Value -> numberValue >= other
        is KBitVec16Value -> numberValue >= other
        is KBitVec32Value -> numberValue >= other
        is KBitVec64Value -> numberValue >= other
        is KBitVecCustomValue -> value.signedValue(sizeBits) >= other.toBigInteger()
        else -> stringValue.toBigInteger(radix = 2).signedValue(sort.sizeBits) >= other.toBigInteger()
    }

    fun KBitVecValue<*>.signedLessOrEqual(other: KBitVecValue<*>): Boolean = when (this) {
        is KBitVec1Value -> if (value) {
            // 1 <= 0 -> true
            // 1 <= 1 -> true
            true
        } else {
            // 0 <= 1 -> false
            // 0 <= 0 -> true
            val otherValue = (other as KBitVec1Value).value
            !otherValue
        }

        is KBitVec8Value -> numberValue <= (other as KBitVec8Value).numberValue
        is KBitVec16Value -> numberValue <= (other as KBitVec16Value).numberValue
        is KBitVec32Value -> numberValue <= (other as KBitVec32Value).numberValue
        is KBitVec64Value -> numberValue <= (other as KBitVec64Value).numberValue
        is KBitVecCustomValue -> {
            val lhs = value.signedValue(sizeBits)
            val rhs = (other as KBitVecCustomValue).value.signedValue(sizeBits)
            lhs <= rhs
        }
        else -> {
            val lhs = stringValue.toBigInteger(radix = 2).signedValue(sort.sizeBits)
            val rhs = other.stringValue.toBigInteger(radix = 2).signedValue(sort.sizeBits)
            lhs <= rhs
        }
    }

    fun KBitVecValue<*>.unsignedLessOrEqual(other: KBitVecValue<*>): Boolean = when (this) {
        is KBitVec1Value -> value <= (other as KBitVec1Value).value
        is KBitVec8Value -> numberValue.toUByte() <= (other as KBitVec8Value).numberValue.toUByte()
        is KBitVec16Value -> numberValue.toUShort() <= (other as KBitVec16Value).numberValue.toUShort()
        is KBitVec32Value -> numberValue.toUInt() <= (other as KBitVec32Value).numberValue.toUInt()
        is KBitVec64Value -> numberValue.toULong() <= (other as KBitVec64Value).numberValue.toULong()
        is KBitVecCustomValue -> value <= (other as KBitVecCustomValue).value
        // MSB first -> lexical order works
        else -> stringValue <= other.stringValue
    }

    fun KBitVecValue<*>.signedLess(other: KBitVecValue<*>): Boolean =
        signedLessOrEqual(other) && this != other

    fun KBitVecValue<*>.unsignedLess(other: KBitVecValue<*>): Boolean =
        unsignedLessOrEqual(other) && this != other

    fun KBitVecValue<*>.signedGreaterOrEqual(other: KBitVecValue<*>): Boolean =
        other.signedLessOrEqual(this)

    fun KBitVecValue<*>.unsignedGreaterOrEqual(other: KBitVecValue<*>): Boolean =
        other.unsignedLessOrEqual(this)

    fun KBitVecValue<*>.signedGreater(other: KBitVecValue<*>): Boolean =
        other.signedLess(this)

    fun KBitVecValue<*>.unsignedGreater(other: KBitVecValue<*>): Boolean =
        other.unsignedLess(this)

    operator fun KBitVecValue<*>.unaryMinus(): KBitVecValue<*> =
        ctx.bvZero(sort.sizeBits) - this

    operator fun KBitVecValue<*>.plus(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> a xor b },
        bv8 = { a, b -> (a + b).toByte() },
        bv16 = { a, b -> (a + b).toShort() },
        bv32 = { a, b -> a + b },
        bv64 = { a, b -> a + b },
        bvDefault = { a, b -> a + b },
    )

    operator fun KBitVecValue<*>.minus(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> a xor b },
        bv8 = { a, b -> (a - b).toByte() },
        bv16 = { a, b -> (a - b).toShort() },
        bv32 = { a, b -> a - b },
        bv64 = { a, b -> a - b },
        bvDefault = { a, b -> a - b },
    )

    operator fun KBitVecValue<*>.times(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> a && b },
        bv8 = { a, b -> (a * b).toByte() },
        bv16 = { a, b -> (a * b).toShort() },
        bv32 = { a, b -> a * b },
        bv64 = { a, b -> a * b },
        bvDefault = { a, b -> a * b },
    )

    fun KBitVecValue<*>.signedDivide(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> a == b },
        bv8 = { a, b -> (a / b).toByte() },
        bv16 = { a, b -> (a / b).toShort() },
        bv32 = { a, b -> a / b },
        bv64 = { a, b -> a / b },
        bvDefault = { a, b -> a / b },
    )

    fun KBitVecValue<*>.unsignedDivide(other: KBitVecValue<*>): KBitVecValue<*> = bvUnsignedOperation(
        other = other,
        bv1 = { a, b -> a == b },
        bv8 = { a, b -> (a / b).toUByte() },
        bv16 = { a, b -> (a / b).toUShort() },
        bv32 = { a, b -> a / b },
        bv64 = { a, b -> a / b },
        bvDefault = { a, b -> a / b },
    )

    fun KBitVecValue<*>.signedRem(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> a != b },
        bv8 = { a, b -> (a.rem(b)).toByte() },
        bv16 = { a, b -> (a.rem(b)).toShort() },
        bv32 = { a, b -> a.rem(b) },
        bv64 = { a, b -> a.rem(b) },
        bvDefault = { a, b -> a.rem(b) },
    )

    fun KBitVecValue<*>.unsignedRem(other: KBitVecValue<*>): KBitVecValue<*> = bvUnsignedOperation(
        other = other,
        bv1 = { a, b -> a != b },
        bv8 = { a, b -> (a.rem(b)).toUByte() },
        bv16 = { a, b -> (a.rem(b)).toUShort() },
        bv32 = { a, b -> a.rem(b) },
        bv64 = { a, b -> a.rem(b) },
        bvDefault = { a, b -> a.rem(b) },
    )

    fun KBitVecValue<*>.signedMod(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> a != b },
        bv8 = { a, b -> a.mod(b) },
        bv16 = { a, b -> a.mod(b) },
        bv32 = { a, b -> a.mod(b) },
        bv64 = { a, b -> a.mod(b) },
        bvDefault = { a, b ->
            val size = sort.sizeBits
            val aAbs = a.abs().normalizeValue(size)
            val bAbs = b.abs().normalizeValue(size)
            val u = aAbs.mod(bAbs).normalizeValue(size)
            when {
                u == BigInteger.ZERO -> BigInteger.ZERO
                a >= BigInteger.ZERO && b >= BigInteger.ZERO -> u
                a < BigInteger.ZERO && b >= BigInteger.ZERO -> (-u + b).normalizeValue(size)
                a >= BigInteger.ZERO && b < BigInteger.ZERO -> (u + b).normalizeValue(size)
                else -> (-u).normalizeValue(size)
            }
        },
    )

    fun KBitVecValue<*>.bitwiseNot(): KBitVecValue<*> = bvOperation(
        other = ctx.bvZero(sort.sizeBits),
        bv1 = { a, _ -> a.not() },
        bv8 = { a, _ -> a.inv() },
        bv16 = { a, _ -> a.inv() },
        bv32 = { a, _ -> a.inv() },
        bv64 = { a, _ -> a.inv() },
        bvDefault = { a, _ -> a.inv() },
    )

    fun KBitVecValue<*>.bitwiseOr(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> a || b },
        bv8 = { a, b -> a or b },
        bv16 = { a, b -> a or b },
        bv32 = { a, b -> a or b },
        bv64 = { a, b -> a or b },
        bvDefault = { a, b -> a or b },
    )

    fun KBitVecValue<*>.bitwiseXor(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> a xor b },
        bv8 = { a, b -> a xor b },
        bv16 = { a, b -> a xor b },
        bv32 = { a, b -> a xor b },
        bv64 = { a, b -> a xor b },
        bvDefault = { a, b -> a xor b },
    )

    fun KBitVecValue<*>.bitwiseAnd(other: KBitVecValue<*>): KBitVecValue<*> =
        (this.bitwiseNot().bitwiseOr(other.bitwiseNot())).bitwiseNot()

    private val bv8PossibleShift = 0 until Byte.SIZE_BITS
    private val bv16PossibleShift = 0 until Short.SIZE_BITS
    private val bv32PossibleShift = 0 until Int.SIZE_BITS
    private val bv64PossibleShift = 0 until Long.SIZE_BITS

    fun KBitVecValue<*>.shiftLeft(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> if (b) false else a },
        bv8 = { a, b -> if (b !in bv8PossibleShift) 0 else (a.toInt() shl b.toInt()).toByte() },
        bv16 = { a, b -> if (b !in bv16PossibleShift) 0 else (a.toInt() shl b.toInt()).toShort() },
        bv32 = { a, b -> if (b !in bv32PossibleShift) 0 else (a shl b) },
        bv64 = { a, b -> if (b !in bv64PossibleShift) 0 else (a shl b.toInt()) },
        bvDefault = { a, b ->
            if (b < BigInteger.ZERO || b >= sort.sizeBits.toInt().toBigInteger()) {
                BigInteger.ZERO
            } else {
                a shl b.toInt()
            }
        },
    )

    fun KBitVecValue<*>.shiftRightLogical(other: KBitVecValue<*>): KBitVecValue<*> = bvUnsignedOperation(
        other = other,
        bv1 = { a, b -> if (b) false else a },
        bv8 = { a, b -> if (b.toInt() !in bv8PossibleShift) 0u else (a.toInt() ushr b.toInt()).toUByte() },
        bv16 = { a, b -> if (b.toInt() !in bv16PossibleShift) 0u else (a.toInt() ushr b.toInt()).toUShort() },
        bv32 = { a, b -> if (b.toInt() !in bv32PossibleShift) 0u else (a.toInt() ushr b.toInt()).toUInt() },
        bv64 = { a, b -> if (b.toLong() !in bv64PossibleShift) 0u else (a.toLong() ushr b.toInt()).toULong() },
        bvDefault = { a, b ->
            if (b < BigInteger.ZERO || b >= sort.sizeBits.toInt().toBigInteger()) {
                BigInteger.ZERO
            } else {
                a shr b.toInt()
            }
        },
    )

    fun KBitVecValue<*>.shiftRightArith(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, _ -> a },
        bv8 = { a, b -> if (b !in bv8PossibleShift) (if (a < 0) -1 else 0) else (a.toInt() shr b.toInt()).toByte() },
        bv16 = { a, b -> if (b !in bv16PossibleShift) (if (a < 0) -1 else 0) else (a.toInt() shr b.toInt()).toShort() },
        bv32 = { a, b -> if (b !in bv32PossibleShift) (if (a < 0) -1 else 0) else (a shr b) },
        bv64 = { a, b -> if (b !in bv64PossibleShift) (if (a < 0) -1L else 0L) else (a shr b.toInt()) },
        bvDefault = { a, b ->
            if (b < BigInteger.ZERO || b >= sort.sizeBits.toInt().toBigInteger()) {
                if (a < BigInteger.ZERO) BigInteger.valueOf(-1) else BigInteger.ZERO
            } else {
                a shr b.toInt()
            }
        },
    )

    fun KBitVecValue<*>.powerOfTwoOrNull(): Int? {
        val value = bigIntValue()
        val valueMinusOne = value - BigInteger.ONE
        if ((value and valueMinusOne) != BigInteger.ZERO) return null
        return valueMinusOne.bitLength()
    }

    fun KBitVecValue<*>.bigIntValue(): BigInteger = when (this) {
        is KBitVec1Value -> if (value) BigInteger.ONE else BigInteger.ZERO
        is KBitVecNumberValue<*, *> -> numberValue.toBigInteger()
        is KBitVecCustomValue -> value
        else -> stringValue.toBigInteger(radix = 2)
    }

    fun KBitVecValue<*>.toBigIntegerSigned(): BigInteger =
        toBigIntegerUnsigned().signedValue(sort.sizeBits)

    fun KBitVecValue<*>.toBigIntegerUnsigned(): BigInteger =
        bigIntValue().normalizeValue(sort.sizeBits)

    fun concatBv(lhs: KBitVecValue<*>, rhs: KBitVecValue<*>): KBitVecValue<*> = with(lhs.ctx) {
        when {
            lhs is KBitVec8Value && rhs is KBitVec8Value -> {
                var result = lhs.numberValue.toUByte().toInt() shl Byte.SIZE_BITS
                result = result or rhs.numberValue.toUByte().toInt()
                mkBv(result.toShort())
            }

            lhs is KBitVec16Value && rhs is KBitVec16Value -> {
                var result = lhs.numberValue.toUShort().toInt() shl Short.SIZE_BITS
                result = result or rhs.numberValue.toUShort().toInt()
                mkBv(result)
            }

            lhs is KBitVec32Value && rhs is KBitVec32Value -> {
                var result = lhs.numberValue.toUInt().toLong() shl Int.SIZE_BITS
                result = result or rhs.numberValue.toUInt().toLong()
                mkBv(result)
            }

            else -> {
                val lhsValue = lhs.bigIntValue().normalizeValue(lhs.sort.sizeBits)
                val rhsValue = rhs.bigIntValue().normalizeValue(rhs.sort.sizeBits)
                val concatenatedValue = concatValues(lhsValue, rhsValue, rhs.sort.sizeBits)
                mkBv(concatenatedValue, lhs.sort.sizeBits + rhs.sort.sizeBits)
            }
        }
    }

    fun KBitVecValue<*>.extractBv(high: Int, low: Int): KBitVecValue<*> = with(ctx) {
        val size = (high - low + 1).toUInt()
        val value = bigIntValue().normalizeValue(sort.sizeBits)
        val trimLowerBits = value.shiftRight(low)
        val trimHigherBits = trimLowerBits.and(binaryOnes(onesCount = size))
        mkBv(trimHigherBits, size)
    }

    fun KBitVecValue<*>.signExtension(extensionSize: UInt): KBitVecValue<*> =
        if (!signBit()) {
            zeroExtension(extensionSize)
        } else {
            val extension = binaryOnes(onesCount = extensionSize)
            val value = bigIntValue().normalizeValue(sort.sizeBits)
            val extendedValue = concatValues(extension, value, sort.sizeBits)
            ctx.mkBv(extendedValue, sort.sizeBits + extensionSize)
        }

    fun KBitVecValue<*>.zeroExtension(extensionSize: UInt): KBitVecValue<*> = with(ctx) {
        mkBv(bigIntValue().normalizeValue(sort.sizeBits), sort.sizeBits + extensionSize)
    }

    private fun binaryOnes(onesCount: UInt): BigInteger = powerOfTwo(onesCount) - BigInteger.ONE

    private fun concatValues(lhs: BigInteger, rhs: BigInteger, rhsSize: UInt): BigInteger =
        lhs.shiftLeft(rhsSize.toInt()).or(rhs)

    @Suppress("LongParameterList")
    private inline fun KBitVecValue<*>.bvUnsignedOperation(
        other: KBitVecValue<*>,
        crossinline bv1: (Boolean, Boolean) -> Boolean,
        crossinline bv8: (UByte, UByte) -> UByte,
        crossinline bv16: (UShort, UShort) -> UShort,
        crossinline bv32: (UInt, UInt) -> UInt,
        crossinline bv64: (ULong, ULong) -> ULong,
        crossinline bvDefault: (BigInteger, BigInteger) -> BigInteger,
    ): KBitVecValue<*> = when (this@bvUnsignedOperation) {
        is KBitVec1Value -> bv1Operation(other, op = bv1)
        is KBitVec8Value -> bv8UnsignedOperation(other, op = bv8)
        is KBitVec16Value -> bv16UnsignedOperation(other, op = bv16)
        is KBitVec32Value -> bv32UnsignedOperation(other, op = bv32)
        is KBitVec64Value -> bv64UnsignedOperation(other, op = bv64)
        is KBitVecCustomValue -> bvCustomOperation(other, signed = false, operation = bvDefault)
        else -> bvOperationDefault(other, signed = false, operation = bvDefault)
    }

    @Suppress("LongParameterList")
    private inline fun KBitVecValue<*>.bvOperation(
        other: KBitVecValue<*>,
        crossinline bv1: (Boolean, Boolean) -> Boolean,
        crossinline bv8: (Byte, Byte) -> Byte,
        crossinline bv16: (Short, Short) -> Short,
        crossinline bv32: (Int, Int) -> Int,
        crossinline bv64: (Long, Long) -> Long,
        crossinline bvDefault: (BigInteger, BigInteger) -> BigInteger,
    ): KBitVecValue<*> = when (this@bvOperation) {
        is KBitVec1Value -> bv1Operation(other, op = bv1)
        is KBitVec8Value -> bv8Operation(other, op = bv8)
        is KBitVec16Value -> bv16Operation(other, op = bv16)
        is KBitVec32Value -> bv32Operation(other, op = bv32)
        is KBitVec64Value -> bv64Operation(other, op = bv64)
        is KBitVecCustomValue -> bvCustomOperation(other, signed = true, operation = bvDefault)
        else -> bvOperationDefault(other, signed = true, operation = bvDefault)
    }

    private inline fun KBitVec1Value.bv1Operation(
        other: KBitVecValue<*>,
        crossinline op: (Boolean, Boolean) -> Boolean
    ): KBitVec1Value = bvNumericOperation(
        this, other,
        unwrap = { it.value },
        wrap = ctx::mkBv,
        op = op
    )

    private inline fun KBitVec8Value.bv8Operation(
        other: KBitVecValue<*>,
        crossinline op: (Byte, Byte) -> Byte
    ): KBitVec8Value = bvNumericOperation(
        this, other,
        unwrap = { it.numberValue },
        wrap = ctx::mkBv,
        op = op
    )

    private inline fun KBitVec8Value.bv8UnsignedOperation(
        other: KBitVecValue<*>,
        crossinline op: (UByte, UByte) -> UByte
    ): KBitVec8Value = bvNumericOperation(
        this, other,
        unwrap = { it.numberValue.toUByte() },
        wrap = { ctx.mkBv(it.toByte()) },
        op = op
    )

    private inline fun KBitVec16Value.bv16Operation(
        other: KBitVecValue<*>,
        crossinline op: (Short, Short) -> Short
    ): KBitVec16Value = bvNumericOperation(
        this, other,
        unwrap = { it.numberValue },
        wrap = ctx::mkBv,
        op = op
    )

    private inline fun KBitVec16Value.bv16UnsignedOperation(
        other: KBitVecValue<*>,
        crossinline op: (UShort, UShort) -> UShort
    ): KBitVec16Value = bvNumericOperation(
        this, other,
        unwrap = { it.numberValue.toUShort() },
        wrap = { ctx.mkBv(it.toShort()) },
        op = op
    )

    private inline fun KBitVec32Value.bv32Operation(
        other: KBitVecValue<*>,
        crossinline op: (Int, Int) -> Int
    ): KBitVec32Value = bvNumericOperation(
        this, other,
        unwrap = { it.numberValue },
        wrap = ctx::mkBv,
        op = op
    )

    private inline fun KBitVec32Value.bv32UnsignedOperation(
        other: KBitVecValue<*>,
        crossinline op: (UInt, UInt) -> UInt
    ): KBitVec32Value = bvNumericOperation(
        this, other,
        unwrap = { it.numberValue.toUInt() },
        wrap = { ctx.mkBv(it.toInt()) },
        op = op
    )

    private inline fun KBitVec64Value.bv64Operation(
        other: KBitVecValue<*>,
        crossinline op: (Long, Long) -> Long
    ): KBitVec64Value = bvNumericOperation(
        this, other,
        unwrap = { it.numberValue },
        wrap = ctx::mkBv,
        op = op
    )

    private inline fun KBitVec64Value.bv64UnsignedOperation(
        other: KBitVecValue<*>,
        crossinline op: (ULong, ULong) -> ULong
    ): KBitVec64Value = bvNumericOperation(
        this, other,
        unwrap = { it.numberValue.toULong() },
        wrap = { ctx.mkBv(it.toLong()) },
        op = op
    )

    private inline fun KBitVecCustomValue.bvCustomOperation(
        other: KBitVecValue<*>,
        signed: Boolean = false,
        crossinline operation: (BigInteger, BigInteger) -> BigInteger
    ): KBitVecCustomValue = bvNumericOperation(
        this, other,
        unwrap = { if (!signed) it.value else it.value.signedValue(sizeBits) },
        wrap = { ctx.mkBv(it.normalizeValue(sizeBits), sizeBits).uncheckedCast() },
        op = operation
    )

    private inline fun KBitVecValue<*>.bvOperationDefault(
        other: KBitVecValue<*>,
        signed: Boolean = false,
        crossinline operation: (BigInteger, BigInteger) -> BigInteger
    ): KBitVecValue<*> = bvNumericOperation(
        this, other,
        unwrap = {
            val bigIntValue = it.stringValue.toBigInteger(radix = 2)
            if (!signed) bigIntValue else bigIntValue.signedValue(sort.sizeBits)
        },
        wrap = { ctx.mkBv(it.normalizeValue(sort.sizeBits), sort.sizeBits) },
        op = operation
    )

    private inline fun <reified T, reified V> bvNumericOperation(
        arg0: KBitVecValue<*>, arg1: KBitVecValue<*>,
        crossinline unwrap: (T) -> V,
        crossinline wrap: (V) -> T,
        crossinline op: (V, V) -> V
    ): T {
        val a0 = unwrap(arg0 as T)
        val a1 = unwrap(arg1 as T)
        return wrap(op(a0, a1))
    }

    private fun BigInteger.signedValue(size: UInt): BigInteger {
        val maxValue = powerOfTwo(size - 1u) - BigInteger.ONE
        return if (this > maxValue) {
            this - powerOfTwo(size)
        } else {
            this
        }
    }

    private interface BvSpecialValueSource {
        val bv1: Boolean
        val bv8: Byte
        val bv16: Short
        val bv32: Int
        val bv64: Long
        fun bvDefault(size: UInt): BigInteger
    }

    private fun KContext.mkBvSpecialValue(size: UInt, source: BvSpecialValueSource): KBitVecValue<*> =
        when (size.toInt()) {
            1 -> mkBv(source.bv1)
            Byte.SIZE_BITS -> mkBv(source.bv8)
            Short.SIZE_BITS -> mkBv(source.bv16)
            Int.SIZE_BITS -> mkBv(source.bv32)
            Long.SIZE_BITS -> mkBv(source.bv64)
            else -> mkBv(source.bvDefault(size), size)
        }

    private fun KBitVecValue<*>.isBvSpecialValue(source: BvSpecialValueSource) = when (this) {
        is KBitVec1Value -> value == source.bv1
        is KBitVec8Value -> numberValue == source.bv8
        is KBitVec16Value -> numberValue == source.bv16
        is KBitVec32Value -> numberValue == source.bv32
        is KBitVec64Value -> numberValue == source.bv64
        is KBitVecCustomValue -> value == source.bvDefault(sizeBits)
        else -> this == ctx.mkBvSpecialValue(sort.sizeBits, source)
    }

}
