package org.ksmt.utils

import org.ksmt.KContext
import org.ksmt.expr.KBitVec16Value
import org.ksmt.expr.KBitVec1Value
import org.ksmt.expr.KBitVec32Value
import org.ksmt.expr.KBitVec64Value
import org.ksmt.expr.KBitVec8Value
import org.ksmt.expr.KBitVecValue
import org.ksmt.sort.KBvSort
import java.math.BigInteger
import kotlin.experimental.inv
import kotlin.experimental.or
import kotlin.experimental.xor

object BvUtils {

    fun KContext.bvMinValueSigned(size: UInt): KBitVecValue<*> = when (size.toInt()) {
        1 -> mkBv(false)
        Byte.SIZE_BITS -> mkBv(Byte.MIN_VALUE)
        Short.SIZE_BITS -> mkBv(Short.MIN_VALUE)
        Int.SIZE_BITS -> mkBv(Int.MIN_VALUE)
        Long.SIZE_BITS -> mkBv(Long.MIN_VALUE)
        else -> {
            val binaryValue = "1" + "0".repeat(size.toInt() - 1)
            mkBv(binaryValue, size)
        }
    }

    fun KContext.bvMaxValueSigned(size: UInt): KBitVecValue<*> = when (size.toInt()) {
        1 -> mkBv(true)
        Byte.SIZE_BITS -> mkBv(Byte.MAX_VALUE)
        Short.SIZE_BITS -> mkBv(Short.MAX_VALUE)
        Int.SIZE_BITS -> mkBv(Int.MAX_VALUE)
        Long.SIZE_BITS -> mkBv(Long.MAX_VALUE)
        else -> {
            val binaryValue = "0" + "1".repeat(size.toInt() - 1)
            mkBv(binaryValue, size)
        }
    }

    fun KContext.bvMaxValueUnsigned(size: UInt): KBitVecValue<*> = when (size.toInt()) {
        1 -> mkBv(true)
        Byte.SIZE_BITS -> mkBv((-1).toByte())
        Short.SIZE_BITS -> mkBv((-1).toShort())
        Int.SIZE_BITS -> mkBv(-1)
        Long.SIZE_BITS -> mkBv(-1L)
        else -> {
            val binaryValue = "1".repeat(size.toInt())
            mkBv(binaryValue, size)
        }
    }

    fun KContext.bvZero(size: UInt): KBitVecValue<*> = bvValue(size, 0)
    fun KContext.bvOne(size: UInt): KBitVecValue<*> = bvValue(size, 1)

    fun KContext.bvValue(size: UInt, value: Int): KBitVecValue<*> = when (size.toInt()) {
        1 -> mkBv(value == 1)
        Byte.SIZE_BITS -> mkBv(value.toByte())
        Short.SIZE_BITS -> mkBv(value.toShort())
        Int.SIZE_BITS -> mkBv(value)
        Long.SIZE_BITS -> mkBv(value.toLong())
        else -> mkBv(value, size)
    }

    fun KBitVecValue<*>.signedGreaterOrEqual(other: Int): Boolean = when (this) {
        is KBitVec1Value -> value >= (other == 1)
        is KBitVec8Value -> numberValue >= other
        is KBitVec16Value -> numberValue >= other
        is KBitVec32Value -> numberValue >= other
        is KBitVec64Value -> numberValue >= other
        else -> signedBigIntFromBinary(stringValue) >= other.toBigInteger()
    }

    fun KBitVecValue<*>.signedLessOrEqual(other: KBitVecValue<*>): Boolean = when (this) {
        is KBitVec1Value -> value <= (other as KBitVec1Value).value
        is KBitVec8Value -> numberValue <= (other as KBitVec8Value).numberValue
        is KBitVec16Value -> numberValue <= (other as KBitVec16Value).numberValue
        is KBitVec32Value -> numberValue <= (other as KBitVec32Value).numberValue
        is KBitVec64Value -> numberValue <= (other as KBitVec64Value).numberValue
        else -> signedBigIntFromBinary(stringValue) <= signedBigIntFromBinary(other.stringValue)
    }

    fun KBitVecValue<*>.unsignedLessOrEqual(other: KBitVecValue<*>): Boolean = when (this) {
        is KBitVec1Value -> value <= (other as KBitVec1Value).value
        is KBitVec8Value -> numberValue.toUByte() <= (other as KBitVec8Value).numberValue.toUByte()
        is KBitVec16Value -> numberValue.toUShort() <= (other as KBitVec16Value).numberValue.toUShort()
        is KBitVec32Value -> numberValue.toUInt() <= (other as KBitVec32Value).numberValue.toUInt()
        is KBitVec64Value -> numberValue.toULong() <= (other as KBitVec64Value).numberValue.toULong()
        // MSB first -> lexical order works
        else -> stringValue <= other.stringValue
    }

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
        bvDefault = { a, b -> a.mod(b) },
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

    fun KBitVecValue<*>.shiftLeft(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> if (b) false else a },
        bv8 = { a, b -> (a.toInt() shl b.toInt()).toByte() },
        bv16 = { a, b -> (a.toInt() shl b.toInt()).toShort() },
        bv32 = { a, b -> a shl b },
        bv64 = { a, b -> a shl b.toInt() },
        bvDefault = { a, b -> a shl b.toInt() },
    )

    fun KBitVecValue<*>.shiftRightLogical(other: KBitVecValue<*>): KBitVecValue<*> = bvUnsignedOperation(
        other = other,
        bv1 = { a, b -> if (b) false else a },
        bv8 = { a, b -> (a.toInt() ushr b.toInt()).toUByte() },
        bv16 = { a, b -> (a.toInt() ushr b.toInt()).toUShort() },
        bv32 = { a, b -> (a.toInt() ushr b.toInt()).toUInt() },
        bv64 = { a, b -> (a.toLong() ushr b.toInt()).toULong() },
        bvDefault = { a, b -> (a shr b.toInt()) },
    )

    fun KBitVecValue<*>.shiftRightArith(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, _ -> a },
        bv8 = { a, b -> (a.toInt() shr b.toInt()).toByte() },
        bv16 = { a, b -> (a.toInt() shr b.toInt()).toShort() },
        bv32 = { a, b -> a shr b },
        bv64 = { a, b -> a shr b.toInt() },
        bvDefault = { a, b -> a shr b.toInt() },
    )

    fun KBitVecValue<*>.powerOfTwoOrNull(): Int? {
        val value = unsignedBigIntFromBinary(stringValue)
        val valueMinusOne = value - BigInteger.ONE
        if ((value and valueMinusOne) != BigInteger.ZERO) return null
        return valueMinusOne.bitLength()
    }

    fun KBitVecValue<*>.intValueOrNull(): Int? = when (this) {
        is KBitVec1Value -> if (value) 1 else 0
        is KBitVec8Value -> numberValue.toInt()
        is KBitVec16Value -> numberValue.toInt()
        is KBitVec32Value -> numberValue
        is KBitVec64Value -> if (numberValue <= Int.MAX_VALUE) numberValue.toInt() else null
        else -> stringValue.toIntOrNull(radix = 2)
    }

    fun KContext.mkBvFromBigInteger(value: BigInteger, size: UInt): KBitVecValue<KBvSort> {
        val normalizedValue = value.mod(BigInteger.valueOf(2).pow(size.toInt()))
        val resultBinary = unsignedBinaryString(normalizedValue).padStart(size.toInt(), '0')
        return mkBv(resultBinary, size)
    }

    fun KBitVecValue<*>.toBigIntegerSigned(): BigInteger =
        signedBigIntFromBinary(stringValue)

    fun KBitVecValue<*>.toBigIntegerUnsigned(): BigInteger =
        unsignedBigIntFromBinary(stringValue)

    @Suppress("LongParameterList")
    private inline fun KBitVecValue<*>.bvUnsignedOperation(
        other: KBitVecValue<*>,
        bv1: (Boolean, Boolean) -> Boolean,
        bv8: (UByte, UByte) -> UByte,
        bv16: (UShort, UShort) -> UShort,
        bv32: (UInt, UInt) -> UInt,
        bv64: (ULong, ULong) -> ULong,
        bvDefault: (BigInteger, BigInteger) -> BigInteger,
    ): KBitVecValue<*> = when (this@bvUnsignedOperation) {
        is KBitVec1Value -> bv1Operation(other, bv1)
        is KBitVec8Value -> bv8UnsignedOperation(other, bv8)
        is KBitVec16Value -> bv16UnsignedOperation(other, bv16)
        is KBitVec32Value -> bv32UnsignedOperation(other, bv32)
        is KBitVec64Value -> bv64UnsignedOperation(other, bv64)
        else -> bvOperationDefault(other, signed = false, bvDefault)
    }

    @Suppress("LongParameterList")
    private inline fun KBitVecValue<*>.bvOperation(
        other: KBitVecValue<*>,
        bv1: (Boolean, Boolean) -> Boolean,
        bv8: (Byte, Byte) -> Byte,
        bv16: (Short, Short) -> Short,
        bv32: (Int, Int) -> Int,
        bv64: (Long, Long) -> Long,
        bvDefault: (BigInteger, BigInteger) -> BigInteger,
    ): KBitVecValue<*> = when (this@bvOperation) {
        is KBitVec1Value -> bv1Operation(other, bv1)
        is KBitVec8Value -> bv8Operation(other, bv8)
        is KBitVec16Value -> bv16Operation(other, bv16)
        is KBitVec32Value -> bv32Operation(other, bv32)
        is KBitVec64Value -> bv64Operation(other, bv64)
        else -> bvOperationDefault(other, signed = true, bvDefault)
    }

    private inline fun KBitVec1Value.bv1Operation(other: KBitVecValue<*>, op: (Boolean, Boolean) -> Boolean) =
        ctx.mkBv(op(value, (other as KBitVec1Value).value))

    private inline fun KBitVec8Value.bv8Operation(other: KBitVecValue<*>, op: (Byte, Byte) -> Byte) =
        ctx.mkBv(op(numberValue, (other as KBitVec8Value).numberValue))

    private inline fun KBitVec8Value.bv8UnsignedOperation(other: KBitVecValue<*>, op: (UByte, UByte) -> UByte) =
        ctx.mkBv(op(numberValue.toUByte(), (other as KBitVec8Value).numberValue.toUByte()).toByte())

    private inline fun KBitVec16Value.bv16Operation(other: KBitVecValue<*>, op: (Short, Short) -> Short) =
        ctx.mkBv(op(numberValue, (other as KBitVec16Value).numberValue))

    private inline fun KBitVec16Value.bv16UnsignedOperation(other: KBitVecValue<*>, op: (UShort, UShort) -> UShort) =
        ctx.mkBv(op(numberValue.toUShort(), (other as KBitVec16Value).numberValue.toUShort()).toShort())

    private inline fun KBitVec32Value.bv32Operation(other: KBitVecValue<*>, op: (Int, Int) -> Int) =
        ctx.mkBv(op(numberValue, (other as KBitVec32Value).numberValue))

    private inline fun KBitVec32Value.bv32UnsignedOperation(other: KBitVecValue<*>, op: (UInt, UInt) -> UInt) =
        ctx.mkBv(op(numberValue.toUInt(), (other as KBitVec32Value).numberValue.toUInt()).toInt())

    private inline fun KBitVec64Value.bv64Operation(other: KBitVecValue<*>, op: (Long, Long) -> Long) =
        ctx.mkBv(op(numberValue, (other as KBitVec64Value).numberValue))

    private inline fun KBitVec64Value.bv64UnsignedOperation(other: KBitVecValue<*>, op: (ULong, ULong) -> ULong) =
        ctx.mkBv(op(numberValue.toULong(), (other as KBitVec64Value).numberValue.toULong()).toLong())

    private inline fun KBitVecValue<*>.bvOperationDefault(
        rhs: KBitVecValue<*>,
        signed: Boolean = false,
        operation: (BigInteger, BigInteger) -> BigInteger
    ): KBitVecValue<*> = with(ctx) {
        val lhs = this@bvOperationDefault
        val size = lhs.sort.sizeBits
        val lValue = lhs.stringValue.let { if (!signed) unsignedBigIntFromBinary(it) else signedBigIntFromBinary(it) }
        val rValue = rhs.stringValue.let { if (!signed) unsignedBigIntFromBinary(it) else signedBigIntFromBinary(it) }
        val resultValue = operation(lValue, rValue)
        mkBvFromBigInteger(resultValue, size)
    }

    private fun signedBigIntFromBinary(value: String): BigInteger {
        var result = BigInteger(value, 2)
        val maxValue = BigInteger.valueOf(2).pow(value.length - 1)
        if (result >= maxValue) {
            result -= BigInteger.valueOf(2).pow(value.length)
        }
        return result
    }

    private fun unsignedBigIntFromBinary(value: String): BigInteger =
        BigInteger(value, 2)

    private fun unsignedBinaryString(value: BigInteger): String =
        value.toString(2)

}
