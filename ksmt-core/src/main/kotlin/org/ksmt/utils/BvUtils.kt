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
        1 -> mkBv(true) // sign bit 1
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
        1 -> mkBv(false) // sign bit 0
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
        else -> signedBigIntFromBinary(stringValue) >= other.toBigInteger()
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
        val normalizedValue = value.normalizeValue(size)
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

    private inline fun KBitVecValue<*>.bvOperationDefault(
        other: KBitVecValue<*>,
        signed: Boolean = false,
        crossinline operation: (BigInteger, BigInteger) -> BigInteger
    ): KBitVecValue<*> = bvNumericOperation(
        this, other,
        unwrap = { if (!signed) unsignedBigIntFromBinary(it.stringValue) else signedBigIntFromBinary(it.stringValue) },
        wrap = {
            val size = this.sort.sizeBits
            ctx.mkBvFromBigInteger(it, size)
        },
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

    private fun BigInteger.normalizeValue(size: UInt): BigInteger =
        this.mod(BigInteger.valueOf(2).pow(size.toInt()))

}
