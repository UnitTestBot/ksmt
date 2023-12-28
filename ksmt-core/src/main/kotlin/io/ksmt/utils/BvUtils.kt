package io.ksmt.utils

import io.ksmt.KContext
import io.ksmt.expr.KBitVec16Value
import io.ksmt.expr.KBitVec1Value
import io.ksmt.expr.KBitVec32Value
import io.ksmt.expr.KBitVec64Value
import io.ksmt.expr.KBitVec8Value
import io.ksmt.expr.KBitVecCustomValue
import io.ksmt.expr.KBitVecNumberValue
import io.ksmt.expr.KBitVecValue
import io.ksmt.sort.KBvSort
import java.math.BigInteger
import kotlin.experimental.and
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

    @JvmStatic
    fun <T : KBvSort> KContext.bvMinValueSigned(size: UInt): KBitVecValue<T> =
        mkBvSpecialValue(size, bvMinValueSigned)

    @JvmStatic
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

    @JvmStatic
    fun <T : KBvSort> KContext.bvMaxValueSigned(size: UInt): KBitVecValue<T> =
        mkBvSpecialValue(size, bvMaxValueSigned)

    @JvmStatic
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

    @JvmStatic
    fun <T : KBvSort> KContext.bvMaxValueUnsigned(size: UInt): KBitVecValue<T> =
        mkBvSpecialValue(size, bvMaxValueUnsigned)

    @JvmStatic
    fun KBitVecValue<*>.isBvMaxValueUnsigned(): Boolean = isBvSpecialValue(bvMaxValueUnsigned)

    private val bvZeroValue = object : BvSpecialValueSource {
        override val bv1: Boolean = false
        override val bv8: Byte = 0
        override val bv16: Short = 0
        override val bv32: Int = 0
        override val bv64: Long = 0

        override fun bvDefault(size: UInt): BigInteger = BigInteger.ZERO
    }

    @JvmStatic
    fun <T : KBvSort> KContext.bvZero(size: UInt): KBitVecValue<T> =
        mkBvSpecialValue(size, bvZeroValue)

    @JvmStatic
    fun KBitVecValue<*>.isBvZero(): Boolean = isBvSpecialValue(bvZeroValue)

    private val bvOneValue = object : BvSpecialValueSource {
        override val bv1: Boolean = true
        override val bv8: Byte = 1
        override val bv16: Short = 1
        override val bv32: Int = 1
        override val bv64: Long = 1L

        override fun bvDefault(size: UInt): BigInteger = BigInteger.ONE
    }

    @JvmStatic
    fun <T : KBvSort> KContext.bvOne(size: UInt): KBitVecValue<T> =
        mkBvSpecialValue(size, bvOneValue)

    @JvmStatic
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

    @JvmStatic
    fun <T : KBvSort> KContext.bvValue(size: UInt, value: Int): KBitVecValue<T> =
        mkBvSpecialValue(size, BvIntValue(value))

    @JvmStatic
    fun KBitVecValue<*>.bvValueIs(value: Int): Boolean = isBvSpecialValue(BvIntValue(value))

    @JvmStatic
    fun KBitVecValue<*>.getBit(bit: UInt): Boolean {
        check(bit < sort.sizeBits) { "Requested bit is out of bounds for $sort" }
        return when (this) {
            is KBitVec1Value -> value
            is KBitVecNumberValue<*, *> -> ((numberValue.toLong() shr bit.toInt()) and 0x1L) == 0x1L
            is KBitVecCustomValue -> value.testBit(bit.toInt())
            else -> stringValue.let { it[it.lastIndex - bit.toInt()] == '1' }
        }
    }

    @JvmStatic
    fun KBitVecValue<*>.signBit(): Boolean = getBit(sort.sizeBits - 1u)

    @JvmStatic
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

        is KBitVec8Value -> byteValue >= other
        is KBitVec16Value -> shortValue >= other
        is KBitVec32Value -> intValue >= other
        is KBitVec64Value -> longValue >= other
        is KBitVecCustomValue -> value.signedValue(sizeBits) >= other.toBigInteger()
        else -> stringValue.toBigInteger(radix = 2).signedValue(sort.sizeBits) >= other.toBigInteger()
    }

    @JvmStatic
    @Suppress("ComplexMethod")
    fun KBitVecValue<*>.signedLessOrEqual(other: KBitVecValue<*>): Boolean = when {
        this is KBitVec1Value && other is KBitVec1Value -> if (value) {
            // 1 <= 0 -> true
            // 1 <= 1 -> true
            true
        } else {
            // 0 <= 1 -> false
            // 0 <= 0 -> true
            !other.value
        }

        this is KBitVec8Value && other is KBitVec8Value -> byteValue <= other.byteValue
        this is KBitVec16Value && other is KBitVec16Value -> shortValue <= other.shortValue
        this is KBitVec32Value && other is KBitVec32Value -> intValue <= other.intValue
        this is KBitVec64Value && other is KBitVec64Value -> longValue <= other.longValue
        this is KBitVecCustomValue && other is KBitVecCustomValue -> {
            val lhs = value.signedValue(sizeBits)
            val rhs = other.value.signedValue(sizeBits)
            lhs <= rhs
        }

        else -> {
            val lhs = stringValue.toBigInteger(radix = 2).signedValue(sort.sizeBits)
            val rhs = other.stringValue.toBigInteger(radix = 2).signedValue(sort.sizeBits)
            lhs <= rhs
        }
    }

    @JvmStatic
    fun KBitVecValue<*>.unsignedLessOrEqual(other: KBitVecValue<*>): Boolean = when {
        this is KBitVec1Value && other is KBitVec1Value -> value <= other.value
        this is KBitVec8Value && other is KBitVec8Value -> byteValue.toUByte() <= other.byteValue.toUByte()
        this is KBitVec16Value && other is KBitVec16Value -> shortValue.toUShort() <= other.shortValue.toUShort()
        this is KBitVec32Value && other is KBitVec32Value -> intValue.toUInt() <= other.intValue.toUInt()
        this is KBitVec64Value && other is KBitVec64Value -> longValue.toULong() <= other.longValue.toULong()
        this is KBitVecCustomValue && other is KBitVecCustomValue -> value <= other.value
        // MSB first -> lexical order works
        else -> stringValue <= other.stringValue
    }

    @JvmStatic
    fun KBitVecValue<*>.signedLess(other: KBitVecValue<*>): Boolean =
        signedLessOrEqual(other) && this != other

    @JvmStatic
    fun KBitVecValue<*>.unsignedLess(other: KBitVecValue<*>): Boolean =
        unsignedLessOrEqual(other) && this != other

    @JvmStatic
    fun KBitVecValue<*>.signedGreaterOrEqual(other: KBitVecValue<*>): Boolean =
        other.signedLessOrEqual(this)

    @JvmStatic
    fun KBitVecValue<*>.unsignedGreaterOrEqual(other: KBitVecValue<*>): Boolean =
        other.unsignedLessOrEqual(this)

    @JvmStatic
    fun KBitVecValue<*>.signedGreater(other: KBitVecValue<*>): Boolean =
        other.signedLess(this)

    @JvmStatic
    fun KBitVecValue<*>.unsignedGreater(other: KBitVecValue<*>): Boolean =
        other.unsignedLess(this)

    @JvmStatic
    operator fun <T : KBvSort> KBitVecValue<T>.unaryMinus(): KBitVecValue<T> = bvOperation(
        other = this,
        bv1 = { a, _ -> a xor false },
        bv8 = { a, _ -> (-a).toByte() },
        bv16 = { a, _ -> (-a).toShort() },
        bv32 = { a, _ -> -a },
        bv64 = { a, _ -> -a },
        bvDefault = { a, _ -> -a },
    )

    @JvmStatic
    operator fun <T : KBvSort> KBitVecValue<T>.plus(other: KBitVecValue<T>): KBitVecValue<T> = bvOperation(
        other = other,
        bv1 = { a, b -> a xor b },
        bv8 = { a, b -> (a + b).toByte() },
        bv16 = { a, b -> (a + b).toShort() },
        bv32 = { a, b -> a + b },
        bv64 = { a, b -> a + b },
        bvDefault = { a, b -> a + b },
    )

    @JvmStatic
    operator fun <T : KBvSort> KBitVecValue<T>.minus(other: KBitVecValue<T>): KBitVecValue<T> = bvOperation(
        other = other,
        bv1 = { a, b -> a xor b },
        bv8 = { a, b -> (a - b).toByte() },
        bv16 = { a, b -> (a - b).toShort() },
        bv32 = { a, b -> a - b },
        bv64 = { a, b -> a - b },
        bvDefault = { a, b -> a - b },
    )

    @JvmStatic
    operator fun <T : KBvSort> KBitVecValue<T>.times(other: KBitVecValue<T>): KBitVecValue<T> = bvOperation(
        other = other,
        bv1 = { a, b -> a && b },
        bv8 = { a, b -> (a * b).toByte() },
        bv16 = { a, b -> (a * b).toShort() },
        bv32 = { a, b -> a * b },
        bv64 = { a, b -> a * b },
        bvDefault = { a, b -> a * b },
    )

    @JvmStatic
    fun <T : KBvSort> KBitVecValue<T>.signedDivide(other: KBitVecValue<T>): KBitVecValue<T> = bvOperation(
        other = other,
        signIsImportant = true,
        bv1 = { a, _ -> a },
        bv8 = { a, b -> (a / b).toByte() },
        bv16 = { a, b -> (a / b).toShort() },
        bv32 = { a, b -> a / b },
        bv64 = { a, b -> a / b },
        bvDefault = { a, b -> a / b },
    )

    @JvmStatic
    fun <T : KBvSort> KBitVecValue<T>.unsignedDivide(other: KBitVecValue<T>): KBitVecValue<T> = bvUnsignedOperation(
        other = other,
        bv1 = { a, _ -> a },
        bv8 = { a, b -> (a / b).toUByte() },
        bv16 = { a, b -> (a / b).toUShort() },
        bv32 = { a, b -> a / b },
        bv64 = { a, b -> a / b },
        bvDefault = { a, b -> a / b },
    )

    @JvmStatic
    fun <T : KBvSort> KBitVecValue<T>.signedRem(other: KBitVecValue<T>): KBitVecValue<T> = bvOperation(
        other = other,
        signIsImportant = true,
        bv1 = { _, _ -> false },
        bv8 = { a, b -> (a.rem(b)).toByte() },
        bv16 = { a, b -> (a.rem(b)).toShort() },
        bv32 = { a, b -> a.rem(b) },
        bv64 = { a, b -> a.rem(b) },
        bvDefault = { a, b -> a.rem(b) },
    )

    @JvmStatic
    fun <T : KBvSort> KBitVecValue<T>.unsignedRem(other: KBitVecValue<T>): KBitVecValue<T> = bvUnsignedOperation(
        other = other,
        bv1 = { _, _ -> false },
        bv8 = { a, b -> (a.rem(b)).toUByte() },
        bv16 = { a, b -> (a.rem(b)).toUShort() },
        bv32 = { a, b -> a.rem(b) },
        bv64 = { a, b -> a.rem(b) },
        bvDefault = { a, b -> a.rem(b) },
    )

    @JvmStatic
    fun <T : KBvSort> KBitVecValue<T>.signedMod(other: KBitVecValue<T>): KBitVecValue<T> = bvOperation(
        other = other,
        signIsImportant = true,
        bv1 = { _, _ -> false },
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

    @JvmStatic
    fun <T : KBvSort> KBitVecValue<T>.bitwiseNot(): KBitVecValue<T> = bvOperation(
        other = this,
        bv1 = { a, _ -> a.not() },
        bv8 = { a, _ -> a.inv() },
        bv16 = { a, _ -> a.inv() },
        bv32 = { a, _ -> a.inv() },
        bv64 = { a, _ -> a.inv() },
        bvDefault = { a, _ -> a.inv() },
    )

    @JvmStatic
    fun <T : KBvSort> KBitVecValue<T>.bitwiseOr(other: KBitVecValue<T>): KBitVecValue<T> = bvOperation(
        other = other,
        bv1 = { a, b -> a || b },
        bv8 = { a, b -> a or b },
        bv16 = { a, b -> a or b },
        bv32 = { a, b -> a or b },
        bv64 = { a, b -> a or b },
        bvDefault = { a, b -> a or b },
    )

    @JvmStatic
    fun <T : KBvSort> KBitVecValue<T>.bitwiseXor(other: KBitVecValue<T>): KBitVecValue<T> = bvOperation(
        other = other,
        bv1 = { a, b -> a xor b },
        bv8 = { a, b -> a xor b },
        bv16 = { a, b -> a xor b },
        bv32 = { a, b -> a xor b },
        bv64 = { a, b -> a xor b },
        bvDefault = { a, b -> a xor b },
    )

    @JvmStatic
    fun <T : KBvSort> KBitVecValue<T>.bitwiseAnd(other: KBitVecValue<T>): KBitVecValue<T> = bvOperation(
        other = other,
        bv1 = { a, b -> a && b },
        bv8 = { a, b -> a and b },
        bv16 = { a, b -> a and b },
        bv32 = { a, b -> a and b },
        bv64 = { a, b -> a and b },
        bvDefault = { a, b -> a and b },
    )

    private val bv8PossibleShift = 0 until Byte.SIZE_BITS
    private val bv16PossibleShift = 0 until Short.SIZE_BITS
    private val bv32PossibleShift = 0 until Int.SIZE_BITS
    private val bv64PossibleShift = 0 until Long.SIZE_BITS

    @JvmStatic
    fun <T : KBvSort> KBitVecValue<T>.shiftLeft(other: KBitVecValue<T>): KBitVecValue<T> = bvOperation(
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

    @JvmStatic
    fun <T : KBvSort> KBitVecValue<T>.shiftRightLogical(other: KBitVecValue<T>): KBitVecValue<T> = bvUnsignedOperation(
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

    @JvmStatic
    fun <T : KBvSort> KBitVecValue<T>.shiftRightArith(other: KBitVecValue<T>): KBitVecValue<T> = bvOperation(
        other = other,
        signIsImportant = true,
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

    @JvmStatic
    fun KBitVecValue<*>.powerOfTwoOrNull(): Int? {
        val value = bigIntValue()
        val valueMinusOne = value - BigInteger.ONE
        if ((value and valueMinusOne) != BigInteger.ZERO) return null
        return valueMinusOne.bitLength()
    }

    @JvmStatic
    fun KBitVecValue<*>.bigIntValue(): BigInteger = when (this) {
        is KBitVec1Value -> if (value) BigInteger.ONE else BigInteger.ZERO
        is KBitVecNumberValue<*, *> -> numberValue.toBigInteger()
        is KBitVecCustomValue -> value
        else -> stringValue.toBigInteger(radix = 2)
    }

    @JvmStatic
    fun KBitVecValue<*>.toBigIntegerSigned(): BigInteger =
        toBigIntegerUnsigned().signedValue(sort.sizeBits)

    @JvmStatic
    fun KBitVecValue<*>.toBigIntegerUnsigned(): BigInteger =
        bigIntValue().normalizeValue(sort.sizeBits)

    @JvmStatic
    fun concatBv(lhs: KBitVecValue<*>, rhs: KBitVecValue<*>): KBitVecValue<*> = with(lhs.ctx) {
        when {
            lhs is KBitVec8Value && rhs is KBitVec8Value -> {
                var result = lhs.byteValue.toUByte().toInt() shl Byte.SIZE_BITS
                result = result or rhs.byteValue.toUByte().toInt()
                mkBv(result.toShort())
            }

            lhs is KBitVec16Value && rhs is KBitVec16Value -> {
                var result = lhs.shortValue.toUShort().toInt() shl Short.SIZE_BITS
                result = result or rhs.shortValue.toUShort().toInt()
                mkBv(result)
            }

            lhs is KBitVec32Value && rhs is KBitVec32Value -> {
                var result = lhs.intValue.toUInt().toLong() shl Int.SIZE_BITS
                result = result or rhs.intValue.toUInt().toLong()
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

    @JvmStatic
    fun KBitVecValue<*>.extractBv(high: Int, low: Int): KBitVecValue<*> = with(ctx) {
        val size = (high - low + 1).toUInt()
        val value = bigIntValue().normalizeValue(sort.sizeBits)
        val trimLowerBits = value.shiftRight(low)
        val trimHigherBits = trimLowerBits.and(binaryOnes(onesCount = size))
        mkBv(trimHigherBits, size)
    }

    @JvmStatic
    fun KBitVecValue<*>.signExtension(extensionSize: UInt): KBitVecValue<*> =
        if (!signBit()) {
            zeroExtension(extensionSize)
        } else {
            val extension = binaryOnes(onesCount = extensionSize)
            val value = bigIntValue().normalizeValue(sort.sizeBits)
            val extendedValue = concatValues(extension, value, sort.sizeBits)
            ctx.mkBv(extendedValue, sort.sizeBits + extensionSize)
        }

    @JvmStatic
    fun KBitVecValue<*>.zeroExtension(extensionSize: UInt): KBitVecValue<*> = with(ctx) {
        mkBv(bigIntValue().normalizeValue(sort.sizeBits), sort.sizeBits + extensionSize)
    }

    // Add max value without creation of Bv expr
    @JvmStatic
    fun <T : KBvSort> KBitVecValue<T>.addMaxValueSigned(): KBitVecValue<T> = bvOperation(
        other = this,
        bv1 = { a, _ -> a xor bvMaxValueSigned.bv1 },
        bv8 = { a, _ -> (a + bvMaxValueSigned.bv8).toByte() },
        bv16 = { a, _ -> (a + bvMaxValueSigned.bv16).toShort() },
        bv32 = { a, _ -> a + bvMaxValueSigned.bv32 },
        bv64 = { a, _ -> a + bvMaxValueSigned.bv64 },
        bvDefault = { a, _ -> a + bvMaxValueSigned.bvDefault(sort.sizeBits) },
    )

    // Subtract max value without creation of Bv expr
    @JvmStatic
    fun <T : KBvSort> KBitVecValue<T>.subMaxValueSigned(): KBitVecValue<T> = bvOperation(
        other = this,
        bv1 = { a, _ -> a xor bvMaxValueSigned.bv1 },
        bv8 = { a, _ -> (a - bvMaxValueSigned.bv8).toByte() },
        bv16 = { a, _ -> (a - bvMaxValueSigned.bv16).toShort() },
        bv32 = { a, _ -> a - bvMaxValueSigned.bv32 },
        bv64 = { a, _ -> a - bvMaxValueSigned.bv64 },
        bvDefault = { a, _ -> a - bvMaxValueSigned.bvDefault(sort.sizeBits) },
    )

    @JvmStatic
    private fun binaryOnes(onesCount: UInt): BigInteger = powerOfTwo(onesCount) - BigInteger.ONE

    @JvmStatic
    private fun concatValues(lhs: BigInteger, rhs: BigInteger, rhsSize: UInt): BigInteger =
        lhs.shiftLeft(rhsSize.toInt()).or(rhs)

    @Suppress("LongParameterList")
    private inline fun <T : KBvSort> KBitVecValue<*>.bvUnsignedOperation(
        other: KBitVecValue<*>,
        crossinline bv1: (Boolean, Boolean) -> Boolean,
        crossinline bv8: (UByte, UByte) -> UByte,
        crossinline bv16: (UShort, UShort) -> UShort,
        crossinline bv32: (UInt, UInt) -> UInt,
        crossinline bv64: (ULong, ULong) -> ULong,
        crossinline bvDefault: (BigInteger, BigInteger) -> BigInteger,
    ): KBitVecValue<T> = when {
        this@bvUnsignedOperation is KBitVec1Value && other is KBitVec1Value -> bv1Operation(other, op = bv1)
        this@bvUnsignedOperation is KBitVec8Value && other is KBitVec8Value -> bv8UnsignedOperation(other, op = bv8)
        this@bvUnsignedOperation is KBitVec16Value && other is KBitVec16Value -> bv16UnsignedOperation(other, op = bv16)
        this@bvUnsignedOperation is KBitVec32Value && other is KBitVec32Value -> bv32UnsignedOperation(other, op = bv32)
        this@bvUnsignedOperation is KBitVec64Value && other is KBitVec64Value -> bv64UnsignedOperation(other, op = bv64)
        this@bvUnsignedOperation is KBitVecCustomValue && other is KBitVecCustomValue ->
            bvCustomOperation(other, signed = false, operation = bvDefault)
        else -> bvOperationDefault(other, signed = false, operation = bvDefault)
    }.uncheckedCast()

    @Suppress("LongParameterList")
    private inline fun <T : KBvSort> KBitVecValue<*>.bvOperation(
        other: KBitVecValue<*>,
        signIsImportant: Boolean = false,
        crossinline bv1: (Boolean, Boolean) -> Boolean,
        crossinline bv8: (Byte, Byte) -> Byte,
        crossinline bv16: (Short, Short) -> Short,
        crossinline bv32: (Int, Int) -> Int,
        crossinline bv64: (Long, Long) -> Long,
        crossinline bvDefault: (BigInteger, BigInteger) -> BigInteger,
    ): KBitVecValue<T> = when {
        this@bvOperation is KBitVec1Value && other is KBitVec1Value -> bv1Operation(other, op = bv1)
        this@bvOperation is KBitVec8Value && other is KBitVec8Value -> bv8Operation(other, op = bv8)
        this@bvOperation is KBitVec16Value && other is KBitVec16Value -> bv16Operation(other, op = bv16)
        this@bvOperation is KBitVec32Value && other is KBitVec32Value -> bv32Operation(other, op = bv32)
        this@bvOperation is KBitVec64Value && other is KBitVec64Value -> bv64Operation(other, op = bv64)
        this@bvOperation is KBitVecCustomValue && other is KBitVecCustomValue ->
            bvCustomOperation(other, signed = signIsImportant, operation = bvDefault)
        else -> bvOperationDefault(other, signed = signIsImportant, operation = bvDefault)
    }.uncheckedCast()

    private inline fun KBitVec1Value.bv1Operation(
        other: KBitVec1Value,
        crossinline op: (Boolean, Boolean) -> Boolean
    ): KBitVec1Value = bvNumericOperation(
        this, other,
        unwrap = { it.value },
        wrap = ctx::mkBv,
        op = op
    )

    private inline fun KBitVec8Value.bv8Operation(
        other: KBitVec8Value,
        crossinline op: (Byte, Byte) -> Byte
    ): KBitVec8Value = bvNumericOperation(
        this, other,
        unwrap = { it.byteValue },
        wrap = ctx::mkBv,
        op = op
    )

    private inline fun KBitVec8Value.bv8UnsignedOperation(
        other: KBitVec8Value,
        crossinline op: (UByte, UByte) -> UByte
    ): KBitVec8Value = bvNumericOperation(
        this, other,
        unwrap = { it.byteValue.toUByte() },
        wrap = { ctx.mkBv(it.toByte()) },
        op = op
    )

    private inline fun KBitVec16Value.bv16Operation(
        other: KBitVec16Value,
        crossinline op: (Short, Short) -> Short
    ): KBitVec16Value = bvNumericOperation(
        this, other,
        unwrap = { it.shortValue },
        wrap = ctx::mkBv,
        op = op
    )

    private inline fun KBitVec16Value.bv16UnsignedOperation(
        other: KBitVec16Value,
        crossinline op: (UShort, UShort) -> UShort
    ): KBitVec16Value = bvNumericOperation(
        this, other,
        unwrap = { it.shortValue.toUShort() },
        wrap = { ctx.mkBv(it.toShort()) },
        op = op
    )

    private inline fun KBitVec32Value.bv32Operation(
        other: KBitVec32Value,
        crossinline op: (Int, Int) -> Int
    ): KBitVec32Value = bvNumericOperation(
        this, other,
        unwrap = { it.intValue },
        wrap = ctx::mkBv,
        op = op
    )

    private inline fun KBitVec32Value.bv32UnsignedOperation(
        other: KBitVec32Value,
        crossinline op: (UInt, UInt) -> UInt
    ): KBitVec32Value = bvNumericOperation(
        this, other,
        unwrap = { it.intValue.toUInt() },
        wrap = { ctx.mkBv(it.toInt()) },
        op = op
    )

    private inline fun KBitVec64Value.bv64Operation(
        other: KBitVec64Value,
        crossinline op: (Long, Long) -> Long
    ): KBitVec64Value = bvNumericOperation(
        this, other,
        unwrap = { it.longValue },
        wrap = ctx::mkBv,
        op = op
    )

    private inline fun KBitVec64Value.bv64UnsignedOperation(
        other: KBitVec64Value,
        crossinline op: (ULong, ULong) -> ULong
    ): KBitVec64Value = bvNumericOperation(
        this, other,
        unwrap = { it.longValue.toULong() },
        wrap = { ctx.mkBv(it.toLong()) },
        op = op
    )

    private inline fun KBitVecCustomValue.bvCustomOperation(
        other: KBitVecCustomValue,
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
        arg0: T, arg1: T,
        crossinline unwrap: (T) -> V,
        crossinline wrap: (V) -> T,
        crossinline op: (V, V) -> V
    ): T = wrap(op(unwrap(arg0), unwrap(arg1)))

    @JvmStatic
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

    @JvmStatic
    private fun <T : KBvSort> KContext.mkBvSpecialValue(size: UInt, source: BvSpecialValueSource): KBitVecValue<T> =
        when (size.toInt()) {
            1 -> mkBv(source.bv1)
            Byte.SIZE_BITS -> mkBv(source.bv8)
            Short.SIZE_BITS -> mkBv(source.bv16)
            Int.SIZE_BITS -> mkBv(source.bv32)
            Long.SIZE_BITS -> mkBv(source.bv64)
            else -> mkBv(source.bvDefault(size), size)
        }.uncheckedCast()

    @JvmStatic
    private fun KBitVecValue<*>.isBvSpecialValue(source: BvSpecialValueSource) = when (this) {
        is KBitVec1Value -> value == source.bv1
        is KBitVec8Value -> byteValue == source.bv8
        is KBitVec16Value -> shortValue == source.bv16
        is KBitVec32Value -> intValue == source.bv32
        is KBitVec64Value -> longValue == source.bv64
        is KBitVecCustomValue -> value == source.bvDefault(sizeBits)
        else -> stringValue == ctx.mkBvSpecialValue<KBvSort>(sort.sizeBits, source).stringValue
    }

}
