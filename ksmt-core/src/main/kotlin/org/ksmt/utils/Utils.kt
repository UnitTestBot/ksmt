package org.ksmt.utils

import org.ksmt.sort.KFp16Sort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpSort

// We can have here `0` as a pad symbol since `toString` can return a string
// containing fewer symbols than `sizeBits` only for non-negative numbers
fun Number.toBinary(): String = when (this) {
    is Byte -> toUByte().toString(radix = 2).padStart(Byte.SIZE_BITS, '0')
    is Short -> toUShort().toString(radix = 2).padStart(Short.SIZE_BITS, '0')
    is Int -> toUInt().toString(radix = 2).padStart(Int.SIZE_BITS, '0')
    is Long -> toULong().toString(radix = 2).padStart(Long.SIZE_BITS, '0')
    is Float -> toRawBits().toBinary()
    is Double -> toRawBits().toBinary()
    else -> error("Unsupported type for transformation into a binary string: ${this::class.simpleName}")
}

/**
 * Significand for Fp16 takes 10 bits from the float value: from 14 to 23 bits
 */
val Float.halfPrecisionSignificand: Int
    get() = (toRawBits() and 0b0000_0000_0111_1111_1110_0000_0000_0000) shr 13

/**
 * Take an exponent from the float value. Depending on the [isBiased] is can be either shifted by
 * [KFp16Sort.exponentShiftSize] or not.
 */
fun Float.getHalfPrecisionExponent(isBiased: Boolean): Int {
    // take an unbiased exponent from the given value
    val unbiasedFloatExponent = getExponent(isBiased = false)
    // extract a sign bit from it -- fifth one
    val signBit = (unbiasedFloatExponent shr 4) and 1
    // take remaining bits of the exponent
    val otherBits = unbiasedFloatExponent and 0b1111
    // create an unbiased fp16 exponent containing five bits
    val unbiasedFp16Exponent = (signBit shl 4) or otherBits

    return if (isBiased) unbiasedFp16Exponent + KFp16Sort.exponentShiftSize else unbiasedFp16Exponent
}

/**
 * Extracts a significand from the float value.
 *
 * @see [Float.toRawBits]
 */
val Float.significand: Int get() = toRawBits() and 0x7fffff

/**
 * Extracts an exponent from the float value.
 * Depending on the [isBiased] it can be either biased or not.
 *
 * @see [Float.toRawBits]
 */
fun Float.getExponent(isBiased: Boolean): Int {
    // extract exponent using the mask and move it to the right without saving the sign
    val exponent = (toRawBits() and 0x7f80_0000) ushr 23

    return if (isBiased) exponent else exponent - KFp32Sort.exponentShiftSize
}

val Float.signBit: Int get() = (toRawBits() shr 31) and 1
val Float.booleanSignBit: Boolean get() = signBit == 1

/**
 * Extracts a significand from the float value.
 *
 * @see [Double.toRawBits]
 */
val Double.significand: Long get() = toRawBits() and 0x000f_ffff_ffff_ffff

/**
 * Extracts an exponent from the float value.
 * Depending on the [isBiased] it can be either biased or not.
 *
 * @see [Double.toRawBits]
 */
fun Double.getExponent(isBiased: Boolean): Long {
    // extract exponent using the mask and move it to the right without saving the sign
    val exponent = (toRawBits() and 0x7ff0_0000_0000_0000) ushr 52

    return if (isBiased) exponent else exponent - KFp64Sort.exponentShiftSize
}

val Double.signBit: Int get() = (toRawBits() shr 63).toInt() and 1
val Double.booleanSignBit: Boolean get() = signBit == 1

/**
 * Extracts a significand of the specific [sort] from a float value.
 */
fun Float.extractSignificand(sort: KFpSort): Int {
    val sizeBits = sort.significandBits.toInt()
    // If we need more bits in significand than we have in Fp32's significand, take all ones
    // Otherwise, take `sizeBits - 1` of them.
    val fp32SignificandBits = KFp32Sort.significandBits.toInt()
    val significandMask = if (sizeBits >= fp32SignificandBits) -1 else (1 shl sizeBits - 1) - 1

    // we want to take first `n` bits from the float significand.
    // We take at least zero to avoid incorrect shift for sorts with a large significand bits number.
    val shiftForSortSpecificSignificandBits = maxOf(0, fp32SignificandBits - sizeBits)

    return (toRawBits() shr shiftForSortSpecificSignificandBits) and significandMask
}

/**
 * Extends the given 32-bits value to a new 64-bits one with zeroes.
 *
 * We should use this function instead of [Int.toLong] to avoid moving size bit from its place.
 *
 * It is important in situations when we want to create a bitvector from (-1) -- it is 32 ones in binary representation.
 * Then we want to extend it to 64 bits, but [Int.toLong] will move sign one into the first place in a new value,
 * therefore we will have not a value of 32 zeros following by 32 ones, but 100...0011..11, that will change
 * the value of the bitvector we wanted to construct.
 */
fun Int.extendWithLeadingZeros(): Long = toLong().let {
    ((((it shr 63) and 1) shl 31) or it) and 0xffff_ffff
}

/**
 * Extracts an exponent of the specific [sort] from the fp32 value.
 */
fun Float.extractExponent(sort: KFpSort, isBiased: Boolean): Int {
    // extract an exponent from the value in fp32 format
    val exponent = getExponent(isBiased = false)

    // find a sign of the exponent
    val sign = (exponent shr 7) and 1

    // take last (exponentBits - 1) bits of the exponent, where exponentBits is specific for the sort
    val exponentBits = exponent and ((1 shl (sort.exponentBits.toInt() - 1)) - 1)

    // construct a sort specific exponent
    val constructedExponent = (sign shl sort.exponentBits.toInt() - 1) or exponentBits

    return if (isBiased) constructedExponent + sort.exponentShiftSize() else constructedExponent
}


/**
 * Extracts a significand of the specific [sort] from a double value.
 */
fun Double.extractSignificand(sort: KFpSort): Long {
    val significandBits = sort.significandBits.toInt()
    val fp64Bits = KFp64Sort.significandBits.toInt()

    // If we need more bits in significand than we have in Fp32's significand, take all ones
    // Otherwise, take `sizeBits - 1` of them.
    val significandMask = if (significandBits >= fp64Bits) -1 else (1L shl significandBits - 1) - 1
    // we want to take first `n` bits from the float significand.
    // We take at least zero to avoid incorrect shift for sorts with a large significand bits number.
    val shiftForSortSpecificSignificandBits = maxOf(0, 53 - significandBits)

    return (toRawBits() shr shiftForSortSpecificSignificandBits) and significandMask
}

/**
 * Extracts an exponent of the specific [sort] from the fp64 value.
 */
fun Double.extractExponent(sort: KFpSort, isBiased: Boolean): Long {
    val exponent = getExponent(isBiased = false)
    val sign = (exponent shr 10) and 1
    val exponentBits = exponent and (1L shl (sort.exponentBits.toInt() - 1)) - 1

    val constructedExponent = (sign shl sort.exponentBits.toInt() - 1) or exponentBits

    return if (isBiased) constructedExponent + sort.exponentShiftSize() else constructedExponent
}

inline fun <reified T, reified Base> Base.cast(): T where T : Base = this as T

@Suppress("UNCHECKED_CAST")
fun <Base, T> Base.uncheckedCast(): T = this as T
