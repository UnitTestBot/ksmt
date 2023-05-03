package io.ksmt.solver.bitwuzla

import java.math.BigInteger

private const val BYTE_MASK = 0xff

/**
 * Construct BigInteger value from bits array of the form:
 * array[0] = bits[31:0], array[1] = bits[64:32], ...
 * */
fun bvBitsToBigInteger(bvBits: IntArray): BigInteger {
    val valueByteArray = ByteArray(bvBits.size * Int.SIZE_BYTES) {
        val arrayIdx = bvBits.lastIndex - it / Int.SIZE_BYTES
        val byteIdx = Int.SIZE_BYTES - 1 - it % Int.SIZE_BYTES

        val bytes = bvBits[arrayIdx]
        (bytes ushr (byteIdx * Byte.SIZE_BITS) and BYTE_MASK).toByte()
    }
    return BigInteger(1, valueByteArray)
}

/**
 * Convert BigInteger [value] with [sizeBits] bits into
 * array of the form: array[0] = bits[31:0], array[1] = bits[64:32], ...
 * */
fun bigIntegerToBvBits(value: BigInteger, sizeBits: Int): IntArray {
    // array of the form: array[0] = bits[sizeBits:sizeBits-8], ..., array[n] = bits[8:0]
    val valueByteArray = value.toByteArray()

    // Trim leading zeroes
    var firstByteIdx = 0
    while (firstByteIdx < valueByteArray.size && valueByteArray[firstByteIdx] == 0.toByte()) {
        firstByteIdx++
    }

    val intArraySize = sizeBits / Int.SIZE_BITS + if (sizeBits % Int.SIZE_BITS != 0) 1 else 0
    val valueIntArray = IntArray(intArraySize)
    for (byteIdx in firstByteIdx..valueByteArray.lastIndex) {
        val reversedIdx = valueByteArray.lastIndex - byteIdx
        val arrayIdx = valueIntArray.lastIndex - reversedIdx / Int.SIZE_BYTES
        val shift = reversedIdx % Int.SIZE_BYTES
        val byteValue = valueByteArray[byteIdx].toInt() and BYTE_MASK
        valueIntArray[arrayIdx] = valueIntArray[arrayIdx] or (byteValue shl (shift * Byte.SIZE_BITS))
    }

    return valueIntArray
}
