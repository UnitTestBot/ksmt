package org.ksmt.solver.bitwuzla

import java.math.BigInteger

fun bvBitsToBigInteger(bvBits: IntArray): BigInteger {
    val valueByteArray = ByteArray(bvBits.size * Int.SIZE_BYTES) {
        val arrayIdx = bvBits.lastIndex - it / Int.SIZE_BYTES
        val byteIdx = Int.SIZE_BYTES - 1 - it % Int.SIZE_BYTES

        val bytes = bvBits[arrayIdx]
        (bytes ushr (byteIdx * Byte.SIZE_BITS) and 0xff).toByte()
    }
    return BigInteger(1, valueByteArray)
}

fun bigIntegerToBvBits(value: BigInteger, sizeBits: Int): IntArray {
    val valueByteArray = value.toByteArray()

    // Trim leading zeroes
    var firstByteIdx = 0
    while (valueByteArray[firstByteIdx] == 0.toByte()) {
        firstByteIdx++
    }

    val intArraySize = sizeBits / Int.SIZE_BITS + if (sizeBits % Int.SIZE_BITS != 0) 1 else 0
    val valueIntArray = IntArray(intArraySize)
    for (byteIdx in valueByteArray.lastIndex downTo firstByteIdx) {
        val reversedIdx = valueByteArray.lastIndex - byteIdx
        val arrayIdx = valueIntArray.lastIndex - reversedIdx / Int.SIZE_BYTES
        val shift = reversedIdx % Int.SIZE_BYTES
        val byteValue = valueByteArray[byteIdx].toInt() and 0xff
        valueIntArray[arrayIdx] = valueIntArray[arrayIdx] or (byteValue shl (shift * Byte.SIZE_BITS))
    }

    return valueIntArray
}
