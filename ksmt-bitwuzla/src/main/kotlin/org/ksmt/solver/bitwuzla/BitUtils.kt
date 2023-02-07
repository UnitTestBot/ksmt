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

fun bigIntegerToBvBits(value: BigInteger): IntArray {
    val valueByteArray = value.toByteArray()
    val size = valueByteArray.size / Int.SIZE_BYTES + if (valueByteArray.size % Int.SIZE_BYTES != 0) 1 else 0
    return IntArray(size) {
        val firstByteIdx = valueByteArray.size - Int.SIZE_BYTES - it * Int.SIZE_BYTES
        var intValue = 0
        for (byteIdx in 0 until Int.SIZE_BYTES) {
            val resolvedByteIdx = firstByteIdx + byteIdx
            if (resolvedByteIdx >= 0) {
                val byteValue = valueByteArray[resolvedByteIdx].toUByte().toInt()
                intValue = intValue or (byteValue shl (byteIdx * Byte.SIZE_BITS))
            }
        }
        intValue
    }
}
