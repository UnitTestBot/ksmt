package io.ksmt.runner.serializer

import com.jetbrains.rd.framework.AbstractBuffer
import java.math.BigInteger

fun AbstractBuffer.writeBigInteger(value: BigInteger) {
    val bytes = value.toByteArray()
    writeByteArray(bytes)
}

fun AbstractBuffer.readBigInteger(): BigInteger {
    val bytes = readByteArray()
    return BigInteger(bytes)
}
