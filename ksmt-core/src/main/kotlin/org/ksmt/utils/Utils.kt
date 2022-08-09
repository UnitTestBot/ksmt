package org.ksmt.utils

// We can have here `0` as a pad symbol since `toString` can return a string
// containing fewer symbols than `sizeBits` only for non-negative numbers
fun Number.toBinary(): String = when (this) {
    is Byte -> toUByte().toString(radix = 2).padStart(Byte.SIZE_BITS, '0')
    is Short -> toUShort().toString(radix = 2).padStart(Short.SIZE_BITS, '0')
    is Int -> toUInt().toString(radix = 2).padStart(Int.SIZE_BITS, '0')
    is Long -> toULong().toString(radix = 2).padStart(Long.SIZE_BITS, '0')
    else -> error("Unsupported type for transformation into a binary string: ${this::class.simpleName}")
}

inline fun <reified T, reified Base> Base.cast(): T where T : Base = this as T

@Suppress("UNCHECKED_CAST")
fun <Base, T> Base.uncheckedCast(): T = this as T