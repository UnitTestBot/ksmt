@file:Suppress("MagicNumber")

package org.ksmt.solver.bitwuzla.bindings

/**
 * The base for strings representing bit-vector values.
 */
enum class BitwuzlaBVBase(val value: Int) {
    BINARY(2), DECIMAL(10), HEXADECIMAL(16);

    val nativeValue: Byte
        get() = value.toByte()
}
