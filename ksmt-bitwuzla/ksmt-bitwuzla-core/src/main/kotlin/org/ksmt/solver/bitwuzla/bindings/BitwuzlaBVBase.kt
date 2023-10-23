@file:Suppress("MagicNumber")

package org.ksmt.solver.bitwuzla.bindings

/**
 * The base for strings representing bit-vector values.
 */
@JvmInline
value class BitwuzlaBVBase(val value: UInt) {

    init {
        require(value == 2u || value == 10u || value == 16u) {
            "BV base must be equal to one of: 2/10/16"
        }
    }

    val nativeValue: Byte
        get() = value.toByte()
}
