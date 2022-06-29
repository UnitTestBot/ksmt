package org.ksmt.solver.bitwuzla.bindings

/**
 * The base for strings representing bit-vector values.<br></br>
 * enum values<br></br>
 * *native declaration : bitwuzla.h:26*
 */
enum class BitwuzlaBVBase(val value: Int) {
    /** < binary  */
    BITWUZLA_BV_BASE_BIN(0),

    /** < decimal  */
    BITWUZLA_BV_BASE_DEC(1),

    /** < hexadecimal  */
    BITWUZLA_BV_BASE_HEX(2);

    companion object {
        private val valueMapping = BitwuzlaBVBase.values().associateBy { it.value }
        fun fromValue(value: Int): BitwuzlaBVBase = valueMapping.getValue(value)
    }
}
