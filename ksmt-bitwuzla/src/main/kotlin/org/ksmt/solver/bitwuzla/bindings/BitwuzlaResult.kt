package org.ksmt.solver.bitwuzla.bindings

/**
 * A satisfiability result.<br></br>
 * enum values<br></br>
 * *native declaration : bitwuzla.h:2003*
 */
enum class BitwuzlaResult(val value: Int) {
    /** < sat  */
    BITWUZLA_SAT(10),

    /** < unsat  */
    BITWUZLA_UNSAT(20),

    /** < unknown  */
    BITWUZLA_UNKNOWN(0);

    companion object {
        private val valueMapping = BitwuzlaResult.values().associateBy { it.value }
        fun fromValue(value: Int): BitwuzlaResult = valueMapping.getValue(value)
    }
}
