package org.ksmt.solver.bitwuzla.bindings

/**
 * @param sign Binary string representation of the sign bit.
 * @param exponent Binary string representation of the exponent bit-vector value.
 * @param significand Binary string representation of the significand bit-vector value.
 * */

class FpValue {
    @JvmField
    var sign: String = ""
    @JvmField
    var exponent: String = ""
    @JvmField
    var significand: String = ""
}
