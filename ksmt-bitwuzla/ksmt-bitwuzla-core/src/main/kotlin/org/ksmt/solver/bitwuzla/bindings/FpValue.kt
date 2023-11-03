package org.ksmt.solver.bitwuzla.bindings

/**
 * @property sign Binary string representation of the sign bit.
 * @property exponent Binary string representation of the exponent bit-vector value.
 * @property significand Binary string representation of the significand bit-vector value.
 * */
class FpValue {
    @JvmField
    var sign: String = ""
    @JvmField
    var exponent: String = ""
    @JvmField
    var significand: String = ""
}
