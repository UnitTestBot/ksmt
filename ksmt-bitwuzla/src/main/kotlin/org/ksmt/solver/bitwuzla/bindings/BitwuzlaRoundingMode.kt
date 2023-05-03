@file:Suppress("MagicNumber")

package org.ksmt.solver.bitwuzla.bindings

/**
 * Rounding mode for floating-point operations.
 *
 * For some floating-point operations, infinitely precise results may not be
 * representable in a given format. Hence, they are rounded modulo one of five
 * rounding modes to a representable floating-point number.
 *
 * The following rounding modes follow the SMT-LIB theory for floating-point
 * arithmetic, which in turn is based on IEEE Standard 754.
 * The rounding modes are specified in Sections 4.3.1 and 4.3.2 of the IEEE Standard 754.
 */
enum class BitwuzlaRoundingMode(val value: Int) {
    /**
     * Round to the nearest even number.
     *
     * If the two nearest floating-point numbers bracketing an unrepresentable
     * infinitely precise result are equally near, the one with an even least
     * significant digit will be delivered.
     *
     * SMT-LIB: `RNE` `roundNearestTiesToEven`
     */
    BITWUZLA_RM_RNE(0),

    /**
     * Round to the nearest number away from zero.
     *
     * If the two nearest floating-point numbers bracketing an unrepresentable
     * infinitely precise result are equally near, the one with larger magnitude
     * will be selected.
     *
     * SMT-LIB: `RNA` `roundNearestTiesToAway`
     */
    BITWUZLA_RM_RNA(1),

    /**
     * Round towards negative infinity (-oo).
     *
     * The result shall be the format's floating-point number (possibly -oo)
     * closest to and no less than the infinitely precise result.
     *
     * SMT-LIB: `RTN` `roundTowardNegative`
     */
    BITWUZLA_RM_RTN(2),

    /**
     * Round towards positive infinity (+oo).
     *
     * The result shall be the format's floating-point number (possibly +oo)
     * closest to and no less than the infinitely precise result.
     *
     * SMT-LIB: `RTP` `roundTowardPositive`
     */
    BITWUZLA_RM_RTP(3),

    /**
     * Round towards zero.
     *
     * The result shall be the format's floating-point number closest to and no
     * greater in magnitude than the infinitely precise result.
     *
     * SMT-LIB: `RTZ` `roundTowardZero`
     */
    BITWUZLA_RM_RTZ(4), BITWUZLA_RM_MAX(5);

    companion object {
        private val valueMapping = BitwuzlaRoundingMode.values().associateBy { it.value }
        fun fromValue(value: Int): BitwuzlaRoundingMode = valueMapping.getValue(value)
    }
}
