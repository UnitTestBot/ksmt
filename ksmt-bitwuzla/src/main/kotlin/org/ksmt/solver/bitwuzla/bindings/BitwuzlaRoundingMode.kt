package org.ksmt.solver.bitwuzla.bindings

/**
 * Rounding mode for floating-point operations.<br></br>
 * * For some floating-point operations, infinitely precise results may not be<br></br>
 * representable in a given format. Hence, they are rounded modulo one of five<br></br>
 * rounding modes to a representable floating-point number.<br></br>
 * * \verbatim embed:rst:leading-asterisk<br></br>
 * The following rounding modes follow the SMT-LIB theory for floating-point<br></br>
 * arithmetic, which in turn is based on IEEE Standard 754 :cite:`IEEE754`.<br></br>
 * The rounding modes are specified in Sections 4.3.1 and 4.3.2 of the IEEE<br></br>
 * Standard 754.<br></br>
 * \endverbatim<br></br>
 * enum values<br></br>
 * *native declaration : bitwuzla.h:2033*
 */
enum class BitwuzlaRoundingMode(val value: Int) {
    /**
     * Round to the nearest even number.<br></br>
     * If the two nearest floating-point numbers bracketing an unrepresentable<br></br>
     * infinitely precise result are equally near, the one with an even least<br></br>
     * significant digit will be delivered.<br></br>
     * * SMT-LIB: \c RNE \c roundNearestTiesToEven
     */
    BITWUZLA_RM_RNE(0),

    /**
     * Round to the nearest number away from zero.<br></br>
     * If the two nearest floating-point numbers bracketing an unrepresentable<br></br>
     * infinitely precise result are equally near, the one with larger magnitude<br></br>
     * will be selected.<br></br>
     * * SMT-LIB: \c RNA \c roundNearestTiesToAway
     */
    BITWUZLA_RM_RNA(1),

    /**
     * Round towards negative infinity (-oo).<br></br>
     * The result shall be the format\u2019s floating-point number (possibly -oo)<br></br>
     * closest to and no less than the infinitely precise result.<br></br>
     * * SMT-LIB: \c RTN \c roundTowardNegative
     */
    BITWUZLA_RM_RTN(2),

    /**
     * Round towards positive infinity (+oo).<br></br>
     * The result shall be the format\u2019s floating-point number (possibly +oo)<br></br>
     * closest to and no less than the infinitely precise result.<br></br>
     * * SMT-LIB: \c RTP \c roundTowardPositive
     */
    BITWUZLA_RM_RTP(3),

    /**
     * Round towards zero.<br></br>
     * The result shall be the format\u2019s floating-point number closest to and no<br></br>
     * greater in magnitude than the infinitely precise result.<br></br>
     * * SMT-LIB: \c RTZ \c roundTowardZero
     */
    BITWUZLA_RM_RTZ(4), BITWUZLA_RM_MAX(5);

    companion object {
        private val valueMapping = BitwuzlaRoundingMode.values().associateBy { it.value }
        fun fromValue(value: Int): BitwuzlaRoundingMode = valueMapping.getValue(value)
    }
}