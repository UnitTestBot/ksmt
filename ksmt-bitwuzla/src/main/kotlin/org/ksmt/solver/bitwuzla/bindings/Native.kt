@file:Suppress(
    "FunctionName",
    "FunctionParameterNaming",
    "LongParameterList",
    "LargeClass",
    "unused",
    "UNUSED_PARAMETER"
)

package org.ksmt.solver.bitwuzla.bindings

import com.sun.jna.Memory
import com.sun.jna.Native
import com.sun.jna.Pointer
import com.sun.jna.ptr.IntByReference
import com.sun.jna.ptr.PointerByReference

typealias Bitwuzla = Pointer
typealias BitwuzlaTerm = Pointer
typealias BitwuzlaSort = Pointer

object Native {
    init {
        Native.register("bitwuzla")
    }

    /**
     * Create a new Bitwuzla instance.
     *
     * The returned instance must be deleted via [bitwuzla_delete].
     *
     * @return A pointer to the created Bitwuzla instance.
     *
     * @see bitwuzla_delete
     */
    fun bitwuzlaNew(): Bitwuzla =
        bitwuzla_new()

    private external fun bitwuzla_new(): Bitwuzla


    /**
     * Delete a Bitwuzla instance.
     *
     * The given instance must have been created via [bitwuzla_new].
     *
     * @param bitwuzla The Bitwuzla instance to delete.
     *
     * @see bitwuzla_new
     */
    fun bitwuzlaDelete(bitwuzla: Bitwuzla) =
        bitwuzla_delete(bitwuzla)

    private external fun bitwuzla_delete(bitwuzla: Bitwuzla)


    /**
     * Reset a Bitwuzla instance.
     *
     * This deletes the given instance and creates a new instance in place.
     * The given instance must have been created via [bitwuzla_new].
     *
     * Note:  All sorts and terms associated with the given instance are released
     * and thus invalidated.
     *
     * @param bitwuzla The Bitwuzla instance to reset.
     *
     * @see bitwuzla_new
     */
    fun bitwuzlaReset(bitwuzla: Bitwuzla) =
        bitwuzla_reset(bitwuzla)

    private external fun bitwuzla_reset(bitwuzla: Bitwuzla)


    /**
     * Get copyright information.
     *
     * @param bitwuzla The Bitwuzla instance.
     */
    fun bitwuzlaCopyright(bitwuzla: Bitwuzla): String =
        bitwuzla_copyright(bitwuzla)

    private external fun bitwuzla_copyright(bitwuzla: Bitwuzla): String


    /**
     * Get version information.
     *
     * @param bitwuzla The Bitwuzla instance.
     */
    fun bitwuzlaVersion(bitwuzla: Bitwuzla): String =
        bitwuzla_version(bitwuzla)

    private external fun bitwuzla_version(bitwuzla: Bitwuzla): String


    /**
     * Get git information.
     *
     * @param bitwuzla The Bitwuzla instance.
     */
    fun bitwuzlaGitId(bitwuzla: Bitwuzla): String =
        bitwuzla_git_id(bitwuzla)

    private external fun bitwuzla_git_id(bitwuzla: Bitwuzla): String


    /**
     * If termination callback function has been configured via
     * [bitwuzlaSetTerminationCallback], call this termination function.
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return True if `bitwuzla` has been terminated.
     *
     * @see bitwuzlaSetTerminationCallback
     * @see bitwuzlaGetTerminationCallbackState
     */
    fun bitwuzlaTerminate(bitwuzla: Bitwuzla): Boolean =
        bitwuzla_terminate(bitwuzla)

    private external fun bitwuzla_terminate(bitwuzla: Bitwuzla): Boolean


    /**
     * Configure a termination callback function.
     *
     * The `state` of the callback can be retrieved via
     * [bitwuzlaGetTerminationCallbackState].
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param `fun` The callback function, returns a value != 0 if `bitwuzla` has
     * been terminated.
     * @param state The argument to the callback function.
     *
     * @see bitwuzla_terminate
     * @see bitwuzlaGetTerminationCallbackState
     */
    fun bitwuzlaSetTerminationCallback(bitwuzla: Bitwuzla, function: Pointer, state: Pointer) {
        throw UnsupportedOperationException("callbacks are not supported")
    }


    /**
     * Get the state of the termination callback function.
     *
     * The returned object representing the state of the callback corresponds to
     * the `state` configured as argument to the callback function via
     * [bitwuzlaSetTerminationCallback].
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return The object passed as argument `state` to the callback function.
     *
     * @see bitwuzla_terminate
     * @see bitwuzlaSetTerminationCallback
     */
    fun bitwuzlaGetTerminationCallbackState(bitwuzla: Bitwuzla): Pointer {
        throw UnsupportedOperationException("callbacks are not supported")
    }


    /**
     * Configure an abort callback function, which is called instead of exit
     * on abort conditions.
     *
     * Note:  This function is not thread safe (the function pointer is maintained
     * as a global variable). It you use threading, make sure to set the
     * abort callback prior to creating threads.
     *
     * @param `fun` The callback function, the argument `msg` explains the reason
     * for the abort.
     */
    fun bitwuzlaSetAbortCallback(function: Pointer) {
        throw UnsupportedOperationException("callbacks are not supported")
    }


    /**
     * Set option.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param option The option.
     * @param value The option value.
     *
     * @see BitwuzlaOption
     */
    fun bitwuzlaSetOption(bitwuzla: Bitwuzla, option: BitwuzlaOption, value: Int) {
        bitwuzla_set_option(bitwuzla, option.value, value)
    }

    private external fun bitwuzla_set_option(bitwuzla: Bitwuzla, option: Int, value: Int)


    /**
     * Set option value for string options.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param option The option.
     * @param value The option string value.
     *
     * @see BitwuzlaOption
     */
    fun bitwuzlaSetOptionStr(bitwuzla: Bitwuzla, option: BitwuzlaOption, value: String) {
        bitwuzla_set_option_str(bitwuzla, option.value, value)
    }

    private external fun bitwuzla_set_option_str(bitwuzla: Bitwuzla, option: Int, value: String)


    /**
     * Get the current value of an option.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param option The option.
     *
     * @return The option value.
     *
     * @see BitwuzlaOption
     */
    fun bitwuzlaGetOption(bitwuzla: Bitwuzla, option: BitwuzlaOption): Int =
        bitwuzla_get_option(bitwuzla, option.value)

    private external fun bitwuzla_get_option(bitwuzla: Bitwuzla, option: Int): Int


    /**
     * Get the current value of an option as a string if option can be configured
     * via a string value.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param option The option.
     *
     * @return The option value.
     *
     * @see BitwuzlaOption
     * @see bitwuzla_set_option_str
     */
    fun bitwuzlaGetOptionStr(bitwuzla: Bitwuzla, option: BitwuzlaOption): String =
        bitwuzla_get_option_str(bitwuzla, option.value)

    private external fun bitwuzla_get_option_str(bitwuzla: Bitwuzla, option: Int): String


    /**
     * Get the details of an option.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param option The option.
     *
     * @return The option info. Will be valid until the next
     * [bitwuzlaGetOptionInfo] call.
     *
     * @see BitwuzlaOptionInfo
     */
    fun bitwuzlaGetOptionInfo(bitwuzla: Bitwuzla, option: BitwuzlaOption): Pointer {
        TODO("BitwuzlaOptionInfo structure is not implemented")
    }


    /**
     * Create an array sort.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param index The index sort of the array sort.
     * @param element The element sort of the array sort.
     *
     * @return An array sort which maps sort `index` to sort `element`.
     *
     * @see bitwuzla_sort_is_array
     * @see bitwuzla_sort_array_get_index
     * @see bitwuzla_sort_array_get_element
     * @see bitwuzla_term_is_array
     * @see bitwuzla_term_array_get_index_sort
     * @see bitwuzla_term_array_get_element_sort
     */
    fun bitwuzlaMkArraySort(
        bitwuzla: Bitwuzla,
        index: BitwuzlaSort,
        element: BitwuzlaSort
    ): BitwuzlaSort = bitwuzla_mk_array_sort(bitwuzla, index, element)

    private external fun bitwuzla_mk_array_sort(
        bitwuzla: Bitwuzla,
        index: BitwuzlaSort,
        element: BitwuzlaSort
    ): BitwuzlaSort


    /**
     * Create a Boolean sort.
     *
     * Note:  A Boolean sort is a bit-vector sort of size 1.
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return A Boolean sort.
     */
    fun bitwuzlaMkBoolSort(bitwuzla: Bitwuzla): BitwuzlaSort =
        bitwuzla_mk_bool_sort(bitwuzla)

    private external fun bitwuzla_mk_bool_sort(bitwuzla: Bitwuzla): BitwuzlaSort


    /**
     * Create a bit-vector sort of given size.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param size The size of the bit-vector sort.
     *
     * @return A bit-vector sort of given size.
     *
     * @see bitwuzla_sort_is_bv
     * @see bitwuzla_sort_bv_get_size
     * @see bitwuzla_term_is_bv
     * @see bitwuzla_term_bv_get_size
     */
    fun bitwuzlaMkBvSort(bitwuzla: Bitwuzla, size: Int): BitwuzlaSort =
        bitwuzla_mk_bv_sort(bitwuzla, size)

    private external fun bitwuzla_mk_bv_sort(bitwuzla: Bitwuzla, size: Int): BitwuzlaSort


    /**
     * Create a floating-point sort of given exponent and significand size.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param expSize The size of the exponent.
     * @param sigSize The size of the significand (including sign bit).
     *
     * @return A floating-point sort of given format.
     *
     * @see bitwuzla_sort_is_fp
     * @see bitwuzla_sort_fp_get_exp_size
     * @see bitwuzla_sort_fp_get_sig_size
     * @see bitwuzla_term_is_fp
     * @see bitwuzla_term_fp_get_exp_size
     * @see bitwuzla_term_fp_get_sig_size
     */
    fun bitwuzlaMkFpSort(bitwuzla: Bitwuzla, expSize: Int, sigSize: Int): BitwuzlaSort =
        bitwuzla_mk_fp_sort(bitwuzla, expSize, sigSize)

    private external fun bitwuzla_mk_fp_sort(bitwuzla: Bitwuzla, exp_size: Int, sig_size: Int): BitwuzlaSort


    /**
     * Create a function sort.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param arity The number of arguments to the function.
     * @param domain The domain sorts (the sorts of the arguments). The number of
     * sorts in this vector must match `arity`.
     * @param codomain The codomain sort (the sort of the return value).
     *
     * @return A function sort of given domain and codomain sorts.
     *
     * @see bitwuzla_sort_is_fun
     * @see bitwuzla_sort_fun_get_arity
     * @see bitwuzla_sort_fun_get_domain_sorts
     * @see bitwuzla_sort_fun_get_codomain
     * @see bitwuzla_term_is_fun
     * @see bitwuzla_term_fun_get_arity
     * @see bitwuzla_term_fun_get_domain_sorts
     * @see bitwuzla_term_fun_get_codomain_sort
     */
    fun bitwuzlaMkFunSort(
        bitwuzla: Bitwuzla,
        arity: Int,
        domain: Array<BitwuzlaSort>,
        codomain: BitwuzlaSort
    ): BitwuzlaSort = bitwuzla_mk_fun_sort(bitwuzla, arity, domain.mkPtr(), codomain)

    private external fun bitwuzla_mk_fun_sort(
        bitwuzla: Bitwuzla,
        arity: Int,
        domain: Pointer,
        codomain: BitwuzlaSort
    ): BitwuzlaSort

    /**
     * Create a Roundingmode sort.
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return A Roundingmode sort.
     *
     * @see bitwuzla_sort_is_rm
     * @see bitwuzla_term_is_rm
     */
    fun bitwuzlaMkRmSort(bitwuzla: Bitwuzla): BitwuzlaSort =
        bitwuzla_mk_rm_sort(bitwuzla)

    private external fun bitwuzla_mk_rm_sort(bitwuzla: Bitwuzla): BitwuzlaSort


    /**
     * Create a true value.
     *
     * Note:  This creates a bit-vector value 1 of size 1.
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return A term representing the bit-vector value 1 of size 1.
     */
    fun bitwuzlaMkTrue(bitwuzla: Bitwuzla): BitwuzlaTerm =
        bitwuzla_mk_true(bitwuzla)

    private external fun bitwuzla_mk_true(bitwuzla: Bitwuzla): BitwuzlaTerm


    /**
     * Create a false value.
     *
     * Note:  This creates a bit-vector value 0 of size 1.
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return A term representing the bit-vector value 0 of size 1.
     */
    fun bitwuzlaMkFalse(bitwuzla: Bitwuzla): BitwuzlaTerm =
        bitwuzla_mk_false(bitwuzla)

    private external fun bitwuzla_mk_false(bitwuzla: Bitwuzla): BitwuzlaTerm


    /**
     * Create a bit-vector value zero.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the bit-vector value 0 of given sort.
     *
     * @see bitwuzla_mk_bv_sort
     */
    fun bitwuzlaMkBvZero(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm =
        bitwuzla_mk_bv_zero(bitwuzla, sort)

    private external fun bitwuzla_mk_bv_zero(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm


    /**
     * Create a bit-vector value one.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the bit-vector value 1 of given sort.
     *
     * @see bitwuzla_mk_bv_sort
     */
    fun bitwuzlaMkBvOne(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm =
        bitwuzla_mk_bv_one(bitwuzla, sort)

    private external fun bitwuzla_mk_bv_one(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm


    /**
     * Create a bit-vector value where all bits are set to 1.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the bit-vector value of given sort
     * where all bits are set to 1.
     *
     * @see bitwuzla_mk_bv_sort
     */
    fun bitwuzlaMkBvOnes(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm =
        bitwuzla_mk_bv_ones(bitwuzla, sort)

    private external fun bitwuzla_mk_bv_ones(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm


    /**
     * Create a bit-vector minimum signed value.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the bit-vector value of given sort where the MSB
     * is set to 1 and all remaining bits are set to 0.
     *
     * @see bitwuzla_mk_bv_sort
     */
    fun bitwuzlaMkBvMinSigned(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm =
        bitwuzla_mk_bv_min_signed(bitwuzla, sort)

    private external fun bitwuzla_mk_bv_min_signed(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm


    /**
     * Create a bit-vector maximum signed value.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the bit-vector value of given sort where the MSB
     * is set to 0 and all remaining bits are set to 1.
     *
     * @see bitwuzla_mk_bv_sort
     */
    fun bitwuzlaMkBvMaxSigned(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm =
        bitwuzla_mk_bv_max_signed(bitwuzla, sort)

    private external fun bitwuzla_mk_bv_max_signed(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm


    /**
     * Create a floating-point positive zero value (SMT-LIB: `+zero`).
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the floating-point positive zero value of given
     * floating-point sort.
     *
     * @see bitwuzla_mk_fp_sort
     */
    fun bitwuzlaMkFpPosZero(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm =
        bitwuzla_mk_fp_pos_zero(bitwuzla, sort)

    private external fun bitwuzla_mk_fp_pos_zero(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm


    /**
     * Create a floating-point negative zero value (SMT-LIB: `-zero`).
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the floating-point negative zero value of given
     * floating-point sort.
     *
     * @see bitwuzla_mk_fp_sort
     */
    fun bitwuzlaMkFpNegZero(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm =
        bitwuzla_mk_fp_neg_zero(bitwuzla, sort)

    private external fun bitwuzla_mk_fp_neg_zero(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm


    /**
     * Create a floating-point positive infinity value (SMT-LIB: `+oo`).
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the floating-point positive infinity value of
     * given floating-point sort.
     *
     * @see bitwuzla_mk_fp_sort
     */
    fun bitwuzlaMkFpPosInf(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm =
        bitwuzla_mk_fp_pos_inf(bitwuzla, sort)

    private external fun bitwuzla_mk_fp_pos_inf(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm


    /**
     * Create a floating-point negative infinity value (SMT-LIB: `-oo`).
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the floating-point negative infinity value of
     * given floating-point sort.
     *
     * @see bitwuzla_mk_fp_sort
     */
    fun bitwuzlaMkFpNegInf(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm =
        bitwuzla_mk_fp_neg_inf(bitwuzla, sort)

    private external fun bitwuzla_mk_fp_neg_inf(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm


    /**
     * Create a floating-point NaN value.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the floating-point NaN value of given
     * floating-point sort.
     *
     * @see bitwuzla_mk_fp_sort
     */
    fun bitwuzlaMkFpNan(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm =
        bitwuzla_mk_fp_nan(bitwuzla, sort)

    private external fun bitwuzla_mk_fp_nan(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm


    /**
     * Create a bit-vector value from its string representation.
     *
     * Parameter `base` determines the base of the string representation.
     *
     * Note:  Given value must fit into a bit-vector of given size (sort).
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param sort The sort of the value.
     * @param value A string representing the value.
     * @param base The base in which the string is given.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VAL], representing the bit-vector value
     * of given sort.
     *
     * @see bitwuzla_mk_bv_sort
     * @see BitwuzlaBVBase
     */
    fun bitwuzlaMkBvValue(
        bitwuzla: Bitwuzla,
        sort: BitwuzlaSort,
        value: String,
        base: BitwuzlaBVBase
    ): BitwuzlaTerm = bitwuzla_mk_bv_value(bitwuzla, sort, value, base.value)

    private external fun bitwuzla_mk_bv_value(
        bitwuzla: Bitwuzla,
        sort: BitwuzlaSort,
        value: String,
        base: Int
    ): BitwuzlaTerm


    /**
     * Create a bit-vector value from its unsigned integer representation.
     *
     * Note:  If given value does not fit into a bit-vector of given size (sort),
     * the value is truncated to fit.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param sort The sort of the value.
     * @param value The unsigned integer representation of the bit-vector value.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VAL], representing the bit-vector value
     * of given sort.
     *
     * @see bitwuzla_mk_bv_sort
     */
    fun bitwuzlaMkBvValueUint64(bitwuzla: Bitwuzla, sort: BitwuzlaSort, value: Long): BitwuzlaTerm =
        bitwuzla_mk_bv_value_uint64(bitwuzla, sort, value)

    private external fun bitwuzla_mk_bv_value_uint64(bitwuzla: Bitwuzla, sort: BitwuzlaSort, value: Long): BitwuzlaTerm


    /**
     * Create a floating-point value from its IEEE 754 standard representation
     * given as three bit-vector values representing the sign bit, the exponent and
     * the significand.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param bvSign The sign bit.
     * @param bvExponent The exponent bit-vector value.
     * @param bvSignificand The significand bit-vector value.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VAL], representing the floating-point
     * value.
     */
    fun bitwuzlaMkFpValue(
        bitwuzla: Bitwuzla,
        bvSign: BitwuzlaTerm,
        bvExponent: BitwuzlaTerm,
        bvSignificand: BitwuzlaTerm
    ): BitwuzlaTerm = bitwuzla_mk_fp_value(bitwuzla, bvSign, bvExponent, bvSignificand)

    private external fun bitwuzla_mk_fp_value(
        bitwuzla: Bitwuzla,
        bv_sign: BitwuzlaTerm,
        bv_exponent: BitwuzlaTerm,
        bv_significand: BitwuzlaTerm
    ): BitwuzlaTerm


    /**
     * Create a floating-point value from its real representation, given as a
     * decimal string, with respect to given rounding mode.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param sort The sort of the value.
     * @param rm The rounding mode.
     * @param real The decimal string representing a real value.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VAL], representing the floating-point
     * value of given sort.
     *
     * @see bitwuzla_mk_fp_sort
     */
    fun bitwuzlaMkFpValueFromReal(
        bitwuzla: Bitwuzla,
        sort: BitwuzlaSort,
        rm: BitwuzlaTerm,
        real: String
    ): BitwuzlaTerm = bitwuzla_mk_fp_value_from_real(bitwuzla, sort, rm, real)

    private external fun bitwuzla_mk_fp_value_from_real(
        bitwuzla: Bitwuzla,
        sort: BitwuzlaSort,
        rm: BitwuzlaTerm,
        real: String
    ): BitwuzlaTerm


    /**
     * Create a floating-point value from its rational representation, given as a
     * two decimal strings representing the numerator and denominator, with respect
     * to given rounding mode.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param sort The sort of the value.
     * @param rm The rounding mode.
     * @param num The decimal string representing the numerator.
     * @param den The decimal string representing the denominator.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VAL], representing the floating-point
     * value of given sort.
     *
     * @see bitwuzla_mk_fp_sort
     */
    fun bitwuzlaMkFpValueFromRational(
        bitwuzla: Bitwuzla,
        sort: BitwuzlaSort,
        rm: BitwuzlaTerm,
        num: String,
        den: String
    ): BitwuzlaTerm = bitwuzla_mk_fp_value_from_rational(bitwuzla, sort, rm, num, den)

    private external fun bitwuzla_mk_fp_value_from_rational(
        bitwuzla: Bitwuzla,
        sort: BitwuzlaSort,
        rm: BitwuzlaTerm,
        num: String,
        den: String
    ): BitwuzlaTerm


    /**
     * Create a rounding mode value.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param rm The rounding mode value.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VAL], representing the rounding mode
     * value.
     *
     * @see BitwuzlaRoundingMode
     */
    fun bitwuzlaMkRmValue(bitwuzla: Bitwuzla, rm: BitwuzlaRoundingMode): BitwuzlaTerm =
        bitwuzla_mk_rm_value(bitwuzla, rm.value)

    private external fun bitwuzla_mk_rm_value(bitwuzla: Bitwuzla, rm: Int): BitwuzlaTerm


    /**
     * Create a term of given kind with one argument term.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param kind The operator kind.
     * @param arg The argument to the operator.
     *
     * @return A term representing an operation of given kind.
     *
     * @see  BitwuzlaKind
     */
    fun bitwuzlaMkTerm1(bitwuzla: Bitwuzla, kind: BitwuzlaKind, arg: BitwuzlaTerm): BitwuzlaTerm =
        bitwuzla_mk_term1(bitwuzla, kind.value, arg)

    private external fun bitwuzla_mk_term1(bitwuzla: Bitwuzla, kind: Int, arg: BitwuzlaTerm): BitwuzlaTerm


    /**
     * Create a term of given kind with two argument terms.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param kind The operator kind.
     * @param arg0 The first argument to the operator.
     * @param arg1 The second argument to the operator.
     *
     * @return A term representing an operation of given kind.
     *
     * @see  BitwuzlaKind
     */
    fun bitwuzlaMkTerm2(
        bitwuzla: Bitwuzla,
        kind: BitwuzlaKind,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm
    ): BitwuzlaTerm = bitwuzla_mk_term2(bitwuzla, kind.value, arg0, arg1)

    private external fun bitwuzla_mk_term2(
        bitwuzla: Bitwuzla,
        kind: Int,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm
    ): BitwuzlaTerm


    /**
     * Create a term of given kind with three argument terms.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param kind The operator kind.
     * @param arg0 The first argument to the operator.
     * @param arg1 The second argument to the operator.
     * @param arg2 The third argument to the operator.
     *
     * @return A term representing an operation of given kind.
     *
     * @see  BitwuzlaKind
     */
    fun bitwuzlaMkTerm3(
        bitwuzla: Bitwuzla,
        kind: BitwuzlaKind,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        arg2: BitwuzlaTerm
    ): BitwuzlaTerm = bitwuzla_mk_term3(bitwuzla, kind.value, arg0, arg1, arg2)

    private external fun bitwuzla_mk_term3(
        bitwuzla: Bitwuzla,
        kind: Int,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        arg2: BitwuzlaTerm
    ): BitwuzlaTerm


    /**
     * Create a term of given kind with the given argument terms.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param kind The operator kind.
     * @param args The argument terms.
     *
     * @return A term representing an operation of given kind.
     *
     * @see  BitwuzlaKind
     */
    fun bitwuzlaMkTerm(bitwuzla: Bitwuzla, kind: BitwuzlaKind, args: Array<BitwuzlaTerm>): BitwuzlaTerm =
        bitwuzla_mk_term(bitwuzla, kind.value, args.size, args.mkPtr())

    private external fun bitwuzla_mk_term(bitwuzla: Bitwuzla, kind: Int, argc: Int, args: Pointer): BitwuzlaTerm

    /**
     * Create an indexed term of given kind with one argument term and one index.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param kind The operator kind.
     * @param arg The argument term.
     * @param idx The index.
     *
     * @return A term representing an indexed operation of given kind.
     *
     * @see  BitwuzlaKind
     */
    fun bitwuzlaMkTerm1Indexed1(
        bitwuzla: Bitwuzla,
        kind: BitwuzlaKind,
        arg: BitwuzlaTerm,
        idx: Int
    ): BitwuzlaTerm = bitwuzla_mk_term1_indexed1(bitwuzla, kind.value, arg, idx)

    private external fun bitwuzla_mk_term1_indexed1(
        bitwuzla: Bitwuzla,
        kind: Int,
        arg: BitwuzlaTerm,
        idx: Int
    ): BitwuzlaTerm

    /**
     * Create an indexed term of given kind with one argument term and two indices.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param kind The operator kind.
     * @param arg The argument term.
     * @param idx0 The first index.
     * @param idx1 The second index.
     *
     * @return A term representing an indexed operation of given kind.
     *
     * @see  BitwuzlaKind
     */
    fun bitwuzlaMkTerm1Indexed2(
        bitwuzla: Bitwuzla,
        kind: BitwuzlaKind,
        arg: BitwuzlaTerm,
        idx0: Int,
        idx1: Int
    ): BitwuzlaTerm = bitwuzla_mk_term1_indexed2(bitwuzla, kind.value, arg, idx0, idx1)

    private external fun bitwuzla_mk_term1_indexed2(
        bitwuzla: Bitwuzla,
        kind: Int,
        arg: BitwuzlaTerm,
        idx0: Int,
        idx1: Int
    ): BitwuzlaTerm


    /**
     * Create an indexed term of given kind with two argument terms and one index.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param kind The operator kind.
     * @param arg0 The first argument term.
     * @param arg1 The second argument term.
     * @param idx The index.
     *
     * @return A term representing an indexed operation of given kind.
     *
     * @see  BitwuzlaKind
     */
    fun bitwuzlaMkTerm2Indexed1(
        bitwuzla: Bitwuzla,
        kind: BitwuzlaKind,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        idx: Int
    ): BitwuzlaTerm = bitwuzla_mk_term2_indexed1(bitwuzla, kind.value, arg0, arg1, idx)

    private external fun bitwuzla_mk_term2_indexed1(
        bitwuzla: Bitwuzla,
        kind: Int,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        idx: Int
    ): BitwuzlaTerm


    /**
     * Create an indexed term of given kind with two argument terms and two indices.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param kind The operator kind.
     * @param arg0 The first argument term.
     * @param arg1 The second argument term.
     * @param idx0 The first index.
     * @param idx1 The second index.
     *
     * @return A term representing an indexed operation of given kind.
     *
     * @see  BitwuzlaKind
     */
    fun bitwuzlaMkTerm2Indexed2(
        bitwuzla: Bitwuzla,
        kind: BitwuzlaKind,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        idx0: Int,
        idx1: Int
    ): BitwuzlaTerm = bitwuzla_mk_term2_indexed2(bitwuzla, kind.value, arg0, arg1, idx0, idx1)

    private external fun bitwuzla_mk_term2_indexed2(
        bitwuzla: Bitwuzla,
        kind: Int,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        idx0: Int,
        idx1: Int
    ): BitwuzlaTerm


    /**
     * Create an indexed term of given kind with the given argument terms and
     * indices.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param kind The operator kind.
     * @param args The argument terms.
     * @param idxs The indices.
     *
     * @return A term representing an indexed operation of given kind.
     *
     * @see  BitwuzlaKind
     */
    fun bitwuzlaMkTermIndexed(
        bitwuzla: Bitwuzla,
        kind: BitwuzlaKind,
        args: Array<BitwuzlaTerm>,
        idxs: IntArray
    ): BitwuzlaTerm = bitwuzla_mk_term_indexed(bitwuzla, kind.value, args.size, args.mkPtr(), idxs.size, idxs)

    private external fun bitwuzla_mk_term_indexed(
        bitwuzla: Bitwuzla,
        kind: Int,
        argc: Int,
        args: Pointer,
        idxc: Int,
        idxs: IntArray
    ): BitwuzlaTerm


    /**
     * Create a (first-order) constant of given sort with given symbol.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param sort The sort of the constant.
     * @param symbol The symbol of the constant.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_CONST], representing the constant.
     *
     * @see bitwuzla_mk_array_sort
     * @see bitwuzla_mk_bool_sort
     * @see bitwuzla_mk_bv_sort
     * @see bitwuzla_mk_fp_sort
     * @see bitwuzlaMkFunSort
     * @see bitwuzla_mk_rm_sort
     */
    fun bitwuzlaMkConst(bitwuzla: Bitwuzla, sort: BitwuzlaSort, symbol: String): BitwuzlaTerm =
        bitwuzla_mk_const(bitwuzla, sort, symbol)

    private external fun bitwuzla_mk_const(bitwuzla: Bitwuzla, sort: BitwuzlaSort, symbol: String): BitwuzlaTerm


    /**
     * Create a one-dimensional constant array of given sort, initialized with
     * given value.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param sort The sort of the array.
     * @param value The term to initialize the elements of the array with.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_CONST_ARRAY], representing a constant
     * array of given sort.
     *
     * @see bitwuzla_mk_array_sort
     */
    fun bitwuzlaMkConstArray(bitwuzla: Bitwuzla, sort: BitwuzlaSort, value: BitwuzlaTerm): BitwuzlaTerm =
        bitwuzla_mk_const_array(bitwuzla, sort, value)

    private external fun bitwuzla_mk_const_array(
        bitwuzla: Bitwuzla,
        sort: BitwuzlaSort,
        value: BitwuzlaTerm
    ): BitwuzlaTerm


    /**
     * Create a variable of given sort with given symbol.
     *
     * Note:  This creates a variable to be bound by quantifiers or lambdas.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param sort The sort of the variable.
     * @param symbol The symbol of the variable.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VAR], representing the variable.
     *
     * @see bitwuzla_mk_bool_sort
     * @see bitwuzla_mk_bv_sort
     * @see bitwuzla_mk_fp_sort
     * @see bitwuzlaMkFunSort
     * @see bitwuzla_mk_rm_sort
     */
    fun bitwuzlaMkVar(bitwuzla: Bitwuzla, sort: BitwuzlaSort, symbol: String): BitwuzlaTerm =
        bitwuzla_mk_var(bitwuzla, sort, symbol)

    private external fun bitwuzla_mk_var(bitwuzla: Bitwuzla, sort: BitwuzlaSort, symbol: String): BitwuzlaTerm


    /**
     * Push context levels.
     *
     * Requires that incremental solving has been enabled via
     * [bitwuzla_set_option].
     *
     * Note:  Assumptions added via this [bitwuzla_assume] are not affected by
     * context level changes and are only valid until the next
     * [bitwuzla_check_sat] call, no matter at which level they were
     * assumed.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param nlevels The number of context levels to push.
     *
     * @see bitwuzla_set_option
     * @see BitwuzlaOption.BITWUZLA_OPT_INCREMENTAL
     */
    fun bitwuzlaPush(bitwuzla: Bitwuzla, nlevels: Int) =
        bitwuzla_push(bitwuzla, nlevels)

    private external fun bitwuzla_push(bitwuzla: Bitwuzla, nlevels: Int)


    /**
     * Pop context levels.
     *
     * Requires that incremental solving has been enabled via
     * [bitwuzla_set_option].
     *
     * Note:  Assumptions added via this [bitwuzla_assume] are not affected by
     * context level changes and are only valid until the next
     * [bitwuzla_check_sat] call, no matter at which level they were
     * assumed.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param nlevels The number of context levels to pop.
     *
     * @see bitwuzla_set_option
     * @see BitwuzlaOption.BITWUZLA_OPT_INCREMENTAL
     */
    fun bitwuzlaPop(bitwuzla: Bitwuzla, nlevels: Int) =
        bitwuzla_pop(bitwuzla, nlevels)

    private external fun bitwuzla_pop(bitwuzla: Bitwuzla, nlevels: Int)


    /**
     * Assert formula.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param term The formula to assert.
     */
    fun bitwuzlaAssert(bitwuzla: Bitwuzla, term: BitwuzlaTerm) =
        bitwuzla_assert(bitwuzla, term)

    private external fun bitwuzla_assert(bitwuzla: Bitwuzla, term: BitwuzlaTerm)


    /**
     * Assume formula.
     *
     * Requires that incremental solving has been enabled via
     * [bitwuzla_set_option].
     *
     * Note:  Assumptions added via this function are not affected by context level
     * changes and are only valid until the next [bitwuzla_check_sat] call,
     * no matter at which level they were assumed.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param term The formula to assume.
     *
     * @see bitwuzla_set_option
     * @see bitwuzla_is_unsat_assumption
     * @see bitwuzla_get_unsat_assumptions
     * @see BitwuzlaOption.BITWUZLA_OPT_INCREMENTAL
     */
    fun bitwuzlaAssume(bitwuzla: Bitwuzla, term: BitwuzlaTerm) =
        bitwuzla_assume(bitwuzla, term)

    private external fun bitwuzla_assume(bitwuzla: Bitwuzla, term: BitwuzlaTerm)


    /**
     * Determine if an assumption is an unsat assumption.
     *
     * Unsat assumptions are assumptions that force an input formula to become
     * unsatisfiable. Unsat assumptions handling in Bitwuzla is analogous to
     * failed assumptions in MiniSAT.
     *
     * Requires that incremental solving has been enabled via
     * [bitwuzla_set_option].
     *
     * Requires that the last [bitwuzla_check_sat] query returned
     * [BitwuzlaResult.BITWUZLA_UNSAT].
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param term The assumption to check for.
     *
     * @return True if given assumption is an unsat assumption.
     *
     * @see bitwuzla_set_option
     * @see bitwuzla_assume
     * @see bitwuzla_check_sat
     * @see BitwuzlaOption.BITWUZLA_OPT_INCREMENTAL
     */
    fun bitwuzlaIsUnsatAssumption(bitwuzla: Bitwuzla, term: BitwuzlaTerm): Boolean =
        bitwuzla_is_unsat_assumption(bitwuzla, term)

    private external fun bitwuzla_is_unsat_assumption(bitwuzla: Bitwuzla, term: BitwuzlaTerm): Boolean


    /**
     * Get the set of unsat assumptions.
     *
     * Unsat assumptions are assumptions that force an input formula to become
     * unsatisfiable. Unsat assumptions handling in Bitwuzla is analogous to
     * failed assumptions in MiniSAT.
     *
     * Requires that incremental solving has been enabled via
     * [bitwuzla_set_option].
     *
     * Requires that the last [bitwuzla_check_sat] query returned
     * [BitwuzlaResult.BITWUZLA_UNSAT].
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return An array with unsat assumptions.
     *
     * @see bitwuzla_set_option
     * @see bitwuzla_assume
     * @see bitwuzla_check_sat
     * @see BitwuzlaOption.BITWUZLA_OPT_INCREMENTAL
     */
    fun bitwuzlaGetUnsatAssumptions(bitwuzla: Bitwuzla): Array<BitwuzlaTerm> {
        val size = IntByReference()
        val resultPtr = bitwuzla_get_unsat_assumptions(bitwuzla, size)
        return resultPtr.load(size.value)
    }

    private external fun bitwuzla_get_unsat_assumptions(bitwuzla: Bitwuzla, size: IntByReference): Pointer


    /**
     * Get the set unsat core (unsat assertions).
     *
     * The unsat core consists of the set of assertions that force an input formula
     * to become unsatisfiable.
     *
     * Requires that the last [bitwuzla_check_sat] query returned
     * [BitwuzlaResult.BITWUZLA_UNSAT].
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return An array with unsat assertions.
     *
     * @see bitwuzla_assert
     * @see bitwuzla_check_sat
     */
    fun bitwuzlaGetUnsatCore(bitwuzla: Bitwuzla): Array<BitwuzlaTerm> {
        val size = IntByReference()
        val resultPtr = bitwuzla_get_unsat_core(bitwuzla, size)
        return resultPtr.load(size.value)
    }

    private external fun bitwuzla_get_unsat_core(bitwuzla: Bitwuzla, size: IntByReference): Pointer


    /**
     * Assert all added assumptions.
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @see bitwuzla_assume
     */
    fun bitwuzlaFixateAssumptions(bitwuzla: Bitwuzla) =
        bitwuzla_fixate_assumptions(bitwuzla)

    private external fun bitwuzla_fixate_assumptions(bitwuzla: Bitwuzla)


    /**
     * Reset all added assumptions.
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @see bitwuzla_assume
     */
    fun bitwuzlaResetAssumptions(bitwuzla: Bitwuzla) =
        bitwuzla_reset_assumptions(bitwuzla)

    private external fun bitwuzla_reset_assumptions(bitwuzla: Bitwuzla)


    /**
     * Simplify the current input formula.
     *
     * Note:  Assumptions are not considered for simplification.
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return [BitwuzlaResult.BITWUZLA_SAT] if the input formula was simplified to true,
     * [BitwuzlaResult.BITWUZLA_UNSAT] if it was simplified to false, and
     * [BitwuzlaResult.BITWUZLA_UNKNOWN] otherwise.
     *
     * @see bitwuzla_assert
     * @see BitwuzlaResult
     */
    fun bitwuzlaSimplify(bitwuzla: Bitwuzla): BitwuzlaResult =
        BitwuzlaResult.fromValue(bitwuzla_simplify(bitwuzla))

    private external fun bitwuzla_simplify(bitwuzla: Bitwuzla): Int


    /**
     * Check satisfiability of current input formula.
     *
     * An input formula consists of assertions added via [bitwuzla_assert].
     * The search for a solution can by guided by making assumptions via
     * [bitwuzla_assume].
     *
     * Note:  Assertions and assumptions are combined via Boolean and.  Multiple
     * calls to this function require enabling incremental solving via
     * [bitwuzla_set_option].
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return [BitwuzlaResult.BITWUZLA_SAT] if the input formula is satisfiable and
     * [BitwuzlaResult.BITWUZLA_UNSAT] if it is unsatisfiable, and [BitwuzlaResult.BITWUZLA_UNKNOWN]
     * when neither satisfiability nor unsatisfiability was determined.
     * This can happen when `bitwuzla` was terminated via a termination
     * callback.
     *
     * @see bitwuzla_assert
     * @see bitwuzla_assume
     * @see bitwuzla_set_option
     * @see BitwuzlaOption.BITWUZLA_OPT_INCREMENTAL
     * @see BitwuzlaResult
     */
    fun bitwuzlaCheckSat(bitwuzla: Bitwuzla): BitwuzlaResult =
        BitwuzlaResult.fromValue(bitwuzla_check_sat(bitwuzla))

    private external fun bitwuzla_check_sat(bitwuzla: Bitwuzla): Int


    /**
     * Get a term representing the model value of a given term.
     *
     * Requires that the last [bitwuzla_check_sat] query returned
     * [BitwuzlaResult.BITWUZLA_SAT].
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param term The term to query a model value for.
     *
     * @return A term representing the model value of term `term`.
     *
     * @see bitwuzla_check_sat
     */
    fun bitwuzlaGetValue(bitwuzla: Bitwuzla, term: BitwuzlaTerm): BitwuzlaTerm =
        bitwuzla_get_value(bitwuzla, term)

    private external fun bitwuzla_get_value(bitwuzla: Bitwuzla, term: BitwuzlaTerm): BitwuzlaTerm


    /**
     * Get string representation of the current model value of given bit-vector
     * term.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param term The term to query a model value for.
     *
     * @return Binary string representation of current model value of term \p term.
     * Return value is valid until next `bitwuzla_get_bv_value` call.
     */
    fun bitwuzlaGetBvValue(bitwuzla: Bitwuzla, term: BitwuzlaTerm): String =
        bitwuzla_get_bv_value(bitwuzla, term)

    private external fun bitwuzla_get_bv_value(bitwuzla: Bitwuzla, term: BitwuzlaTerm): String


    /**
     * Get string of IEEE 754 standard representation of the current model value of
     * given floating-point term.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param term The term to query a model value for.
     *
     * @see FpValue
     */
    fun bitwuzlaGetFpValue(bitwuzla: Bitwuzla, term: BitwuzlaTerm): FpValue {
        val signPtr = PointerByReference()
        val exponentPtr = PointerByReference()
        val significandPtr = PointerByReference()
        bitwuzla_get_fp_value(bitwuzla, term, signPtr, exponentPtr, significandPtr)
        return FpValue(
            signPtr.value.getString(0),
            exponentPtr.value.getString(0),
            significandPtr.value.getString(0)
        )
    }

    private external fun bitwuzla_get_fp_value(
        bitwuzla: Bitwuzla,
        term: BitwuzlaTerm,
        sign: PointerByReference,
        exponent: PointerByReference,
        significand: PointerByReference
    )

    /**
     * @param sign Binary string representation of the sign bit.
     * @param exponent Binary string representation of the exponent bit-vector
     * value.
     * @param significand Binary string representation of the significand
     * bit-vector value.
     * */
    class FpValue(val sign: String, val exponent: String, val significand: String)

    /**
     * Get string representation of the current model value of given rounding mode
     * term.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param term The rounding mode term to query a model value for.
     *
     * @return String representation of rounding mode (RNA, RNE, RTN, RTP, RTZ).
     */
    fun bitwuzlaGetRmValue(bitwuzla: Bitwuzla, term: BitwuzlaTerm): String =
        bitwuzla_get_rm_value(bitwuzla, term)

    private external fun bitwuzla_get_rm_value(bitwuzla: Bitwuzla, term: BitwuzlaTerm): String


    /**
     * Get the current model value of given array term.
     *
     * The string representation of `indices` and `values` can be queried via
     * [bitwuzla_get_bv_value], [bitwuzla_get_fp_value], and
     * [bitwuzla_get_rm_value].
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param term The term to query a model value for.
     *
     * @see ArrayValue
     */
    fun bitwuzlaGetArrayValue(bitwuzla: Bitwuzla, term: BitwuzlaTerm): ArrayValue {
        val size = IntByReference()
        val indices = PointerByReference()
        val values = PointerByReference()
        val defaultValue = PointerByReference()
        bitwuzla_get_array_value(bitwuzla, term, indices, values, size, defaultValue)
        val sz = size.value
        return ArrayValue(
            sz,
            indices.value.load(sz),
            values.value.load(sz),
            defaultValue.value
        )
    }

    private external fun bitwuzla_get_array_value(
        bitwuzla: Bitwuzla,
        term: BitwuzlaTerm,
        indices: PointerByReference,
        values: PointerByReference,
        size: IntByReference,
        default_value: PointerByReference
    )

    /**
     * @param indices List of indices of size `size`. 1:1 mapping to `values`,
     * i.e., `index(i) -> value(i)`.
     * @param values List of values of size `size`.
     * @param size Size of `indices` and `values` list.
     * @param defaultValue The value of all other indices not in `indices` and
     * is set when base array is a constant array.
     */
    class ArrayValue(
        val size: Int,
        val indices: Array<BitwuzlaTerm>,
        val values: Array<BitwuzlaTerm>,
        val defaultValue: BitwuzlaTerm?
    )

    /**
     * Get the current model value of given function term.
     *
     * The string representation of `args` and `values` can be queried via
     * [bitwuzla_get_bv_value], [bitwuzla_get_fp_value], and
     * [bitwuzla_get_rm_value].
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param term The term to query a model value for.
     *
     * @see FunValue
     */
    fun bitwuzlaGetFunValue(bitwuzla: Bitwuzla, term: BitwuzlaTerm): FunValue {
        val arityPtr = IntByReference()
        val sizePtr = IntByReference()
        val argsPtr = PointerByReference()
        val valuesPtr = PointerByReference()
        bitwuzla_get_fun_value(bitwuzla, term, argsPtr, arityPtr, valuesPtr, sizePtr)
        val size = sizePtr.value
        val arity = arityPtr.value
        val argsPtrList = argsPtr.value.load(size)
        val args = Array(size) { argsPtrList[it].load(arity) }
        return FunValue(size, arity, args, valuesPtr.value.load(size))
    }

    private external fun bitwuzla_get_fun_value(
        bitwuzla: Bitwuzla,
        term: BitwuzlaTerm,
        args: PointerByReference,
        arity: IntByReference,
        values: PointerByReference,
        size: IntByReference
    )

    /**
     * @param args List of argument lists (nested lists) of size `size`. Each
     * argument list is of size `arity`.
     * @param arity Size of each argument list in `args`.
     * @param values List of values of size `size`.
     * @param size Size of `indices` and `values` list.
     *
     * **Usage**
     * ```
     * for (int i = 0; i < size; ++i)
     * {
     *   // args[i] are argument lists of size arity
     *   for (int j = 0; j < arity; ++j)
     *   {
     *     // args[i][j] corresponds to value of jth argument of function f
     *   }
     *   // values[i] corresponds to the value of
     *   // (f args[i][0] ... args[i][arity - 1])
     * }
     * ```
     */
    class FunValue(
        val size: Int,
        val arity: Int,
        val args: Array<Array<BitwuzlaTerm>>,
        val values: Array<BitwuzlaTerm>
    )

    /**
     * Print a model for the current input formula.
     *
     * Requires that the last [bitwuzla_check_sat] query returned
     * [BitwuzlaResult.BITWUZLA_SAT].
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param format The output format for printing the model. Either `"btor"` for
     * the BTOR format, or `"smt2"` for the SMT-LIB v2 format.
     * @param file The file to print the model to.
     *
     * @see bitwuzla_check_sat
     */
    fun bitwuzlaPrintModel(bitwuzla: Bitwuzla, format: String, file: FilePtr) = file.use {
        bitwuzla_print_model(bitwuzla, format, it.ptr)
    }

    private external fun bitwuzla_print_model(bitwuzla: Bitwuzla, format: String, file: Pointer)


    /**
     * Print the current input formula.
     *
     * Requires that incremental solving is not enabled.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param format The output format for printing the formula. Either
     * `"aiger_ascii"` for the AIGER ascii format, `"aiger_binary"`
     * for the binary AIGER format, `"btor"` for the BTOR format, or
     * `"smt2"` for the SMT-LIB v2 format.
     * @param file The file to print the formula to.
     */
    fun bitwuzlaDumpFormula(bitwuzla: Bitwuzla, format: String, file: FilePtr) = file.use {
        bitwuzla_dump_formula(bitwuzla, format, it.ptr)
    }

    private external fun bitwuzla_dump_formula(bitwuzla: Bitwuzla, format: String, file: Pointer)


    /**
     * Parse input file.
     *
     * The format of the input file is auto detected.
     * Requires that no terms have been created yet.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param infile The input file.
     * @param infile_name The name of the input file.
     * @param outfile The output file.
     *
     * @return [ParseResult]
     *
     * @see bitwuzla_parse_format
     */
    fun bitwuzlaParse(
        bitwuzla: Bitwuzla,
        infile: FilePtr,
        infile_name: String,
        outfile: FilePtr
    ): ParseResult {
        val errorMsg = PointerByReference()
        val parsedStatus = IntByReference()
        val parsedSmt2 = IntByReference()
        val result = bitwuzla_parse(bitwuzla, infile.ptr, infile_name, outfile.ptr, errorMsg, parsedStatus, parsedSmt2)
        infile.close()
        outfile.close()
        return ParseResult(
            BitwuzlaResult.fromValue(result),
            errorMsg.value?.let { if (Pointer.NULL == it) null else it.getString(0) },
            BitwuzlaResult.fromValue(parsedStatus.value),
            parsedSmt2.value != 0
        )
    }

    private external fun bitwuzla_parse(
        bitwuzla: Bitwuzla,
        infile: Pointer,
        infile_name: String,
        outfile: Pointer,
        error_msg: PointerByReference,
        parsed_status: IntByReference,
        parsed_smt2: IntByReference
    ): Int

    /**
     * @param errorMsg Output parameter, stores an error message in case a parse
     * error occurred, else `null`.
     * @param parsedStatus Output parameter, stores the status of the input in case
     * of SMT-LIB v2 input, if given.
     * @param parsedSmt2 Output parameter, true if parsed input file has been
     * detected as SMT-LIB v2 input.
     * @param result [BitwuzlaResult.BITWUZLA_SAT] if the input formula was simplified to true,
     * [BitwuzlaResult.BITWUZLA_UNSAT] if it was simplified to false,
     * and [BitwuzlaResult.BITWUZLA_UNKNOWN] otherwise.
     */
    class ParseResult(
        val result: BitwuzlaResult,
        val errorMsg: String?,
        val parsedStatus: BitwuzlaResult,
        val parsedSmt2: Boolean
    )


    /**
     * Parse input file, assumed to be given in the specified format.
     *
     * Requires that no terms have been created yet.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param format The input format for printing the model. Either `"btor"` for
     * the BTOR format, `"btor2"` for the BTOR2 format, or `"smt2"`
     * for the SMT-LIB v2 format.
     * @param infile The input file.
     * @param infileName The name of the input file.
     * @param outfile The output file.
     *
     * @return [ParseFormatResult]
     *
     * @see bitwuzla_parse
     */

    fun bitwuzlaParseFormat(
        bitwuzla: Bitwuzla,
        format: String,
        infile: FilePtr,
        infileName: String,
        outfile: FilePtr
    ): ParseFormatResult {
        val errorMsg = PointerByReference()
        val parsedStatus = IntByReference()
        val result = bitwuzla_parse_format(
            bitwuzla, format, infile.ptr, infileName, outfile.ptr, errorMsg, parsedStatus
        )
        infile.close()
        outfile.close()
        return ParseFormatResult(
            BitwuzlaResult.fromValue(result),
            errorMsg.value?.let { if (Pointer.NULL == it) null else it.getString(0) },
            BitwuzlaResult.fromValue(parsedStatus.value)
        )
    }

    private external fun bitwuzla_parse_format(
        bitwuzla: Bitwuzla,
        format: String,
        infile: Pointer,
        infile_name: String,
        outfile: Pointer,
        error_msg: PointerByReference,
        parsed_status: IntByReference
    ): Int


    /**
     * @param errorMsg Output parameter, stores an error message in case a parse
     * error occurred, else `null`.
     * @param parsedStatus Output parameter, stores the status of the input in case
     * of SMT-LIB v2 input, if given.
     * @param result [BitwuzlaResult.BITWUZLA_SAT] if the input formula was simplified to true,
     * [BitwuzlaResult.BITWUZLA_UNSAT] if it was simplified to false,
     * and [BitwuzlaResult.BITWUZLA_UNKNOWN] otherwise.
     */
    class ParseFormatResult(
        val result: BitwuzlaResult,
        val errorMsg: String?,
        val parsedStatus: BitwuzlaResult
    )

    /**
     * Substitute a set of keys with their corresponding values in the given term.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param term The term in which the keys are to be substituted.
     * @param mapKeys The keys.
     * @param mapValues The mapped values.
     *
     * @return The resulting term from this substitution.
     */
    fun bitwuzlaSubstituteTerm(
        bitwuzla: Bitwuzla,
        term: BitwuzlaTerm,
        mapKeys: Array<BitwuzlaTerm>,
        mapValues: Array<BitwuzlaTerm>
    ): BitwuzlaTerm = bitwuzla_substitute_term(bitwuzla, term, mapKeys.size, mapKeys.mkPtr(), mapValues.mkPtr())

    private external fun bitwuzla_substitute_term(
        bitwuzla: Bitwuzla,
        term: BitwuzlaTerm,
        map_size: Int,
        map_keys: Pointer,
        map_values: Pointer
    ): BitwuzlaTerm


    /**
     * Substitute a set of keys with their corresponding values in the set of given
     * terms.
     *
     * The terms in `terms` are replaced with the terms resulting from this
     * substitutions.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param terms The terms in which the keys are to be substituted.
     * @param mapKeys The keys.
     * @param mapValues The mapped values.
     */

    fun bitwuzlaSubstituteTerms(
        bitwuzla: Bitwuzla,
        terms: Array<BitwuzlaTerm>,
        mapKeys: Array<BitwuzlaTerm>,
        mapValues: Array<BitwuzlaTerm>
    ) {
        val termsPtr = terms.mkPtr()
        bitwuzla_substitute_terms(bitwuzla, terms.size, termsPtr, mapKeys.size, mapKeys.mkPtr(), mapValues.mkPtr())
        val result = termsPtr.load(terms.size)
        for (i in terms.indices) {
            terms[i] = result[i]
        }
    }

    private external fun bitwuzla_substitute_terms(
        bitwuzla: Bitwuzla,
        terms_size: Int,
        terms: Pointer,
        map_size: Int,
        map_keys: Pointer,
        map_values: Pointer
    )

    /**
     * Compute the hash value for a sort.
     *
     * @param sort The sort.
     *
     * @return The hash value of the sort.
     */
    fun bitwuzlaSortHash(sort: BitwuzlaSort): Long =
        bitwuzla_sort_hash(sort)

    private external fun bitwuzla_sort_hash(sort: BitwuzlaSort): Long


    /**
     * Get the size of a bit-vector sort.
     *
     * Requires that given sort is a bit-vector sort.
     *
     * @param sort The sort.
     *
     * @return The size of the bit-vector sort.
     */
    fun bitwuzlaSortBvGetSize(sort: BitwuzlaSort): Int =
        bitwuzla_sort_bv_get_size(sort)

    private external fun bitwuzla_sort_bv_get_size(sort: BitwuzlaSort): Int


    /**
     * Get the exponent size of a floating-point sort.
     *
     * Requires that given sort is a floating-point sort.
     *
     * @param sort The sort.
     *
     * @return The exponent size of the floating-point sort.
     */
    fun bitwuzlaSortFpGetExpSize(sort: BitwuzlaSort): Int =
        bitwuzla_sort_fp_get_exp_size(sort)

    private external fun bitwuzla_sort_fp_get_exp_size(sort: BitwuzlaSort): Int


    /**
     * Get the significand size of a floating-point sort.
     *
     * Requires that given sort is a floating-point sort.
     *
     * @param sort The sort.
     *
     * @return The significand size of the floating-point sort.
     */
    fun bitwuzlaSortFpGetSigSize(sort: BitwuzlaSort): Int =
        bitwuzla_sort_fp_get_sig_size(sort)

    private external fun bitwuzla_sort_fp_get_sig_size(sort: BitwuzlaSort): Int


    /**
     * Get the index sort of an array sort.
     *
     * Requires that given sort is an array sort.
     *
     * @param sort The sort.
     *
     * @return The index sort of the array sort.
     */
    fun bitwuzlaSortArrayGetIndex(sort: BitwuzlaSort): BitwuzlaSort =
        bitwuzla_sort_array_get_index(sort)

    private external fun bitwuzla_sort_array_get_index(sort: BitwuzlaSort): BitwuzlaSort


    /**
     * Get the element sort of an array sort.
     *
     * Requires that given sort is an array sort.
     *
     * @param sort The sort.
     *
     * @return The element sort of the array sort.
     */
    fun bitwuzlaSortArrayGetElement(sort: BitwuzlaSort): BitwuzlaSort =
        bitwuzla_sort_array_get_element(sort)

    private external fun bitwuzla_sort_array_get_element(sort: BitwuzlaSort): BitwuzlaSort


    /**
     * Get the domain sorts of a function sort.
     *
     * The domain sorts are returned as an array of sorts of size `size`.
     * Requires that given sort is a function sort.
     *
     * @param sort The sort.
     *
     * @return The domain sorts of the function sort.
     */
    fun bitwuzlaSortFunGetDomainSorts(sort: BitwuzlaSort): Array<BitwuzlaSort> {
        val size = IntByReference()
        val result = bitwuzla_sort_fun_get_domain_sorts(sort, size)
        return result.load(size.value)
    }

    private external fun bitwuzla_sort_fun_get_domain_sorts(sort: BitwuzlaSort, size: IntByReference): Pointer


    /**
     * Get the codomain sort of a function sort.
     *
     * Requires that given sort is a function sort.
     *
     * @param sort The sort.
     *
     * @return The codomain sort of the function sort.
     */
    fun bitwuzlaSortFunGetCodomain(sort: BitwuzlaSort): BitwuzlaSort =
        bitwuzla_sort_fun_get_codomain(sort)

    private external fun bitwuzla_sort_fun_get_codomain(sort: BitwuzlaSort): BitwuzlaSort


    /**
     * Get the arity of a function sort.
     *
     * @param sort The sort.
     *
     * @return The number of arguments of the function sort.
     */
    fun bitwuzlaSortFunGetArity(sort: BitwuzlaSort): Int =
        bitwuzla_sort_fun_get_arity(sort)

    private external fun bitwuzla_sort_fun_get_arity(sort: BitwuzlaSort): Int


    /**
     * Determine if two sorts are equal.
     *
     * @param sort0 The first sort.
     * @param sort1 The second sort.
     *
     * @return True if the given sorts are equal.
     */
    fun bitwuzlaSortIsEqual(sort0: BitwuzlaSort, sort1: BitwuzlaSort): Boolean =
        bitwuzla_sort_is_equal(sort0, sort1)

    private external fun bitwuzla_sort_is_equal(sort0: BitwuzlaSort, sort1: BitwuzlaSort): Boolean


    /**
     * Determine if a sort is an array sort.
     *
     * @param sort The sort.
     *
     * @return True if `sort` is an array sort.
     */
    fun bitwuzlaSortIsArray(sort: BitwuzlaSort): Boolean =
        bitwuzla_sort_is_array(sort)

    private external fun bitwuzla_sort_is_array(sort: BitwuzlaSort): Boolean


    /**
     * Determine if a sort is a bit-vector sort.
     *
     * @param sort The sort.
     *
     * @return True if `sort` is a bit-vector sort.
     */
    fun bitwuzlaSortIsBv(sort: BitwuzlaSort): Boolean =
        bitwuzla_sort_is_bv(sort)

    private external fun bitwuzla_sort_is_bv(sort: BitwuzlaSort): Boolean


    /**
     * Determine if a sort is a floating-point sort.
     *
     * @param sort The sort.
     *
     * @return True if `sort` is a floating-point sort.
     */
    fun bitwuzlaSortIsFp(sort: BitwuzlaSort): Boolean =
        bitwuzla_sort_is_fp(sort)

    private external fun bitwuzla_sort_is_fp(sort: BitwuzlaSort): Boolean


    /**
     * Determine if a sort is a function sort.
     *
     * @param sort The sort.
     *
     * @return True if `sort` is a function sort.
     */
    fun bitwuzlaSortIsFun(sort: BitwuzlaSort): Boolean =
        bitwuzla_sort_is_fun(sort)

    private external fun bitwuzla_sort_is_fun(sort: BitwuzlaSort): Boolean


    /**
     * Determine if a sort is a Roundingmode sort.
     *
     * @param sort The sort.
     *
     * @return True if `sort` is a Roundingmode sort.
     */
    fun bitwuzlaSortIsRm(sort: BitwuzlaSort): Boolean =
        bitwuzla_sort_is_rm(sort)

    private external fun bitwuzla_sort_is_rm(sort: BitwuzlaSort): Boolean


    /**
     * Print sort.
     *
     * @param sort The sort.
     * @param format The output format for printing the term. Either `"btor"` for
     * the BTOR format, or `"smt2"` for the SMT-LIB v2 format. Note
     * for the `"btor"` this function won't do anything since BTOR
     * sorts are printed when printing the term via
     * bitwuzla_term_dump.
     * @param file The file to print the term to.
     */
    fun bitwuzlaSortDump(sort: BitwuzlaSort, format: String, file: FilePtr) = file.use {
        bitwuzla_sort_dump(sort, format, it.ptr)
    }

    private external fun bitwuzla_sort_dump(sort: BitwuzlaSort, format: String, file: Pointer)

    /**
     * Compute the hash value for a term.
     *
     * @param term The term.
     *
     * @return The hash value of the term.
     */
    fun bitwuzlaTermHash(term: BitwuzlaTerm): Long =
        bitwuzla_term_hash(term)

    private external fun bitwuzla_term_hash(term: BitwuzlaTerm): Long


    /**
     * Get the kind of a term.
     *
     * @param term The term.
     *
     * @return The kind of the given term.
     *
     * @see BitwuzlaKind
     */
    fun bitwuzlaTermGetKind(term: BitwuzlaTerm): BitwuzlaKind =
        BitwuzlaKind.fromValue(bitwuzla_term_get_kind(term))

    private external fun bitwuzla_term_get_kind(term: BitwuzlaTerm): Int


    /**
     * Get the child terms of a term.
     *
     * Returns `null` if given term does not have children.
     *
     * @param term The term.
     *
     * @return The children of `term` as an array of terms.
     */
    fun bitwuzlaTermGetChildren(term: BitwuzlaTerm): Array<BitwuzlaTerm> {
        val size = IntByReference()
        val result = bitwuzla_term_get_children(term, size)
        return result.load(size.value)
    }

    private external fun bitwuzla_term_get_children(term: BitwuzlaTerm, size: IntByReference): Pointer


    /**
     * Get the indices of an indexed term.
     *
     * Requires that given term is an indexed term.
     *
     * @param term The term.
     *
     * @return The children of `term` as an array of terms.
     */
    fun bitwuzlaTermGetIndices(term: BitwuzlaTerm): IntArray {
        val size = IntByReference()
        val result = bitwuzla_term_get_indices(term, size)
        if (Pointer.NULL == result) return intArrayOf()
        return result.getIntArray(0, size.value)
    }

    private external fun bitwuzla_term_get_indices(term: BitwuzlaTerm, size: IntByReference): Pointer


    /**
     * Determine if a term is an indexed term.
     *
     * @param term The term.
     *
     * @return True if `term` is an indexed term.
     */
    fun bitwuzlaTermIsIndexed(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_indexed(term)

    private external fun bitwuzla_term_is_indexed(term: BitwuzlaTerm): Boolean


    /**
     * Get the associated Bitwuzla instance of a term.
     *
     * @param term The term.
     *
     * @return The associated Bitwuzla instance.
     */
    fun bitwuzlaTermGetBitwuzla(term: BitwuzlaTerm): Bitwuzla =
        bitwuzla_term_get_bitwuzla(term)

    private external fun bitwuzla_term_get_bitwuzla(term: BitwuzlaTerm): Bitwuzla


    /**
     * Get the sort of a term.
     *
     * @param term The term.
     *
     * @return The sort of the term.
     */
    fun bitwuzlaTermGetSort(term: BitwuzlaTerm): BitwuzlaSort =
        bitwuzla_term_get_sort(term)

    private external fun bitwuzla_term_get_sort(term: BitwuzlaTerm): BitwuzlaSort


    /**
     * Get the index sort of an array term.
     *
     * Requires that given term is an array or an array store term.
     *
     * @param term The term.
     *
     * @return The index sort of the array term.
     */
    fun bitwuzlaTermArrayGetIndexSort(term: BitwuzlaTerm): BitwuzlaSort =
        bitwuzla_term_array_get_index_sort(term)

    private external fun bitwuzla_term_array_get_index_sort(term: BitwuzlaTerm): BitwuzlaSort


    /**
     * Get the element sort of an array term.
     *
     * Requires that given term is an array or an array store term.
     *
     * @param term The term.
     *
     * @return The element sort of the array term.
     */
    fun bitwuzlaTermArrayGetElementSort(term: BitwuzlaTerm): BitwuzlaSort =
        bitwuzla_term_array_get_element_sort(term)

    private external fun bitwuzla_term_array_get_element_sort(term: BitwuzlaTerm): BitwuzlaSort


    /**
     * Get the domain sorts of a function term.
     *
     * The domain sorts are returned as an array of sorts of size `size.
     * Requires that given term is an uninterpreted function, a lambda term, an
     * array store term, or an ite term over function terms.
     *
     * @param term The term.
     *
     * @return The domain sorts of the function term.
     */
    fun bitwuzlaTermFunGetDomainSorts(term: BitwuzlaTerm): Array<BitwuzlaSort> {
        val size = IntByReference()
        val result = bitwuzla_term_fun_get_domain_sorts(term, size)
        return result.load(size.value)
    }

    private external fun bitwuzla_term_fun_get_domain_sorts(term: BitwuzlaTerm, size: IntByReference): Pointer


    /**
     * Get the codomain sort of a function term.
     *
     * Requires that given term is an uninterpreted function, a lambda term, an
     * array store term, or an ite term over function terms.
     *
     * @param term The term.
     *
     * @return The codomain sort of the function term.
     */
    fun bitwuzlaTermFunGetCodomainSort(term: BitwuzlaTerm): BitwuzlaSort =
        bitwuzla_term_fun_get_codomain_sort(term)

    private external fun bitwuzla_term_fun_get_codomain_sort(term: BitwuzlaTerm): BitwuzlaSort


    /**
     * Get the bit-width of a bit-vector term.
     *
     * Requires that given term is a bit-vector term.
     *
     * @param term The term.
     *
     * @return The bit-width of the bit-vector term.
     */
    fun bitwuzlaTermBvGetSize(term: BitwuzlaTerm): Int =
        bitwuzla_term_bv_get_size(term)

    private external fun bitwuzla_term_bv_get_size(term: BitwuzlaTerm): Int


    /**
     * Get the bit-width of the exponent of a floating-point term.
     *
     * Requires that given term is a floating-point term.
     *
     * @param term The term.
     *
     * @return The bit-width of the exponent of the floating-point term.
     */
    fun bitwuzlaTermFpGetExpSize(term: BitwuzlaTerm): Int =
        bitwuzla_term_fp_get_exp_size(term)

    private external fun bitwuzla_term_fp_get_exp_size(term: BitwuzlaTerm): Int


    /**
     * Get the bit-width of the significand of a floating-point term.
     *
     * Requires that given term is a floating-point term.
     *
     * @param term The term.
     *
     * @return The bit-width of the significand of the floating-point term.
     */
    fun bitwuzlaTermFpGetSigSize(term: BitwuzlaTerm): Int =
        bitwuzla_term_fp_get_sig_size(term)

    private external fun bitwuzla_term_fp_get_sig_size(term: BitwuzlaTerm): Int


    /**
     * Get the arity of a function term.
     *
     * Requires that given term is a function term.
     *
     * @param term The term.
     *
     * @return The arity of the function term.
     */
    fun bitwuzlaTermFunGetArity(term: BitwuzlaTerm): Int =
        bitwuzla_term_fun_get_arity(term)

    private external fun bitwuzla_term_fun_get_arity(term: BitwuzlaTerm): Int


    /**
     * Get the symbol of a term.
     *
     * @param term The term.
     *
     * @return The symbol of `term`. `null` if no symbol is defined.
     */
    fun bitwuzlaTermGetSymbol(term: BitwuzlaTerm): String? =
        bitwuzla_term_get_symbol(term)

    private external fun bitwuzla_term_get_symbol(term: BitwuzlaTerm): String?


    /**
     * Set the symbol of a term.
     *
     * @param term The term.
     * @param symbol The symbol.
     */
    fun bitwuzlaTermSetSymbol(term: BitwuzlaTerm, symbol: String) =
        bitwuzla_term_set_symbol(term, symbol)

    private external fun bitwuzla_term_set_symbol(term: BitwuzlaTerm, symbol: String)


    /**
     * Determine if the sorts of two terms are equal.
     *
     * @param term0 The first term.
     * @param term1 The second term.
     *
     * @return True if the sorts of `term0` and `term1` are equal.
     */
    fun bitwuzlaTermIsEqualSort(term0: BitwuzlaTerm, term1: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_equal_sort(term0, term1)

    private external fun bitwuzla_term_is_equal_sort(term0: BitwuzlaTerm, term1: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is an array term.
     *
     * @param term The term.
     *
     * @return True if `term` is an array term.
     */
    fun bitwuzlaTermIsArray(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_array(term)

    private external fun bitwuzla_term_is_array(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a constant.
     *
     * @param term The term.
     *
     * @return True if `term` is a constant.
     */
    fun bitwuzlaTermIsConst(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_const(term)

    private external fun bitwuzla_term_is_const(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a function.
     *
     * @param term The term.
     *
     * @return True if `term` is a function.
     */
    fun bitwuzlaTermIsFun(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_fun(term)

    private external fun bitwuzla_term_is_fun(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a variable.
     *
     * @param term The term.
     *
     * @return True if `term` is a variable.
     */
    fun bitwuzlaTermIsVar(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_var(term)

    private external fun bitwuzla_term_is_var(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a bound variable.
     *
     * @param term The term.
     *
     * @return True if `term` is a variable and bound.
     */
    fun bitwuzlaTermIsBoundVar(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_bound_var(term)

    private external fun bitwuzla_term_is_bound_var(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a value.
     *
     * @param term The term.
     *
     * @return True if `term` is a value.
     */
    fun bitwuzlaTermIsValue(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_value(term)

    private external fun bitwuzla_term_is_value(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a bit-vector value.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector value.
     */
    fun bitwuzlaTermIsBvValue(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_bv_value(term)

    private external fun bitwuzla_term_is_bv_value(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a floating-point value.
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point value.
     */
    fun bitwuzlaTermIsFpValue(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_fp_value(term)

    private external fun bitwuzla_term_is_fp_value(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a rounding mode value.
     *
     * @param term The term.
     *
     * @return True if `term` is a rounding mode value.
     */
    fun bitwuzlaTermIsRmValue(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_rm_value(term)

    private external fun bitwuzla_term_is_rm_value(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a bit-vector term.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector term.
     */
    fun bitwuzlaTermIsBv(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_bv(term)

    private external fun bitwuzla_term_is_bv(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a floating-point term.
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point term.
     */
    fun bitwuzlaTermIsFp(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_fp(term)

    private external fun bitwuzla_term_is_fp(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a rounding mode term.
     *
     * @param term The term.
     *
     * @return True if `term` is a rounding mode term.
     */
    fun bitwuzlaTermIsRm(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_rm(term)

    private external fun bitwuzla_term_is_rm(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a bit-vector value representing zero.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector zero value.
     */
    fun bitwuzlaTermIsBvValueZero(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_bv_value_zero(term)

    private external fun bitwuzla_term_is_bv_value_zero(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a bit-vector value representing one.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector one value.
     */
    fun bitwuzlaTermIsBvValueOne(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_bv_value_one(term)

    private external fun bitwuzla_term_is_bv_value_one(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a bit-vector value with all bits set to one.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector value with all bits set to one.
     */
    fun bitwuzlaTermIsBvValueOnes(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_bv_value_ones(term)

    private external fun bitwuzla_term_is_bv_value_ones(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a bit-vector minimum signed value.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector value with the most significant bit
     * set to 1 and all other bits set to 0.
     */
    fun bitwuzlaTermIsBvValueMinSigned(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_bv_value_min_signed(term)

    private external fun bitwuzla_term_is_bv_value_min_signed(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a bit-vector maximum signed value.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector value with the most significant bit
     * set to 0 and all other bits set to 1.
     */
    fun bitwuzlaTermIsBvValueMaxSigned(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_bv_value_max_signed(term)

    private external fun bitwuzla_term_is_bv_value_max_signed(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a floating-point positive zero (+zero) value.
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point +zero value.
     */
    fun bitwuzlaTermIsFpValuePosZero(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_fp_value_pos_zero(term)

    private external fun bitwuzla_term_is_fp_value_pos_zero(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a floating-point value negative zero (-zero).
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point value negative zero.
     */
    fun bitwuzlaTermIsFpValueNegZero(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_fp_value_neg_zero(term)

    private external fun bitwuzla_term_is_fp_value_neg_zero(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a floating-point positive infinity (+oo) value.
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point +oo value.
     */
    fun bitwuzlaTermIsFpValuePosInf(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_fp_value_pos_inf(term)

    private external fun bitwuzla_term_is_fp_value_pos_inf(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a floating-point negative infinity (-oo) value.
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point -oo value.
     */
    fun bitwuzlaTermIsFpValueNegInf(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_fp_value_neg_inf(term)

    private external fun bitwuzla_term_is_fp_value_neg_inf(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a floating-point NaN value.
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point NaN value.
     */
    fun bitwuzlaTermIsFpValueNan(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_fp_value_nan(term)

    private external fun bitwuzla_term_is_fp_value_nan(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a rounding mode RNA value.
     *
     * @param term The term.
     *
     * @return True if `term` is a roundindg mode RNA value.
     */
    fun bitwuzlaTermIsRmValueRna(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_rm_value_rna(term)

    private external fun bitwuzla_term_is_rm_value_rna(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a rounding mode RNE value.
     *
     * @param term The term.
     *
     * @return True if `term` is a rounding mode RNE value.
     */
    fun bitwuzlaTermIsRmValueRne(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_rm_value_rne(term)

    private external fun bitwuzla_term_is_rm_value_rne(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a rounding mode RTN value.
     *
     * @param term The term.
     *
     * @return True if `term` is a rounding mode RTN value.
     */
    fun bitwuzlaTermIsRmValueRtn(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_rm_value_rtn(term)

    private external fun bitwuzla_term_is_rm_value_rtn(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a rounding mode RTP value.
     *
     * @param term The term.
     *
     * @return True if `term` is a rounding mode RTP value.
     */
    fun bitwuzlaTermIsRmValueRtp(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_rm_value_rtp(term)

    private external fun bitwuzla_term_is_rm_value_rtp(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a rounding mode RTZ value.
     *
     * @param term The term.
     *
     * @return True if `term` is a rounding mode RTZ value.
     */
    fun bitwuzlaTermIsRmValueRtz(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_rm_value_rtz(term)

    private external fun bitwuzla_term_is_rm_value_rtz(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a constant array.
     *
     * @param term The term.
     *
     * @return True if `term` is a constant array.
     */
    fun bitwuzlaTermIsConstArray(term: BitwuzlaTerm): Boolean =
        bitwuzla_term_is_const_array(term)

    private external fun bitwuzla_term_is_const_array(term: BitwuzlaTerm): Boolean


    /**
     * Print term .
     *
     * @param term The term.
     * @param format The output format for printing the term. Either `"btor"` for the
     * BTOR format, or `"smt2"` for the SMT-LIB v2 format.
     * @param file The file to print the term to.
     */
    fun bitwuzlaTermDump(term: BitwuzlaTerm, format: String, file: FilePtr) = file.use {
        bitwuzla_term_dump(term, format, it.ptr)
    }

    private external fun bitwuzla_term_dump(term: BitwuzlaTerm, format: String, file: Pointer)


    /**
     * Get the string representation of a term kind.
     *
     * @return A string representation of the given term kind.
     */
    fun bitwuzlaKindToString(kind: BitwuzlaKind): String =
        bitwuzla_kind_to_string(kind.value)

    private external fun bitwuzla_kind_to_string(kind: Int): String


    /**
     * Get the string representation of a result.
     *
     * @return A string representation of the given result.
     */
    fun bitwuzlaResultToString(result: BitwuzlaResult): String =
        bitwuzla_result_to_string(result.value)

    private external fun bitwuzla_result_to_string(result: Int): String


    /**
     * Get the string representation of a rounding mode.
     *
     * @return A string representation of the rounding mode.
     */
    fun bitwuzlaRmToString(rm: BitwuzlaRoundingMode): String =
        bitwuzla_rm_to_string(rm.value)

    private external fun bitwuzla_rm_to_string(rm: Int): String


    private fun <T : Pointer> Array<T>.mkPtr(): Pointer {
        val memory = Memory(Native.POINTER_SIZE.toLong() * size)
        for (i in indices) {
            memory.setPointer(Native.POINTER_SIZE.toLong() * i, this[i])
        }
        return memory
    }

    private fun Pointer?.load(size: Int): Array<Pointer> {
        if (this == null || Pointer.NULL == this || size == 0) return emptyArray()
        return getPointerArray(0, size)
    }

}
