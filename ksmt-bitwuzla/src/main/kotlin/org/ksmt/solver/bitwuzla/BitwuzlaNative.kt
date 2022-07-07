@file:Suppress("FunctionName", "unused", "UNUSED_PARAMETER")

package org.ksmt.solver.bitwuzla

import com.sun.jna.Memory
import com.sun.jna.Native
import com.sun.jna.Pointer
import com.sun.jna.ptr.IntByReference
import com.sun.jna.ptr.PointerByReference
import org.ksmt.solver.bitwuzla.*
import java.io.File

typealias Bitwuzla = Pointer
typealias BitwuzlaTerm = Pointer
typealias BitwuzlaSort = Pointer

object BitwuzlaNative {
    init {
        Native.register("bitwuzla")
    }

    /**
     * Get the string representation of a term kind.<br></br>
     *
     * @return A string representation of the given term kind.<br></br>
     * Original signature : `char* bitwuzla_kind_to_string(BitwuzlaKind)`<br></br>
     * *native declaration : bitwuzla.h:2000*
     */
    fun bitwuzla_kind_to_string(kind: BitwuzlaKind): String = bitwuzla_kind_to_string(kind.value)

    external fun bitwuzla_kind_to_string(kind: Int): String

    /**
     * Get the string representation of a result.<br></br>
     *
     * @return A string representation of the given result.<br></br>
     * Original signature : `char* bitwuzla_result_to_string(BitwuzlaResult)`<br></br>
     * *native declaration : bitwuzla.h:2017*
     */
    fun bitwuzla_result_to_string(result: BitwuzlaResult): String = bitwuzla_result_to_string(result.value)

    external fun bitwuzla_result_to_string(result: Int): String

    /**
     * Get the string representation of a rounding mode.<br></br>
     *
     * @return A string representation of the rounding mode.<br></br>
     * Original signature : `char* bitwuzla_rm_to_string(BitwuzlaRoundingMode)`<br></br>
     * *native declaration : bitwuzla.h:2089*
     */
    fun bitwuzla_rm_to_string(rm: BitwuzlaRoundingMode): String = bitwuzla_rm_to_string(rm.value)

    external fun bitwuzla_rm_to_string(rm: Int): String

    /**
     * Create a new Bitwuzla instance.<br></br>
     * * The returned instance must be deleted via `bitwuzla_delete()`.<br></br>
     * * @return A pointer to the created Bitwuzla instance.<br></br>
     * * @see<br></br>
     * * `bitwuzla_delete`<br></br>
     * Original signature : `Bitwuzla* bitwuzla_new()`<br></br>
     * *native declaration : bitwuzla.h:2112*
     */
    external fun bitwuzla_new(): Bitwuzla

    /**
     * Delete a Bitwuzla instance.<br></br>
     * * The given instance must have been created via `bitwuzla_new()`.<br></br>
     * * @param bitwuzla The Bitwuzla instance to delete.<br></br>
     * * @see<br></br>
     * * `bitwuzla_new`<br></br>
     * Original signature : `void bitwuzla_delete(Bitwuzla*)`<br></br>
     * *native declaration : bitwuzla.h:2124*
     */
    external fun bitwuzla_delete(bitwuzla: Bitwuzla)

    /**
     * Reset a Bitwuzla instance.<br></br>
     * * This deletes the given instance and creates a new instance in place.<br></br>
     * The given instance must have been created via `bitwuzla_new()`.<br></br>
     * * @note All sorts and terms associated with the given instance are released<br></br>
     * and thus invalidated.<br></br>
     * * @param bitwuzla The Bitwuzla instance to reset.<br></br>
     * * @see<br></br>
     * * `bitwuzla_new`<br></br>
     * Original signature : `void bitwuzla_reset(Bitwuzla*)`<br></br>
     * *native declaration : bitwuzla.h:2140*
     */
    external fun bitwuzla_reset(bitwuzla: Bitwuzla)

    /**
     * Get copyright information.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     * Original signature : `char* bitwuzla_copyright(Bitwuzla*)`<br></br>
     * *native declaration : bitwuzla.h:2147*
     */
    external fun bitwuzla_copyright(bitwuzla: Bitwuzla): String

    /**
     * Get version information.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     * Original signature : `char* bitwuzla_version(Bitwuzla*)`<br></br>
     * *native declaration : bitwuzla.h:2154*
     */
    external fun bitwuzla_version(bitwuzla: Bitwuzla): String

    /**
     * Get git information.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     * Original signature : `char* bitwuzla_git_id(Bitwuzla*)`<br></br>
     * *native declaration : bitwuzla.h:2161*
     */
    external fun bitwuzla_git_id(bitwuzla: Bitwuzla): String

    /**
     * If termination callback function has been configured via<br></br>
     * `bitwuzla_set_termination_callback()`, call this termination function.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     * * @return True if `bitwuzla` has been terminated.<br></br>
     * * @see<br></br>
     * * `bitwuzla_set_termination_callback`<br></br>
     * * `bitwuzla_get_termination_callback_state`<br></br>
     * Original signature : `bool bitwuzla_terminate(Bitwuzla*)`<br></br>
     * *native declaration : bitwuzla.h:2175*
     */
    external fun bitwuzla_terminate(bitwuzla: Bitwuzla): Boolean

    /**
     * Configure a termination callback function.<br></br>
     * * The `state` of the callback can be retrieved via<br></br>
     * `bitwuzla_get_termination_callback_state()`.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param fun   The callback function, returns a value != 0 if `bitwuzla` has<br></br>
     * been terminated.<br></br>
     * @param state The argument to the callback function.<br></br>
     * * @see<br></br>
     * * `bitwuzla_terminate`<br></br>
     * * `bitwuzla_get_termination_callback_state`<br></br>
     * Original signature : `void bitwuzla_set_termination_callback(Bitwuzla*, bitwuzla_set_termination_callback_fun_callback*, void*)`<br></br>
     * *native declaration : bitwuzla.h:2192*
     */
    fun bitwuzla_set_termination_callback(bitwuzla: Bitwuzla, `fun`: Pointer, state: Pointer) {
        throw UnsupportedOperationException("operation is not implemented")
    }

    /**
     * Get the state of the termination callback function.<br></br>
     * * The returned object representing the state of the callback corresponds to<br></br>
     * the `state` configured as argument to the callback function via<br></br>
     * `bitwuzla_set_termination_callback()`.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     * * @return The object passed as argument `state` to the callback function.<br></br>
     * * @see<br></br>
     * * `bitwuzla_terminate`<br></br>
     * * `bitwuzla_set_termination_callback`<br></br>
     * Original signature : `void* bitwuzla_get_termination_callback_state(Bitwuzla*)`<br></br>
     * *native declaration : bitwuzla.h:2211*
     */
    fun bitwuzla_get_termination_callback_state(bitwuzla: Bitwuzla): Pointer {
        throw UnsupportedOperationException("operation is not implemented")
    }

    /**
     * Configure an abort callback function, which is called instead of exit<br></br>
     * on abort conditions.<br></br>
     * * @note This function is not thread safe (the function pointer is maintained<br></br>
     * as a global variable). It you use threading, make sure to set the<br></br>
     * abort callback prior to creating threads.<br></br>
     * * @param fun The callback function, the argument `msg` explains the reason<br></br>
     * for the abort.<br></br>
     * Original signature : `void bitwuzla_set_abort_callback(bitwuzla_set_abort_callback_fun_callback*)`<br></br>
     * *native declaration : bitwuzla.h:2224*
     */
    fun bitwuzla_set_abort_callback(`fun`: Pointer) {
        throw UnsupportedOperationException("operation is not implemented")
    }

    /**
     * Set option.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param option The option.<br></br>
     * @param `val`    The option value.<br></br>
     * * @see<br></br>
     * * `BitwuzlaOption`<br></br>
     * Original signature : `void bitwuzla_set_option(Bitwuzla*, BitwuzlaOption, uint32_t)`<br></br>
     * *native declaration : bitwuzla.h:2236*
     */
    fun bitwuzla_set_option(bitwuzla: Bitwuzla, option: BitwuzlaOption, value: Int) {
        bitwuzla_set_option(bitwuzla, option.value, value)
    }

    external fun bitwuzla_set_option(bitwuzla: Bitwuzla, option: Int, value: Int)

    /**
     * Set option value for string options.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param option The option.<br></br>
     * @param `val`    The option string value.<br></br>
     * * @see<br></br>
     * * `BitwuzlaOption`<br></br>
     * Original signature : `void bitwuzla_set_option_str(Bitwuzla*, BitwuzlaOption, const char*)`<br></br>
     * *native declaration : bitwuzla.h:2250*
     */
    fun bitwuzla_set_option_str(bitwuzla: Bitwuzla, option: BitwuzlaOption, value: String) {
        bitwuzla_set_option_str(bitwuzla, option.value, value)
    }

    external fun bitwuzla_set_option_str(bitwuzla: Bitwuzla, option: Int, value: String)

    /**
     * Get the current value of an option.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param option The option.<br></br>
     * * @return The option value.<br></br>
     * * @see<br></br>
     * * `BitwuzlaOption`<br></br>
     * Original signature : `uint32_t bitwuzla_get_option(Bitwuzla*, BitwuzlaOption)`<br></br>
     * *native declaration : bitwuzla.h:2265*
     */
    fun bitwuzla_get_option(bitwuzla: Bitwuzla, option: BitwuzlaOption): Int =
        bitwuzla_get_option(bitwuzla, option.value)

    external fun bitwuzla_get_option(bitwuzla: Bitwuzla, option: Int): Int

    /**
     * Get the current value of an option as a string if option can be configured<br></br>
     * via a string value.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param option The option.<br></br>
     * * @return The option value.<br></br>
     * * @see<br></br>
     * * `BitwuzlaOption`<br></br>
     * * `bitwuzla_set_option_str`<br></br>
     * Original signature : `char* bitwuzla_get_option_str(Bitwuzla*, BitwuzlaOption)`<br></br>
     * *native declaration : bitwuzla.h:2280*
     */
    fun bitwuzla_get_option_str(bitwuzla: Bitwuzla, option: BitwuzlaOption): String =
        bitwuzla_get_option_str(bitwuzla, option.value)

    external fun bitwuzla_get_option_str(bitwuzla: Bitwuzla, option: Int): String

    /**
     * Get the details of an option.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param option The option.<br></br>
     * @param info   The option info to populate, will be valid until the next<br></br>
     * `bitwuzla_get_option_info` call.<br></br>
     * * @see<br></br>
     * * `BitwuzlaOptionInfo`<br></br>
     * Original signature : `void bitwuzla_get_option_info(Bitwuzla*, BitwuzlaOption, BitwuzlaOptionInfo*)`<br></br>
     * *native declaration : bitwuzla.h:2293*
     */
    fun bitwuzla_get_option_info(bitwuzla: Bitwuzla, option: BitwuzlaOption): Pointer {
        throw UnsupportedOperationException("operation is not implemented")
    }

    /**
     * Create an array sort.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param index   The index sort of the array sort.<br></br>
     * @param element The element sort of the array sort.<br></br>
     * * @return An array sort which maps sort `index` to sort `element`.<br></br>
     * * @see<br></br>
     * * `bitwuzla_sort_is_array`<br></br>
     * * `bitwuzla_sort_array_get_index`<br></br>
     * * `bitwuzla_sort_array_get_element`<br></br>
     * * `bitwuzla_term_is_array`<br></br>
     * * `bitwuzla_term_array_get_index_sort`<br></br>
     * * `bitwuzla_term_array_get_element_sort`<br></br>
     * Original signature : `BitwuzlaSort* bitwuzla_mk_array_sort(Bitwuzla*, const BitwuzlaSort*, const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:2314*
     */
    external fun bitwuzla_mk_array_sort(
        bitwuzla: Bitwuzla,
        index: BitwuzlaSort,
        element: BitwuzlaSort
    ): BitwuzlaSort

    /**
     * Create a Boolean sort.<br></br>
     * * @note A Boolean sort is a bit-vector sort of size 1.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     * * @return A Boolean sort.<br></br>
     * Original signature : `BitwuzlaSort* bitwuzla_mk_bool_sort(Bitwuzla*)`<br></br>
     * *native declaration : bitwuzla.h:2327*
     */
    external fun bitwuzla_mk_bool_sort(bitwuzla: Bitwuzla): BitwuzlaSort

    /**
     * Create a bit-vector sort of given size.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param size The size of the bit-vector sort.<br></br>
     * * @return A bit-vector sort of given size.<br></br>
     * * @see<br></br>
     * * `bitwuzla_sort_is_bv`<br></br>
     * * `bitwuzla_sort_bv_get_size`<br></br>
     * * `bitwuzla_term_is_bv`<br></br>
     * * `bitwuzla_term_bv_get_size`<br></br>
     * Original signature : `BitwuzlaSort* bitwuzla_mk_bv_sort(Bitwuzla*, uint32_t)`<br></br>
     * *native declaration : bitwuzla.h:2343*
     */
    external fun bitwuzla_mk_bv_sort(bitwuzla: Bitwuzla, size: Int): BitwuzlaSort

    /**
     * Create a floating-point sort of given exponent and significand size.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param exp_size The size of the exponent.<br></br>
     * @param sig_size The size of the significand (including sign bit).<br></br>
     * * @return A floating-point sort of given format.<br></br>
     * * @see<br></br>
     * * `bitwuzla_sort_is_fp`<br></br>
     * * `bitwuzla_sort_fp_get_exp_size`<br></br>
     * * `bitwuzla_sort_fp_get_sig_size`<br></br>
     * * `bitwuzla_term_is_fp`<br></br>
     * * `bitwuzla_term_fp_get_exp_size`<br></br>
     * * `bitwuzla_term_fp_get_sig_size`<br></br>
     * Original signature : `BitwuzlaSort* bitwuzla_mk_fp_sort(Bitwuzla*, uint32_t, uint32_t)`<br></br>
     * *native declaration : bitwuzla.h:2362*
     */
    external fun bitwuzla_mk_fp_sort(bitwuzla: Bitwuzla, exp_size: Int, sig_size: Int): BitwuzlaSort

    /**
     * Create a function sort.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param arity    The number of arguments to the function.<br></br>
     * @param domain   The domain sorts (the sorts of the arguments). The number of<br></br>
     * sorts in this vector must match `arity`.<br></br>
     * @param codomain The codomain sort (the sort of the return value).<br></br>
     * * @return A function sort of given domain and codomain sorts.<br></br>
     * * @see<br></br>
     * * `bitwuzla_sort_is_fun`<br></br>
     * * `bitwuzla_sort_fun_get_arity`<br></br>
     * * `bitwuzla_sort_fun_get_domain_sorts`<br></br>
     * * `bitwuzla_sort_fun_get_codomain`<br></br>
     * * `bitwuzla_term_is_fun`<br></br>
     * * `bitwuzla_term_fun_get_arity`<br></br>
     * * `bitwuzla_term_fun_get_domain_sorts`<br></br>
     * * `bitwuzla_term_fun_get_codomain_sort`<br></br>
     * Original signature : `BitwuzlaSort* bitwuzla_mk_fun_sort(Bitwuzla*, uint32_t, const BitwuzlaSort*[], const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:2387*
     */

    external fun bitwuzla_mk_fun_sort(
        bitwuzla: Bitwuzla,
        arity: Int,
        domain: Pointer,
        codomain: BitwuzlaSort
    ): BitwuzlaSort

    fun bitwuzla_mk_fun_sort(
        bitwuzla: Bitwuzla,
        arity: Int,
        domain: Array<BitwuzlaSort>,
        codomain: BitwuzlaSort
    ): BitwuzlaSort = bitwuzla_mk_fun_sort(bitwuzla, arity, domain.mkPtr(), codomain)

    /**
     * Create a Roundingmode sort.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     * * @return A Roundingmode sort.<br></br>
     * * @see<br></br>
     * * `bitwuzla_sort_is_rm`<br></br>
     * * `bitwuzla_term_is_rm`<br></br>
     * Original signature : `BitwuzlaSort* bitwuzla_mk_rm_sort(Bitwuzla*)`<br></br>
     * *native declaration : bitwuzla.h:2403*
     */
    external fun bitwuzla_mk_rm_sort(bitwuzla: Bitwuzla): BitwuzlaSort

    /**
     * Create a true value.<br></br>
     * * @note This creates a bit-vector value 1 of size 1.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     * * @return A term representing the bit-vector value 1 of size 1.<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_true(Bitwuzla*)`<br></br>
     * *native declaration : bitwuzla.h:2414*
     */
    external fun bitwuzla_mk_true(bitwuzla: Bitwuzla): BitwuzlaTerm

    /**
     * Create a false value.<br></br>
     * * @note This creates a bit-vector value 0 of size 1.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     * * @return A term representing the bit-vector value 0 of size 1.<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_false(Bitwuzla*)`<br></br>
     * *native declaration : bitwuzla.h:2425*
     */
    external fun bitwuzla_mk_false(bitwuzla: Bitwuzla): BitwuzlaTerm

    /**
     * Create a bit-vector value zero.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param sort The sort of the value.<br></br>
     * * @return A term representing the bit-vector value 0 of given sort.<br></br>
     * * @see<br></br>
     * * `bitwuzla_mk_bv_sort`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_bv_zero(Bitwuzla*, const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:2438*
     */
    external fun bitwuzla_mk_bv_zero(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a bit-vector value one.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param sort The sort of the value.<br></br>
     * * @return A term representing the bit-vector value 1 of given sort.<br></br>
     * * @see<br></br>
     * * `bitwuzla_mk_bv_sort`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_bv_one(Bitwuzla*, const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:2452*
     */
    external fun bitwuzla_mk_bv_one(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a bit-vector value where all bits are set to 1.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param sort The sort of the value.<br></br>
     * * @return A term representing the bit-vector value of given sort<br></br>
     * where all bits are set to 1.<br></br>
     * * @see<br></br>
     * * `bitwuzla_mk_bv_sort`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_bv_ones(Bitwuzla*, const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:2467*
     */
    external fun bitwuzla_mk_bv_ones(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a bit-vector minimum signed value.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param sort The sort of the value.<br></br>
     * * @return A term representing the bit-vector value of given sort where the MSB<br></br>
     * is set to 1 and all remaining bits are set to 0.<br></br>
     * * @see<br></br>
     * * `bitwuzla_mk_bv_sort`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_bv_min_signed(Bitwuzla*, const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:2482*
     */
    external fun bitwuzla_mk_bv_min_signed(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a bit-vector maximum signed value.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param sort The sort of the value.<br></br>
     * * @return A term representing the bit-vector value of given sort where the MSB<br></br>
     * is set to 0 and all remaining bits are set to 1.<br></br>
     * * @see<br></br>
     * * `bitwuzla_mk_bv_sort`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_bv_max_signed(Bitwuzla*, const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:2496*
     */
    external fun bitwuzla_mk_bv_max_signed(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a floating-point positive zero value (SMT-LIB: `+zero`).<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param sort The sort of the value.<br></br>
     * * @return A term representing the floating-point positive zero value of given<br></br>
     * floating-point sort.<br></br>
     * * @see<br></br>
     * * `bitwuzla_mk_fp_sort`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_fp_pos_zero(Bitwuzla*, const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:2511*
     */
    external fun bitwuzla_mk_fp_pos_zero(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a floating-point negative zero value (SMT-LIB: `-zero`).<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param sort The sort of the value.<br></br>
     * * @return A term representing the floating-point negative zero value of given<br></br>
     * floating-point sort.<br></br>
     * * @see<br></br>
     * * `bitwuzla_mk_fp_sort`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_fp_neg_zero(Bitwuzla*, const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:2526*
     */
    external fun bitwuzla_mk_fp_neg_zero(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a floating-point positive infinity value (SMT-LIB: `+oo`).<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param sort The sort of the value.<br></br>
     * * @return A term representing the floating-point positive infinity value of<br></br>
     * given floating-point sort.<br></br>
     * * @see<br></br>
     * * `bitwuzla_mk_fp_sort`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_fp_pos_inf(Bitwuzla*, const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:2541*
     */
    external fun bitwuzla_mk_fp_pos_inf(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a floating-point negative infinity value (SMT-LIB: `-oo`).<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param sort The sort of the value.<br></br>
     * * @return A term representing the floating-point negative infinity value of<br></br>
     * given floating-point sort.<br></br>
     * * @see<br></br>
     * * `bitwuzla_mk_fp_sort`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_fp_neg_inf(Bitwuzla*, const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:2556*
     */
    external fun bitwuzla_mk_fp_neg_inf(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a floating-point NaN value.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param sort The sort of the value.<br></br>
     * * @return A term representing the floating-point NaN value of given<br></br>
     * floating-point sort.<br></br>
     * * @see<br></br>
     * * `bitwuzla_mk_fp_sort`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_fp_nan(Bitwuzla*, const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:2571*
     */
    external fun bitwuzla_mk_fp_nan(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a bit-vector value from its string representation.<br></br>
     * * Parameter `base` determines the base of the string representation.<br></br>
     * * @note Given value must fit into a bit-vector of given size (sort).<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param sort  The sort of the value.<br></br>
     * @param value A string representing the value.<br></br>
     * @param base  The base in which the string is given.<br></br>
     * * @return A term of kind BITWUZLA_KIND_VAL, representing the bit-vector value<br></br>
     * of given sort.<br></br>
     * * @see<br></br>
     * * `bitwuzla_mk_bv_sort`<br></br>
     * * `BitwuzlaBVBase`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_bv_value(Bitwuzla*, const BitwuzlaSort*, const char*, BitwuzlaBVBase)`<br></br>
     * *native declaration : bitwuzla.h:2593*
     */
    external fun bitwuzla_mk_bv_value(bitwuzla: Bitwuzla, sort: BitwuzlaSort, value: String, base: Int): BitwuzlaTerm

    fun bitwuzla_mk_bv_value(
        bitwuzla: Bitwuzla,
        sort: BitwuzlaSort,
        value: String,
        base: BitwuzlaBVBase
    ): BitwuzlaTerm = bitwuzla_mk_bv_value(bitwuzla, sort, value, base.value)

    /**
     * Create a bit-vector value from its unsigned integer representation.<br></br>
     * * @note If given value does not fit into a bit-vector of given size (sort),<br></br>
     * the value is truncated to fit.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param sort  The sort of the value.<br></br>
     * @param value The unsigned integer representation of the bit-vector value.<br></br>
     * * @return A term of kind BITWUZLA_KIND_VAL, representing the bit-vector value<br></br>
     * of given sort.<br></br>
     * * @see<br></br>
     * * `bitwuzla_mk_bv_sort`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_bv_value_uint64(Bitwuzla*, const BitwuzlaSort*, uint64_t)`<br></br>
     * *native declaration : bitwuzla.h:2614*
     */
    external fun bitwuzla_mk_bv_value_uint64(bitwuzla: Bitwuzla, sort: BitwuzlaSort, value: Long): BitwuzlaTerm

    /**
     * Create a floating-point value from its IEEE 754 standard representation<br></br>
     * given as three bit-vector values representing the sign bit, the exponent and<br></br>
     * the significand.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param bv_sign        The sign bit.<br></br>
     * @param bv_exponent    The exponent bit-vector value.<br></br>
     * @param bv_significand The significand bit-vector value.<br></br>
     * * @return A term of kind BITWUZLA_KIND_VAL, representing the floating-point<br></br>
     * value.<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_fp_value(Bitwuzla*, const BitwuzlaTerm*, const BitwuzlaTerm*, const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:2631*
     */
    external fun bitwuzla_mk_fp_value(
        bitwuzla: Bitwuzla,
        bv_sign: BitwuzlaTerm,
        bv_exponent: BitwuzlaTerm,
        bv_significand: BitwuzlaTerm
    ): BitwuzlaTerm

    /**
     * Create a floating-point value from its real representation, given as a<br></br>
     * decimal string, with respect to given rounding mode.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param sort The sort of the value.<br></br>
     * @param rm   The rounding mode.<br></br>
     * @param real The decimal string representing a real value.<br></br>
     * * @return A term of kind BITWUZLA_KIND_VAL, representing the floating-point<br></br>
     * value of given sort.<br></br>
     * * @see<br></br>
     * * `bitwuzla_mk_fp_sort`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_fp_value_from_real(Bitwuzla*, const BitwuzlaSort*, const BitwuzlaTerm*, const char*)`<br></br>
     * *native declaration : bitwuzla.h:2651*
     */
    external fun bitwuzla_mk_fp_value_from_real(
        bitwuzla: Bitwuzla,
        sort: BitwuzlaSort,
        rm: BitwuzlaTerm,
        real: String
    ): BitwuzlaTerm

    /**
     * Create a floating-point value from its rational representation, given as a<br></br>
     * two decimal strings representing the numerator and denominator, with respect<br></br>
     * to given rounding mode.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param sort The sort of the value.<br></br>
     * @param rm   The rounding mode.<br></br>
     * @param num  The decimal string representing the numerator.<br></br>
     * @param den  The decimal string representing the denominator.<br></br>
     * * @return A term of kind BITWUZLA_KIND_VAL, representing the floating-point<br></br>
     * value of given sort.<br></br>
     * * @see<br></br>
     * * `bitwuzla_mk_fp_sort`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_fp_value_from_rational(Bitwuzla*, const BitwuzlaSort*, const BitwuzlaTerm*, const char*, const char*)`<br></br>
     * *native declaration : bitwuzla.h:2673*
     */
    external fun bitwuzla_mk_fp_value_from_rational(
        bitwuzla: Bitwuzla,
        sort: BitwuzlaSort,
        rm: BitwuzlaTerm,
        num: String,
        den: String
    ): BitwuzlaTerm

    /**
     * Create a rounding mode value.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param rm The rounding mode value.<br></br>
     * * @return A term of kind BITWUZLA_KIND_VAL, representing the rounding mode<br></br>
     * value.<br></br>
     * * @see<br></br>
     * * `BitwuzlaRoundingMode`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_rm_value(Bitwuzla*, BitwuzlaRoundingMode)`<br></br>
     * *native declaration : bitwuzla.h:2691*
     */
    fun bitwuzla_mk_rm_value(bitwuzla: Bitwuzla, rm: BitwuzlaRoundingMode): BitwuzlaTerm =
        bitwuzla_mk_rm_value(bitwuzla, rm.value)

    external fun bitwuzla_mk_rm_value(bitwuzla: Bitwuzla, rm: Int): BitwuzlaTerm

    /**
     * Create a term of given kind with one argument term.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param kind The operator kind.<br></br>
     * @param arg  The argument to the operator.<br></br>
     * * @return A term representing an operation of given kind.<br></br>
     * * @see<br></br>
     * * `BitwuzlaKind`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_term1(Bitwuzla*, BitwuzlaKind, const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:2706*
     */
    fun bitwuzla_mk_term1(bitwuzla: Bitwuzla, kind: BitwuzlaKind, arg: BitwuzlaTerm): BitwuzlaTerm =
        bitwuzla_mk_term1(bitwuzla, kind.value, arg)

    external fun bitwuzla_mk_term1(bitwuzla: Bitwuzla, kind: Int, arg: BitwuzlaTerm): BitwuzlaTerm

    /**
     * Create a term of given kind with two argument terms.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param kind The operator kind.<br></br>
     * @param arg0 The first argument to the operator.<br></br>
     * @param arg1 The second argument to the operator.<br></br>
     * * @return A term representing an operation of given kind.<br></br>
     * * @see<br></br>
     * * `BitwuzlaKind`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_term2(Bitwuzla*, BitwuzlaKind, const BitwuzlaTerm*, const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:2723*
     */
    fun bitwuzla_mk_term2(
        bitwuzla: Bitwuzla,
        kind: BitwuzlaKind,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm
    ): BitwuzlaTerm = bitwuzla_mk_term2(bitwuzla, kind.value, arg0, arg1)

    external fun bitwuzla_mk_term2(
        bitwuzla: Bitwuzla,
        kind: Int,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm
    ): BitwuzlaTerm

    /**
     * Create a term of given kind with three argument terms.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param kind The operator kind.<br></br>
     * @param arg0 The first argument to the operator.<br></br>
     * @param arg1 The second argument to the operator.<br></br>
     * @param arg2 The third argument to the operator.<br></br>
     * * @return A term representing an operation of given kind.<br></br>
     * * @see<br></br>
     * * `BitwuzlaKind`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_term3(Bitwuzla*, BitwuzlaKind, const BitwuzlaTerm*, const BitwuzlaTerm*, const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:2742*
     */
    fun bitwuzla_mk_term3(
        bitwuzla: Bitwuzla,
        kind: BitwuzlaKind,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        arg2: BitwuzlaTerm
    ): BitwuzlaTerm = bitwuzla_mk_term3(bitwuzla, kind.value, arg0, arg1, arg2)

    external fun bitwuzla_mk_term3(
        bitwuzla: Bitwuzla,
        kind: Int,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        arg2: BitwuzlaTerm
    ): BitwuzlaTerm

    /**
     * Create a term of given kind with the given argument terms.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param kind The operator kind.<br></br>
     * @param args The argument terms.<br></br>
     * * @return A term representing an operation of given kind.<br></br>
     * * @see<br></br>
     * * `BitwuzlaKind`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_term(Bitwuzla*, BitwuzlaKind, uint32_t, const BitwuzlaTerm*[])`<br></br>
     * *native declaration : bitwuzla.h:2761*
     */
    external fun bitwuzla_mk_term(bitwuzla: Bitwuzla, kind: Int, argc: Int, args: Pointer): BitwuzlaTerm

    fun bitwuzla_mk_term(bitwuzla: Bitwuzla, kind: BitwuzlaKind, args: Array<BitwuzlaTerm>): BitwuzlaTerm =
        bitwuzla_mk_term(bitwuzla, kind.value, args.size, args.mkPtr())

    /**
     * Create an indexed term of given kind with one argument term and one index.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param kind The operator kind.<br></br>
     * @param arg  The argument term.<br></br>
     * @param idx  The index.<br></br>
     * * @return A term representing an indexed operation of given kind.<br></br>
     * * @see<br></br>
     * * `BitwuzlaKind`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_term1_indexed1(Bitwuzla*, BitwuzlaKind, const BitwuzlaTerm*, uint32_t)`<br></br>
     * *native declaration : bitwuzla.h:2779*
     */
    external fun bitwuzla_mk_term1_indexed1(bitwuzla: Bitwuzla, kind: Int, arg: BitwuzlaTerm, idx: Int): BitwuzlaTerm

    fun bitwuzla_mk_term1_indexed1(
        bitwuzla: Bitwuzla,
        kind: BitwuzlaKind,
        arg: BitwuzlaTerm,
        idx: Int
    ): BitwuzlaTerm = bitwuzla_mk_term1_indexed1(bitwuzla, kind.value, arg, idx)

    /**
     * Create an indexed term of given kind with one argument term and two indices.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param kind The operator kind.<br></br>
     * @param arg  The argument term.<br></br>
     * @param idx0 The first index.<br></br>
     * @param idx1 The second index.<br></br>
     * * @return A term representing an indexed operation of given kind.<br></br>
     * * @see<br></br>
     * * `BitwuzlaKind`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_term1_indexed2(Bitwuzla*, BitwuzlaKind, const BitwuzlaTerm*, uint32_t, uint32_t)`<br></br>
     * *native declaration : bitwuzla.h:2798*
     */
    fun bitwuzla_mk_term1_indexed2(
        bitwuzla: Bitwuzla,
        kind: BitwuzlaKind,
        arg: BitwuzlaTerm,
        idx0: Int,
        idx1: Int
    ): BitwuzlaTerm = bitwuzla_mk_term1_indexed2(bitwuzla, kind.value, arg, idx0, idx1)

    external fun bitwuzla_mk_term1_indexed2(
        bitwuzla: Bitwuzla,
        kind: Int,
        arg: BitwuzlaTerm,
        idx0: Int,
        idx1: Int
    ): BitwuzlaTerm

    /**
     * Create an indexed term of given kind with two argument terms and one index.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param kind The operator kind.<br></br>
     * @param arg0 The first argument term.<br></br>
     * @param arg1 The second argument term.<br></br>
     * @param idx  The index.<br></br>
     * * @return A term representing an indexed operation of given kind.<br></br>
     * * @see<br></br>
     * * `BitwuzlaKind`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_term2_indexed1(Bitwuzla*, BitwuzlaKind, const BitwuzlaTerm*, const BitwuzlaTerm*, uint32_t)`<br></br>
     * *native declaration : bitwuzla.h:2818*
     */
    fun bitwuzla_mk_term2_indexed1(
        bitwuzla: Bitwuzla,
        kind: BitwuzlaKind,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        idx: Int
    ): BitwuzlaTerm = bitwuzla_mk_term2_indexed1(bitwuzla, kind.value, arg0, arg1, idx)

    external fun bitwuzla_mk_term2_indexed1(
        bitwuzla: Bitwuzla,
        kind: Int,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        idx: Int
    ): BitwuzlaTerm

    /**
     * Create an indexed term of given kind with two argument terms and two indices.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param kind The operator kind.<br></br>
     * @param arg0 The first argument term.<br></br>
     * @param arg1 The second argument term.<br></br>
     * @param idx0 The first index.<br></br>
     * @param idx1 The second index.<br></br>
     * * @return A term representing an indexed operation of given kind.<br></br>
     * * @see<br></br>
     * * `BitwuzlaKind`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_term2_indexed2(Bitwuzla*, BitwuzlaKind, const BitwuzlaTerm*, const BitwuzlaTerm*, uint32_t, uint32_t)`<br></br>
     * *native declaration : bitwuzla.h:2839*
     */
    fun bitwuzla_mk_term2_indexed2(
        bitwuzla: Bitwuzla,
        kind: BitwuzlaKind,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        idx0: Int,
        idx1: Int
    ): BitwuzlaTerm = bitwuzla_mk_term2_indexed2(bitwuzla, kind.value, arg0, arg1, idx0, idx1)

    external fun bitwuzla_mk_term2_indexed2(
        bitwuzla: Bitwuzla,
        kind: Int,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        idx0: Int,
        idx1: Int
    ): BitwuzlaTerm

    /**
     * Create an indexed term of given kind with the given argument terms and<br></br>
     * indices.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param kind The operator kind.<br></br>
     * @param args The argument terms.<br></br>
     * @param idxs The indices.<br></br>
     * * @return A term representing an indexed operation of given kind.<br></br>
     * * @see<br></br>
     * * `BitwuzlaKind`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_term_indexed(Bitwuzla*, BitwuzlaKind, uint32_t, const BitwuzlaTerm*[], uint32_t, const uint32_t[])`<br></br>
     * *native declaration : bitwuzla.h:2862*
     */
    fun bitwuzla_mk_term_indexed(
        bitwuzla: Bitwuzla,
        kind: BitwuzlaKind,
        args: Array<BitwuzlaTerm>,
        idxs: IntArray
    ): BitwuzlaTerm = bitwuzla_mk_term_indexed(bitwuzla, kind.value, args.size, args.mkPtr(), idxs.size, idxs)

    external fun bitwuzla_mk_term_indexed(
        bitwuzla: Bitwuzla,
        kind: Int,
        argc: Int,
        args: Pointer,
        idxc: Int,
        idxs: IntArray
    ): BitwuzlaTerm

    /**
     * Create a (first-order) constant of given sort with given symbol.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param sort   The sort of the constant.<br></br>
     * @param symbol The symbol of the constant.<br></br>
     * * @return A term of kind BITWUZLA_KIND_CONST, representing the constant.<br></br>
     * * @see<br></br>
     * * `bitwuzla_mk_array_sort`<br></br>
     * * `bitwuzla_mk_bool_sort`<br></br>
     * * `bitwuzla_mk_bv_sort`<br></br>
     * * `bitwuzla_mk_fp_sort`<br></br>
     * * `bitwuzla_mk_fun_sort`<br></br>
     * * `bitwuzla_mk_rm_sort`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_const(Bitwuzla*, const BitwuzlaSort*, const char*)`<br></br>
     * *native declaration : bitwuzla.h:2886*
     */
    external fun bitwuzla_mk_const(bitwuzla: Bitwuzla, sort: BitwuzlaSort, symbol: String): BitwuzlaTerm

    /**
     * Create a one-dimensional constant array of given sort, initialized with<br></br>
     * given value.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param sort  The sort of the array.<br></br>
     * @param value The term to initialize the elements of the array with.<br></br>
     * * @return A term of kind BITWUZLA_KIND_CONST_ARRAY, representing a constant<br></br>
     * array of given sort.<br></br>
     * * @see<br></br>
     * * `bitwuzla_mk_array_sort`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_const_array(Bitwuzla*, const BitwuzlaSort*, const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:2904*
     */
    external fun bitwuzla_mk_const_array(bitwuzla: Bitwuzla, sort: BitwuzlaSort, value: BitwuzlaTerm): BitwuzlaTerm

    /**
     * Create a variable of given sort with given symbol.<br></br>
     * * @note This creates a variable to be bound by quantifiers or lambdas.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param sort   The sort of the variable.<br></br>
     * @param symbol The symbol of the variable.<br></br>
     * * @return A term of kind BITWUZLA_KIND_VAR, representing the variable.<br></br>
     * * @see<br></br>
     * * `bitwuzla_mk_bool_sort`<br></br>
     * * `bitwuzla_mk_bv_sort`<br></br>
     * * `bitwuzla_mk_fp_sort`<br></br>
     * * `bitwuzla_mk_fun_sort`<br></br>
     * * `bitwuzla_mk_rm_sort`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_mk_var(Bitwuzla*, const BitwuzlaSort*, const char*)`<br></br>
     * *native declaration : bitwuzla.h:2926*
     */
    external fun bitwuzla_mk_var(bitwuzla: Bitwuzla, sort: BitwuzlaSort, symbol: String): BitwuzlaTerm

    /**
     * Push context levels.<br></br>
     * * Requires that incremental solving has been enabled via<br></br>
     * `bitwuzla_set_option()`.<br></br>
     * * @note Assumptions added via this `bitwuzla_assume()` are not affected by<br></br>
     * context level changes and are only valid until the next<br></br>
     * `bitwuzla_check_sat()` call, no matter at which level they were<br></br>
     * assumed.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param nlevels The number of context levels to push.<br></br>
     * * @see<br></br>
     * * `bitwuzla_set_option`<br></br>
     * * `::BITWUZLA_OPT_INCREMENTAL`<br></br>
     * Original signature : `void bitwuzla_push(Bitwuzla*, uint32_t)`<br></br>
     * *native declaration : bitwuzla.h:2948*
     */
    external fun bitwuzla_push(bitwuzla: Bitwuzla, nlevels: Int)

    /**
     * Pop context levels.<br></br>
     * * Requires that incremental solving has been enabled via<br></br>
     * `bitwuzla_set_option()`.<br></br>
     * * @note Assumptions added via this `bitwuzla_assume()` are not affected by<br></br>
     * context level changes and are only valid until the next<br></br>
     * `bitwuzla_check_sat()` call, no matter at which level they were<br></br>
     * assumed.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param nlevels The number of context levels to pop.<br></br>
     * * @see<br></br>
     * * `bitwuzla_set_option`<br></br>
     * * `::BITWUZLA_OPT_INCREMENTAL`<br></br>
     * Original signature : `void bitwuzla_pop(Bitwuzla*, uint32_t)`<br></br>
     * *native declaration : bitwuzla.h:2968*
     */
    external fun bitwuzla_pop(bitwuzla: Bitwuzla, nlevels: Int)

    /**
     * Assert formula.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param term The formula to assert.<br></br>
     * Original signature : `void bitwuzla_assert(Bitwuzla*, const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:2976*
     */
    external fun bitwuzla_assert(bitwuzla: Bitwuzla, term: BitwuzlaTerm)

    /**
     * Assume formula.<br></br>
     * * Requires that incremental solving has been enabled via<br></br>
     * `bitwuzla_set_option()`.<br></br>
     * * @note Assumptions added via this function are not affected by context level<br></br>
     * changes and are only valid until the next `bitwuzla_check_sat()` call,<br></br>
     * no matter at which level they were assumed.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param term The formula to assume.<br></br>
     * * @see<br></br>
     * * `bitwuzla_set_option`<br></br>
     * * `bitwuzla_is_unsat_assumption`<br></br>
     * * `bitwuzla_get_unsat_assumptions`<br></br>
     * * `::BITWUZLA_OPT_INCREMENTAL`<br></br>
     * Original signature : `void bitwuzla_assume(Bitwuzla*, const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:2997*
     */
    external fun bitwuzla_assume(bitwuzla: Bitwuzla, term: BitwuzlaTerm)

    /**
     * Determine if an assumption is an unsat assumption.<br></br>
     * * Unsat assumptions are assumptions that force an input formula to become<br></br>
     * unsatisfiable. Unsat assumptions handling in Bitwuzla is analogous to<br></br>
     * failed assumptions in MiniSAT.<br></br>
     * * Requires that incremental solving has been enabled via<br></br>
     * `bitwuzla_set_option()`.<br></br>
     * * Requires that the last `bitwuzla_check_sat()` query returned<br></br>
     * `::BITWUZLA_UNSAT`.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param term The assumption to check for.<br></br>
     * * @return True if given assumption is an unsat assumption.<br></br>
     * * @see<br></br>
     * * `bitwuzla_set_option`<br></br>
     * * `bitwuzla_assume`<br></br>
     * * `bitwuzla_check_sat`<br></br>
     * * `::BITWUZLA_OPT_INCREMENTAL`<br></br>
     * Original signature : `bool bitwuzla_is_unsat_assumption(Bitwuzla*, const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3023*
     */
    external fun bitwuzla_is_unsat_assumption(bitwuzla: Bitwuzla, term: BitwuzlaTerm): Boolean

    /**
     * Get the set of unsat assumptions.<br></br>
     * * Unsat assumptions are assumptions that force an input formula to become<br></br>
     * unsatisfiable. Unsat assumptions handling in Bitwuzla is analogous to<br></br>
     * failed assumptions in MiniSAT.<br></br>
     * * Requires that incremental solving has been enabled via<br></br>
     * `bitwuzla_set_option()`.<br></br>
     * * Requires that the last `bitwuzla_check_sat()` query returned<br></br>
     * `::BITWUZLA_UNSAT`.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     * * @return An array with unsat assumptions of size `size`.<br></br>
     * * @see<br></br>
     * * `bitwuzla_set_option`<br></br>
     * * `bitwuzla_assume`<br></br>
     * * `bitwuzla_check_sat`<br></br>
     * * `::BITWUZLA_OPT_INCREMENTAL`<br></br>
     * Original signature : `BitwuzlaTerm** bitwuzla_get_unsat_assumptions(Bitwuzla*, size_t*)`<br></br>
     * *native declaration : bitwuzla.h:30jna49*
     */
    fun bitwuzla_get_unsat_assumptions(bitwuzla: Bitwuzla): Array<BitwuzlaTerm> {
        val size = IntByReference()
        val resultPtr = bitwuzla_get_unsat_assumptions(bitwuzla, size)
        val result = resultPtr.getPointerArray(0, size.value)
        return result as Array<BitwuzlaTerm>
    }

    external fun bitwuzla_get_unsat_assumptions(bitwuzla: Bitwuzla, size: IntByReference): Pointer

    /**
     * Get the set unsat core (unsat assertions).<br></br>
     * * The unsat core consists of the set of assertions that force an input formula<br></br>
     * to become unsatisfiable.<br></br>
     * * Requires that the last `bitwuzla_check_sat()` query returned<br></br>
     * `::BITWUZLA_UNSAT`.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     *
     * * @return An array with unsat assertions of size `size`.<br></br>
     * * @see<br></br>
     * * `bitwuzla_assert`<br></br>
     * * `bitwuzla_check_sat`<br></br>
     * Original signature : `BitwuzlaTerm** bitwuzla_get_unsat_core(Bitwuzla*, size_t*)`<br></br>
     * *native declaration : bitwuzla.h:3070*
     */
    fun bitwuzla_get_unsat_core(bitwuzla: Bitwuzla): Array<BitwuzlaTerm> {
        val size = IntByReference()
        val resultPtr = bitwuzla_get_unsat_core(bitwuzla, size)
        val result = resultPtr.getPointerArray(0, size.value)
        return result as Array<BitwuzlaTerm>
    }

    external fun bitwuzla_get_unsat_core(bitwuzla: Bitwuzla, size: IntByReference): Pointer

    /**
     * Assert all added assumptions.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     * * @see<br></br>
     * * `bitwuzla_assume`<br></br>
     * Original signature : `void bitwuzla_fixate_assumptions(Bitwuzla*)`<br></br>
     * *native declaration : bitwuzla.h:3080*
     */
    external fun bitwuzla_fixate_assumptions(bitwuzla: Bitwuzla)

    /**
     * Reset all added assumptions.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     * * @see<br></br>
     * * `bitwuzla_assume`<br></br>
     * Original signature : `void bitwuzla_reset_assumptions(Bitwuzla*)`<br></br>
     * *native declaration : bitwuzla.h:3090*
     */
    external fun bitwuzla_reset_assumptions(bitwuzla: Bitwuzla)

    /**
     * Simplify the current input formula.<br></br>
     * * @note Assumptions are not considered for simplification.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     * * @return `::BITWUZLA_SAT` if the input formula was simplified to true,<br></br>
     * `::BITWUZLA_UNSAT` if it was simplified to false, and<br></br>
     * `::BITWUZLA_UNKNOWN` otherwise.<br></br>
     * * @see<br></br>
     * * `bitwuzla_assert`<br></br>
     * * `BitwuzlaResult`<br></br>
     * Original signature : `BitwuzlaResult bitwuzla_simplify(Bitwuzla*)`<br></br>
     * *native declaration : bitwuzla.h:3107*
     */
    fun bitwuzla_simplify_helper(bitwuzla: Bitwuzla): BitwuzlaResult =
        BitwuzlaResult.fromValue(bitwuzla_simplify(bitwuzla))

    external fun bitwuzla_simplify(bitwuzla: Bitwuzla): Int

    /**
     * Check satisfiability of current input formula.<br></br>
     * * An input formula consists of assertions added via `bitwuzla_assert()`.<br></br>
     * The search for a solution can by guided by making assumptions via<br></br>
     * `bitwuzla_assume()`.<br></br>
     * * @note Assertions and assumptions are combined via Boolean and.  Multiple<br></br>
     * calls to this function require enabling incremental solving via<br></br>
     * `bitwuzla_set_option()`.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     * * @return `::BITWUZLA_SAT` if the input formula is satisfiable and<br></br>
     * `::BITWUZLA_UNSAT` if it is unsatisfiable, and `::BITWUZLA_UNKNOWN`<br></br>
     * when neither satisfiability nor unsatisfiability was determined.<br></br>
     * This can happen when `bitwuzla` was terminated via a termination<br></br>
     * callback.<br></br>
     * * @see<br></br>
     * * `bitwuzla_assert`<br></br>
     * * `bitwuzla_assume`<br></br>
     * * `bitwuzla_set_option`<br></br>
     * * `::BITWUZLA_OPT_INCREMENTAL`<br></br>
     * * `BitwuzlaResult`<br></br>
     * Original signature : `BitwuzlaResult bitwuzla_check_sat(Bitwuzla*)`<br></br>
     * *native declaration : bitwuzla.h:3135*
     */
    fun bitwuzla_check_sat_helper(bitwuzla: Bitwuzla): BitwuzlaResult =
        BitwuzlaResult.fromValue(bitwuzla_check_sat(bitwuzla))

    external fun bitwuzla_check_sat(bitwuzla: Bitwuzla): Int

    /**
     * Get a term representing the model value of a given term.<br></br>
     * * Requires that the last `bitwuzla_check_sat()` query returned<br></br>
     * `::BITWUZLA_SAT`.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param term The term to query a model value for.<br></br>
     * * @return A term representing the model value of term `term`.<br></br>
     * * @see `bitwuzla_check_sat`<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_get_value(Bitwuzla*, const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3150*
     */
    external fun bitwuzla_get_value(bitwuzla: Bitwuzla, term: BitwuzlaTerm): BitwuzlaTerm

    /**
     * Get string representation of the current model value of given bit-vector<br></br>
     * term.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param term The term to query a model value for.<br></br>
     * * @return Binary string representation of current model value of term \p term.<br></br>
     * Return value is valid until next `bitwuzla_get_bv_value` call.<br></br>
     * Original signature : `char* bitwuzla_get_bv_value(Bitwuzla*, const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3163*
     */
    external fun bitwuzla_get_bv_value(bitwuzla: Bitwuzla, term: BitwuzlaTerm): String

    /**
     * Get string of IEEE 754 standard representation of the current model value of<br></br>
     * given floating-point term.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param term The term to query a model value for.<br></br>
     * sign        Binary string representation of the sign bit.<br></br>
     * exponent    Binary string representation of the exponent bit-vector<br></br>
     * value.<br></br>
     * significand Binary string representation of the significand<br></br>
     * bit-vector value.<br></br>
     * Original signature : `void bitwuzla_get_fp_value(Bitwuzla*, const BitwuzlaTerm*, const char**, const char**, const char**)`<br></br>
     * *native declaration : bitwuzla.h:3177*
     */
    fun bitwuzla_get_fp_value(bitwuzla: Bitwuzla, term: BitwuzlaTerm): FpValue {
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

    external fun bitwuzla_get_fp_value(
        bitwuzla: Bitwuzla,
        term: BitwuzlaTerm,
        sign: PointerByReference,
        exponent: PointerByReference,
        significand: PointerByReference
    )

    /**
     * Get string representation of the current model value of given rounding mode<br></br>
     * term.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param term The rounding mode term to query a model value for.<br></br>
     * * @return String representation of rounding mode (RNA, RNE, RTN, RTP, RTZ).<br></br>
     * Original signature : `char* bitwuzla_get_rm_value(Bitwuzla*, const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3192*
     */
    external fun bitwuzla_get_rm_value(bitwuzla: Bitwuzla, term: BitwuzlaTerm): String

    /**
     * Get the current model value of given array term.<br></br>
     * * The string representation of `indices` and `values` can be queried via<br></br>
     * `bitwuzla_get_bv_value()`, `bitwuzla_get_fp_value()`, and<br></br>
     * `bitwuzla_get_rm_value()`.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param term The term to query a model value for.<br></br>
     * indices       List of indices of size `size`. 1:1 mapping to `values`,<br></br>
     * i.e., `index[i] -> value[i]`.<br></br>
     * values        List of values of size `size`.<br></br>
     * size          Size of `indices` and `values` list.<br></br>
     * default_value The value of all other indices not in `indices` and<br></br>
     * is set when base array is a constant array.<br></br>
     * Original signature : `void bitwuzla_get_array_value(Bitwuzla*, const BitwuzlaTerm*, const BitwuzlaTerm***, const BitwuzlaTerm***, size_t*, const BitwuzlaTerm**)`<br></br>
     * *native declaration : bitwuzla.h:3210*
     */
    fun bitwuzla_get_array_value(bitwuzla: Bitwuzla, term: BitwuzlaTerm): ArrayValue {
        val size = IntByReference()
        val indices = PointerByReference()
        val values = PointerByReference()
        val defaultValue = PointerByReference()
        bitwuzla_get_array_value(bitwuzla, term, indices, values, size, defaultValue)
        val sz = size.value
        return ArrayValue(
            sz,
            indices.pointer.getPointerArray(0, sz) as Array<BitwuzlaTerm>,
            values.pointer.getPointerArray(0, sz) as Array<BitwuzlaTerm>,
            defaultValue.value as BitwuzlaTerm
        )
    }

    external fun bitwuzla_get_array_value(
        bitwuzla: Bitwuzla,
        term: BitwuzlaTerm,
        indices: PointerByReference,
        values: PointerByReference,
        size: IntByReference,
        default_value: PointerByReference
    )

    /**
     * Get the current model value of given function term.<br></br>
     * * The string representation of `args` and `values` can be queried via<br></br>
     * `bitwuzla_get_bv_value()`, `bitwuzla_get_fp_value()`, and<br></br>
     * `bitwuzla_get_rm_value()`.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param term The term to query a model value for.<br></br>
     * *@param args   List of argument lists (nested lists) of size `size`. Each<br></br>
     * argument list is of size `arity`.<br></br>
     * *@param arity  Size of each argument list in `args`.<br></br>
     * *@param values List of values of size `size`.<br></br>
     * *@param size   Size of `indices` and `values` list.<br></br>
     * * **Usage**<br></br>
     * ```<br></br>
     * size_t arity, size;<br></br>
     * BitwuzlaTerm ***args, **values;<br></br>
     * bitwuzla_get_fun_value(bzla, f, &args, &arity, &values, &size);<br></br>
     * * for (size_t i = 0; i < size; ++i)<br></br>
     * {<br></br>
     * // args[i] are argument lists of size arity<br></br>
     * for (size_t j = 0; j < arity; ++j)<br></br>
     * {<br></br>
     * // args[i][j] corresponds to value of jth argument of function f<br></br>
     * }<br></br>
     * // values[i] corresponds to the value of<br></br>
     * // (f args[i][0] ... args[i][arity - 1])<br></br>
     * }<br></br>
     * ```<br></br>
     * Original signature : `void bitwuzla_get_fun_value(Bitwuzla*, const BitwuzlaTerm*, const BitwuzlaTerm****, size_t*, const BitwuzlaTerm***, size_t*)`<br></br>
     * *native declaration : bitwuzla.h:3251*
     */
    fun bitwuzla_get_fun_value(bitwuzla: Bitwuzla, term: BitwuzlaTerm): FunValue {
        val arityPtr = IntByReference()
        val sizePtr = IntByReference()
        val argsPtr = PointerByReference()
        val valuesPtr = PointerByReference()
        bitwuzla_get_fun_value(bitwuzla, term, argsPtr, arityPtr, valuesPtr, sizePtr)
        val size = sizePtr.value
        val arity = arityPtr.value
        val argsPtrList = argsPtr.value.getPointerArray(0, size)
        val args = Array(size) { argsPtrList[it].getPointerArray(0, arity) as Array<BitwuzlaTerm> }
        return FunValue(size, arity, args, valuesPtr.value.getPointerArray(0, size) as Array<BitwuzlaTerm>)
    }

    external fun bitwuzla_get_fun_value(
        bitwuzla: Bitwuzla,
        term: BitwuzlaTerm,
        args: PointerByReference,
        arity: IntByReference,
        values: PointerByReference,
        size: IntByReference
    )

    /**
     * Print a model for the current input formula.<br></br>
     * * Requires that the last `bitwuzla_check_sat()` query returned<br></br>
     * `::BITWUZLA_SAT`.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param format The output format for printing the model. Either `"btor"` for<br></br>
     * the BTOR format, or `"smt2"` for the SMT-LIB v2 format.<br></br>
     * @param file   The file to print the model to.<br></br>
     * * @see<br></br>
     * * `bitwuzla_check_sat`<br></br>
     * Original signature : `void bitwuzla_print_model(Bitwuzla*, const char*, FILE*)`<br></br>
     * *native declaration : bitwuzla.h:3272*
     */
    fun bitwuzla_print_model(bitwuzla: Bitwuzla, format: String, file: File) {
        throw UnsupportedOperationException("operation is not implemented")
    }

    /**
     * Print the current input formula.<br></br>
     * * Requires that incremental solving is not enabled.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param format The output format for printing the formula. Either<br></br>
     * `"aiger_ascii"` for the AIGER ascii format, `"aiger_binary"`<br></br>
     * for the binary AIGER format, `"btor"` for the BTOR format, or<br></br>
     * `"smt2"` for the SMT-LIB v2 format.<br></br>
     * @param file   The file to print the formula to.<br></br>
     * Original signature : `void bitwuzla_dump_formula(Bitwuzla*, const char*, FILE*)`<br></br>
     * *native declaration : bitwuzla.h:3286*
     */
    fun bitwuzla_dump_formula(bitwuzla: Bitwuzla, format: String, file: File) {
        throw UnsupportedOperationException("operation is not implemented")
    }

    /**
     * Parse input file.<br></br>
     * * The format of the input file is auto detected.  <br></br>
     * Requires that no terms have been created yet.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param infile        The input file.<br></br>
     * @param infile_name   The name of the input file.<br></br>
     * @param outfile       The output file.<br></br>
     * @param error_msg     Output parameter, stores an error message in case a parse<br></br>
     * error occurred, else \c NULL.<br></br>
     * @param parsed_status Output parameter, stores the status of the input in case<br></br>
     * of SMT-LIB v2 input, if given.<br></br>
     * @param parsed_smt2   Output parameter, true if parsed input file has been<br></br>
     * detected as SMT-LIB v2 input.<br></br>
     * * @return `::BITWUZLA_SAT` if the input formula was simplified to true,<br></br>
     * `::BITWUZLA_UNSAT` if it was simplified to false,<br></br>
     * and `::BITWUZLA_UNKNOWN` otherwise.<br></br>
     * * @see<br></br>
     * * `bitwuzla_parse_format`<br></br>
     * Original signature : `BitwuzlaResult bitwuzla_parse(Bitwuzla*, FILE*, const char*, FILE*, char**, BitwuzlaResult*, bool*)`<br></br>
     * *native declaration : bitwuzla.h:3312*
     */
    fun bitwuzla_parse(
        bitwuzla: Bitwuzla,
        infile: File,
        infile_name: String,
        outfile: File,
        error_msg: Array<String>,
        parsed_status: Array<BitwuzlaResult>,
        parsed_smt2: BooleanArray
    ): BitwuzlaResult {
        throw UnsupportedOperationException("operation is not implemented")
    }

    /**
     * Parse input file, assumed to be given in the specified format.<br></br>
     * * Requires that no terms have been created yet.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param format        The input format for printing the model. Either `"btor"` for<br></br>
     * the BTOR format, `"btor2"` for the BTOR2 format, or `"smt2"`<br></br>
     * for the SMT-LIB v2 format.<br></br>
     * @param infile        The input file.<br></br>
     * @param infile_name   The name of the input file.<br></br>
     * @param outfile       The output file.<br></br>
     * @param error_msg     Output parameter, stores an error message in case a parse<br></br>
     * error occurred, else \c NULL.<br></br>
     * @param parsed_status Output parameter, stores the status of the input in case<br></br>
     * of SMT-LIB v2 input, if given.<br></br>
     * * @return `::BITWUZLA_SAT` if the input formula was simplified to true,<br></br>
     * `::BITWUZLA_UNSAT` if it was simplified to false,<br></br>
     * and ::BITWUZLA_UNKNOWN` otherwise.<br></br>
     * * @see<br></br>
     * * `bitwuzla_parse`<br></br>
     * Original signature : `BitwuzlaResult bitwuzla_parse_format(Bitwuzla*, const char*, FILE*, const char*, FILE*, char**, BitwuzlaResult*)`<br></br>
     * *native declaration : bitwuzla.h:3344*
     */
    fun bitwuzla_parse_format(
        bitwuzla: Bitwuzla,
        format: String,
        infile: File,
        infile_name: String,
        outfile: File,
        error_msg: Array<String>,
        parsed_status: Array<BitwuzlaResult>
    ): BitwuzlaResult {
        throw UnsupportedOperationException("operation is not implemented")
    }

    /**
     * Substitute a set of keys with their corresponding values in the given term.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param term       The term in which the keys are to be substituted.<br></br>
     * @param map_size   The size of the substitution map.<br></br>
     * @param map_keys   The keys.<br></br>
     * @param map_values The mapped values.<br></br>
     * * @return The resulting term from this substitution.<br></br>
     * Original signature : `BitwuzlaTerm* bitwuzla_substitute_term(Bitwuzla*, const BitwuzlaTerm*, size_t, const BitwuzlaTerm*[], const BitwuzlaTerm*[])`<br></br>
     * *native declaration : bitwuzla.h:3363*
     */

    external fun bitwuzla_substitute_term(
        bitwuzla: Bitwuzla,
        term: BitwuzlaTerm,
        map_size: Int,
        map_keys: Pointer,
        map_values: Pointer
    ): BitwuzlaTerm

    fun bitwuzla_substitute_term(
        bitwuzla: Bitwuzla,
        term: BitwuzlaTerm,
        map_keys: Array<BitwuzlaTerm>,
        map_values: Array<BitwuzlaTerm>
    ): BitwuzlaTerm = bitwuzla_substitute_term(bitwuzla, term, map_keys.size, map_keys.mkPtr(), map_values.mkPtr())

    /**
     * Substitute a set of keys with their corresponding values in the set of given<br></br>
     * terms.<br></br>
     * * The terms in `terms` are replaced with the terms resulting from this<br></br>
     * substitutions.<br></br>
     * * @param bitwuzla The Bitwuzla instance.<br></br>
     *
     * @param terms_size The size of the set of terms.<br></br>
     * @param terms      The terms in which the keys are to be substituted.<br></br>
     * @param map_size   The size of the substitution map.<br></br>
     * @param map_keys   The keys.<br></br>
     * @param map_values The mapped values.<br></br>
     * Original signature : `void bitwuzla_substitute_terms(Bitwuzla*, size_t, const BitwuzlaTerm*[], size_t, const BitwuzlaTerm*[], const BitwuzlaTerm*[])`<br></br>
     * *native declaration : bitwuzla.h:3383*
     */
    external fun bitwuzla_substitute_terms(
        bitwuzla: Bitwuzla,
        terms_size: Int,
        terms: Pointer,
        map_size: Int,
        map_keys: Pointer,
        map_values: Pointer
    )

    fun bitwuzla_substitute_terms(
        bitwuzla: Bitwuzla,
        terms: Array<BitwuzlaTerm>,
        map_keys: Array<BitwuzlaTerm>,
        map_values: Array<BitwuzlaTerm>
    ) {
        val termsPtr = terms.mkPtr()
        bitwuzla_substitute_terms(bitwuzla, terms.size, termsPtr, map_keys.size, map_keys.mkPtr(), map_values.mkPtr())
        val result = termsPtr.getPointerArray(0, terms.size)
        for (i in terms.indices) {
            terms[i] = result[i]
        }
    }

    /**
     * Compute the hash value for a sort.<br></br>
     * * @param sort The sort.<br></br>
     * * @return The hash value of the sort.<br></br>
     * Original signature : `size_t bitwuzla_sort_hash(const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:3401*
     */
    external fun bitwuzla_sort_hash(sort: BitwuzlaSort): Long

    /**
     * Get the size of a bit-vector sort.<br></br>
     * * Requires that given sort is a bit-vector sort.<br></br>
     * * @param sort The sort.<br></br>
     * * @return The size of the bit-vector sort.<br></br>
     * Original signature : `uint32_t bitwuzla_sort_bv_get_size(const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:3412*
     */
    external fun bitwuzla_sort_bv_get_size(sort: BitwuzlaSort): Int

    /**
     * Get the exponent size of a floating-point sort.<br></br>
     * * Requires that given sort is a floating-point sort.<br></br>
     * * @param sort The sort.<br></br>
     * * @return The exponent size of the floating-point sort.<br></br>
     * Original signature : `uint32_t bitwuzla_sort_fp_get_exp_size(const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:3423*
     */
    external fun bitwuzla_sort_fp_get_exp_size(sort: BitwuzlaSort): Int

    /**
     * Get the significand size of a floating-point sort.<br></br>
     * * Requires that given sort is a floating-point sort.<br></br>
     * * @param sort The sort.<br></br>
     * * @return The significand size of the floating-point sort.<br></br>
     * Original signature : `uint32_t bitwuzla_sort_fp_get_sig_size(const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:3434*
     */
    external fun bitwuzla_sort_fp_get_sig_size(sort: BitwuzlaSort): Int

    /**
     * Get the index sort of an array sort.<br></br>
     * * Requires that given sort is an array sort.<br></br>
     * * @param sort The sort.<br></br>
     * * @return The index sort of the array sort.<br></br>
     * Original signature : `BitwuzlaSort* bitwuzla_sort_array_get_index(const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:3445*
     */
    external fun bitwuzla_sort_array_get_index(sort: BitwuzlaSort): BitwuzlaSort

    /**
     * Get the element sort of an array sort.<br></br>
     * * Requires that given sort is an array sort.<br></br>
     * * @param sort The sort.<br></br>
     * * @return The element sort of the array sort.<br></br>
     * Original signature : `BitwuzlaSort* bitwuzla_sort_array_get_element(const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:3456*
     */
    external fun bitwuzla_sort_array_get_element(sort: BitwuzlaSort): BitwuzlaSort

    /**
     * Get the domain sorts of a function sort.<br></br>
     * * The domain sorts are returned as an array of sorts of size `size`.<br></br>
     * Requires that given sort is a function sort.<br></br>
     * * @param sort The sort.<br></br>
     *
     *
     * * @return The domain sorts of the function sort.<br></br>
     * Original signature : `BitwuzlaSort** bitwuzla_sort_fun_get_domain_sorts(const BitwuzlaSort*, size_t*)`<br></br>
     * *native declaration : bitwuzla.h:3469*
     */
    fun bitwuzla_sort_fun_get_domain_sorts(sort: BitwuzlaSort): Array<BitwuzlaSort> {
        val size = IntByReference()
        val result = bitwuzla_sort_fun_get_domain_sorts(sort, size)
        return result.getPointerArray(0, size.value) as Array<BitwuzlaSort>
    }

    external fun bitwuzla_sort_fun_get_domain_sorts(sort: BitwuzlaSort, size: IntByReference): Pointer

    /**
     * Get the codomain sort of a function sort.<br></br>
     * * Requires that given sort is a function sort.<br></br>
     * * @param sort The sort.<br></br>
     * * @return The codomain sort of the function sort.<br></br>
     * Original signature : `BitwuzlaSort* bitwuzla_sort_fun_get_codomain(const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:3481*
     */
    external fun bitwuzla_sort_fun_get_codomain(sort: BitwuzlaSort): BitwuzlaSort

    /**
     * Get the arity of a function sort.<br></br>
     * * @param sort The sort.<br></br>
     * * @return The number of arguments of the function sort.<br></br>
     * Original signature : `uint32_t bitwuzla_sort_fun_get_arity(const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:3490*
     */
    external fun bitwuzla_sort_fun_get_arity(sort: BitwuzlaSort): Int

    /**
     * Determine if two sorts are equal.<br></br>
     * * @param sort0 The first sort.<br></br>
     *
     * @param sort1 The second sort.<br></br>
     * * @return True if the given sorts are equal.<br></br>
     * Original signature : `bool bitwuzla_sort_is_equal(const BitwuzlaSort*, const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:3500*
     */
    external fun bitwuzla_sort_is_equal(sort0: BitwuzlaSort, sort1: BitwuzlaSort): Boolean

    /**
     * Determine if a sort is an array sort.<br></br>
     * * @param sort The sort.<br></br>
     * * @return True if `sort` is an array sort.<br></br>
     * Original signature : `bool bitwuzla_sort_is_array(const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:3510*
     */
    external fun bitwuzla_sort_is_array(sort: BitwuzlaSort): Boolean

    /**
     * Determine if a sort is a bit-vector sort.<br></br>
     * * @param sort The sort.<br></br>
     * * @return True if `sort` is a bit-vector sort.<br></br>
     * Original signature : `bool bitwuzla_sort_is_bv(const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:3519*
     */
    external fun bitwuzla_sort_is_bv(sort: BitwuzlaSort): Boolean

    /**
     * Determine if a sort is a floating-point sort.<br></br>
     * * @param sort The sort.<br></br>
     * * @return True if `sort` is a floating-point sort.<br></br>
     * Original signature : `bool bitwuzla_sort_is_fp(const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:3528*
     */
    external fun bitwuzla_sort_is_fp(sort: BitwuzlaSort): Boolean

    /**
     * Determine if a sort is a function sort.<br></br>
     * * @param sort The sort.<br></br>
     * * @return True if `sort` is a function sort.<br></br>
     * Original signature : `bool bitwuzla_sort_is_fun(const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:3537*
     */
    external fun bitwuzla_sort_is_fun(sort: BitwuzlaSort): Boolean

    /**
     * Determine if a sort is a Roundingmode sort.<br></br>
     * * @param sort The sort.<br></br>
     * * @return True if `sort` is a Roundingmode sort.<br></br>
     * Original signature : `bool bitwuzla_sort_is_rm(const BitwuzlaSort*)`<br></br>
     * *native declaration : bitwuzla.h:3546*
     */
    external fun bitwuzla_sort_is_rm(sort: BitwuzlaSort): Boolean

    /**
     * Print sort.<br></br>
     * * @param sort The sort.<br></br>
     *
     * @param format The output format for printing the term. Either `"btor"` for<br></br>
     * the BTOR format, or `"smt2"` for the SMT-LIB v2 format. Note<br></br>
     * for the `"btor"` this function won't do anything since BTOR<br></br>
     * sorts are printed when printing the term via<br></br>
     * bitwuzla_term_dump.<br></br>
     * @param file   The file to print the term to.<br></br>
     * Original signature : `void bitwuzla_sort_dump(const BitwuzlaSort*, const char*, FILE*)`<br></br>
     * *native declaration : bitwuzla.h:3559*
     */
    fun bitwuzla_sort_dump(sort: BitwuzlaSort, format: String, file: File) {
        throw UnsupportedOperationException("operation is not implemented")
    }

    /**
     * Compute the hash value for a term.<br></br>
     * * @param term The term.<br></br>
     * * @return The hash value of the term.<br></br>
     * Original signature : `size_t bitwuzla_term_hash(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3574*
     */
    external fun bitwuzla_term_hash(term: BitwuzlaTerm): Long

    /**
     * Get the kind of a term.<br></br>
     * * @param term The term.<br></br>
     * * @return The kind of the given term.<br></br>
     * * @see BitwuzlaKind<br></br>
     * Original signature : `BitwuzlaKind bitwuzla_term_get_kind(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3585*
     */
    fun bitwuzla_term_get_kind_helper(term: BitwuzlaTerm): BitwuzlaKind =
        BitwuzlaKind.fromValue(bitwuzla_term_get_kind(term))

    external fun bitwuzla_term_get_kind(term: BitwuzlaTerm): Int

    /**
     * Get the child terms of a term.<br></br>
     * * Returns \c NULL if given term does not have children.<br></br>
     * * @param term The term.<br></br>
     *
     *
     * * @return The children of `term` as an array of terms.<br></br>
     * Original signature : `BitwuzlaTerm** bitwuzla_term_get_children(const BitwuzlaTerm*, size_t*)`<br></br>
     * *native declaration : bitwuzla.h:3597*
     */
    fun bitwuzla_term_get_children(term: BitwuzlaTerm): Array<BitwuzlaTerm> {
        val size = IntByReference()
        val result = bitwuzla_term_get_children(term, size)
        return result.getPointerArray(0, size.value) as Array<BitwuzlaTerm>
    }

    external fun bitwuzla_term_get_children(term: BitwuzlaTerm, size: IntByReference): Pointer

    /**
     * Get the indices of an indexed term.<br></br>
     * * Requires that given term is an indexed term.<br></br>
     * * @param term The term.<br></br>
     *
     *
     * * @return The children of `term` as an array of terms.<br></br>
     * Original signature : `uint32_t* bitwuzla_term_get_indices(const BitwuzlaTerm*, size_t*)`<br></br>
     * *native declaration : bitwuzla.h:3610*
     */
    fun bitwuzla_term_get_indices(term: BitwuzlaTerm): IntArray {
        val size = IntByReference()
        val result = bitwuzla_term_get_indices(term, size)
        return result.getIntArray(0, size.value)
    }

    external fun bitwuzla_term_get_indices(term: BitwuzlaTerm, size: IntByReference): Pointer

    /**
     * Determine if a term is an indexed term.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is an indexed term.<br></br>
     * Original signature : `bool bitwuzla_term_is_indexed(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3619*
     */
    external fun bitwuzla_term_is_indexed(term: BitwuzlaTerm): Boolean

    /**
     * Get the associated Bitwuzla instance of a term.<br></br>
     * * @param term The term.<br></br>
     * * @return The associated Bitwuzla instance.<br></br>
     * Original signature : `Bitwuzla* bitwuzla_term_get_bitwuzla(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3628*
     */
    external fun bitwuzla_term_get_bitwuzla(term: BitwuzlaTerm): Bitwuzla

    /**
     * Get the sort of a term.<br></br>
     * * @param term The term.<br></br>
     * * @return The sort of the term.<br></br>
     * Original signature : `BitwuzlaSort* bitwuzla_term_get_sort(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3637*
     */
    external fun bitwuzla_term_get_sort(term: BitwuzlaTerm): BitwuzlaSort

    /**
     * Get the index sort of an array term.<br></br>
     * * Requires that given term is an array or an array store term.<br></br>
     * * @param term The term.<br></br>
     * * @return The index sort of the array term.<br></br>
     * Original signature : `BitwuzlaSort* bitwuzla_term_array_get_index_sort(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3648*
     */
    external fun bitwuzla_term_array_get_index_sort(term: BitwuzlaTerm): BitwuzlaSort

    /**
     * Get the element sort of an array term.<br></br>
     * * Requires that given term is an array or an array store term.<br></br>
     * * @param term The term.<br></br>
     * * @return The element sort of the array term.<br></br>
     * Original signature : `BitwuzlaSort* bitwuzla_term_array_get_element_sort(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3660*
     */
    external fun bitwuzla_term_array_get_element_sort(term: BitwuzlaTerm): BitwuzlaSort

    /**
     * Get the domain sorts of a function term.<br></br>
     * * The domain sorts are returned as an array of sorts of size `size.<br></br>
     * Requires that given term is an uninterpreted function, a lambda term, an<br></br>
     * array store term, or an ite term over function terms.<br></br>
     * * @param term The term.<br></br>
     *
     *
     * * @return The domain sorts of the function term.<br></br>
     * Original signature : `BitwuzlaSort** bitwuzla_term_fun_get_domain_sorts(const BitwuzlaTerm*, size_t*)`<br></br>
     * *native declaration : bitwuzla.h:3675*
     */
    fun bitwuzla_term_fun_get_domain_sorts(term: BitwuzlaTerm): Array<BitwuzlaSort> {
        val size = IntByReference()
        val result = bitwuzla_term_fun_get_domain_sorts(term, size)
        return result.getPointerArray(0, size.value) as Array<BitwuzlaSort>
    }

    external fun bitwuzla_term_fun_get_domain_sorts(term: BitwuzlaTerm, size: IntByReference): Pointer

    /**
     * Get the codomain sort of a function term.<br></br>
     * * Requires that given term is an uninterpreted function, a lambda term, an<br></br>
     * array store term, or an ite term over function terms.<br></br>
     * * @param term The term.<br></br>
     * * @return The codomain sort of the function term.<br></br>
     * Original signature : `BitwuzlaSort* bitwuzla_term_fun_get_codomain_sort(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3688*
     */
    external fun bitwuzla_term_fun_get_codomain_sort(term: BitwuzlaTerm): BitwuzlaSort

    /**
     * Get the bit-width of a bit-vector term.<br></br>
     * * Requires that given term is a bit-vector term.<br></br>
     * * @param term The term.<br></br>
     * * @return The bit-width of the bit-vector term.<br></br>
     * Original signature : `uint32_t bitwuzla_term_bv_get_size(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3700*
     */
    external fun bitwuzla_term_bv_get_size(term: BitwuzlaTerm): Int

    /**
     * Get the bit-width of the exponent of a floating-point term.<br></br>
     * * Requires that given term is a floating-point term.<br></br>
     * * @param term The term.<br></br>
     * * @return The bit-width of the exponent of the floating-point term.<br></br>
     * Original signature : `uint32_t bitwuzla_term_fp_get_exp_size(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3711*
     */
    external fun bitwuzla_term_fp_get_exp_size(term: BitwuzlaTerm): Int

    /**
     * Get the bit-width of the significand of a floating-point term.<br></br>
     * * Requires that given term is a floating-point term.<br></br>
     * * @param term The term.<br></br>
     * * @return The bit-width of the significand of the floating-point term.<br></br>
     * Original signature : `uint32_t bitwuzla_term_fp_get_sig_size(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3722*
     */
    external fun bitwuzla_term_fp_get_sig_size(term: BitwuzlaTerm): Int

    /**
     * Get the arity of a function term.<br></br>
     * * Requires that given term is a function term.<br></br>
     * * @param term The term.<br></br>
     * * @return The arity of the function term.<br></br>
     * Original signature : `uint32_t bitwuzla_term_fun_get_arity(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3733*
     */
    external fun bitwuzla_term_fun_get_arity(term: BitwuzlaTerm): Int

    /**
     * Get the symbol of a term.<br></br>
     * * @param term The term.<br></br>
     * * @return The symbol of `term`. \c NULL if no symbol is defined.<br></br>
     * Original signature : `char* bitwuzla_term_get_symbol(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3742*
     */
    external fun bitwuzla_term_get_symbol(term: BitwuzlaTerm): String

    /**
     * Set the symbol of a term.<br></br>
     * * @param term The term.<br></br>
     *
     * @param symbol The symbol.<br></br>
     * Original signature : `void bitwuzla_term_set_symbol(const BitwuzlaTerm*, const char*)`<br></br>
     * *native declaration : bitwuzla.h:3750*
     */
    external fun bitwuzla_term_set_symbol(term: BitwuzlaTerm, symbol: String)

    /**
     * Determine if the sorts of two terms are equal.<br></br>
     * * @param term0 The first term.<br></br>
     *
     * @param term1 The second term.<br></br>
     * * @return True if the sorts of `term0` and `term1` are equal.<br></br>
     * Original signature : `bool bitwuzla_term_is_equal_sort(const BitwuzlaTerm*, const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3760*
     */
    external fun bitwuzla_term_is_equal_sort(term0: BitwuzlaTerm, term1: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is an array term.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is an array term.<br></br>
     * Original signature : `bool bitwuzla_term_is_array(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3770*
     */
    external fun bitwuzla_term_is_array(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a constant.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a constant.<br></br>
     * Original signature : `bool bitwuzla_term_is_const(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3779*
     */
    external fun bitwuzla_term_is_const(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a function.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a function.<br></br>
     * Original signature : `bool bitwuzla_term_is_fun(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3788*
     */
    external fun bitwuzla_term_is_fun(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a variable.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a variable.<br></br>
     * Original signature : `bool bitwuzla_term_is_var(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3797*
     */
    external fun bitwuzla_term_is_var(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bound variable.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a variable and bound.<br></br>
     * Original signature : `bool bitwuzla_term_is_bound_var(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3806*
     */
    external fun bitwuzla_term_is_bound_var(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a value.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a value.<br></br>
     * Original signature : `bool bitwuzla_term_is_value(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3815*
     */
    external fun bitwuzla_term_is_value(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector value.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a bit-vector value.<br></br>
     * Original signature : `bool bitwuzla_term_is_bv_value(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3824*
     */
    external fun bitwuzla_term_is_bv_value(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point value.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a floating-point value.<br></br>
     * Original signature : `bool bitwuzla_term_is_fp_value(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3833*
     */
    external fun bitwuzla_term_is_fp_value(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode value.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a rounding mode value.<br></br>
     * Original signature : `bool bitwuzla_term_is_rm_value(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3842*
     */
    external fun bitwuzla_term_is_rm_value(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector term.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a bit-vector term.<br></br>
     * Original signature : `bool bitwuzla_term_is_bv(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3851*
     */
    external fun bitwuzla_term_is_bv(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point term.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a floating-point term.<br></br>
     * Original signature : `bool bitwuzla_term_is_fp(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3860*
     */
    external fun bitwuzla_term_is_fp(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode term.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a rounding mode term.<br></br>
     * Original signature : `bool bitwuzla_term_is_rm(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3869*
     */
    external fun bitwuzla_term_is_rm(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector value representing zero.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a bit-vector zero value.<br></br>
     * Original signature : `bool bitwuzla_term_is_bv_value_zero(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3878*
     */
    external fun bitwuzla_term_is_bv_value_zero(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector value representing one.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a bit-vector one value.<br></br>
     * Original signature : `bool bitwuzla_term_is_bv_value_one(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3887*
     */
    external fun bitwuzla_term_is_bv_value_one(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector value with all bits set to one.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a bit-vector value with all bits set to one.<br></br>
     * Original signature : `bool bitwuzla_term_is_bv_value_ones(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3896*
     */
    external fun bitwuzla_term_is_bv_value_ones(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector minimum signed value.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a bit-vector value with the most significant bit<br></br>
     * set to 1 and all other bits set to 0.<br></br>
     * Original signature : `bool bitwuzla_term_is_bv_value_min_signed(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3906*
     */
    external fun bitwuzla_term_is_bv_value_min_signed(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector maximum signed value.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a bit-vector value with the most significant bit<br></br>
     * set to 0 and all other bits set to 1.<br></br>
     * Original signature : `bool bitwuzla_term_is_bv_value_max_signed(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3916*
     */
    external fun bitwuzla_term_is_bv_value_max_signed(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point positive zero (+zero) value.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a floating-point +zero value.<br></br>
     * Original signature : `bool bitwuzla_term_is_fp_value_pos_zero(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3925*
     */
    external fun bitwuzla_term_is_fp_value_pos_zero(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point value negative zero (-zero).<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a floating-point value negative zero.<br></br>
     * Original signature : `bool bitwuzla_term_is_fp_value_neg_zero(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3934*
     */
    external fun bitwuzla_term_is_fp_value_neg_zero(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point positive infinity (+oo) value.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a floating-point +oo value.<br></br>
     * Original signature : `bool bitwuzla_term_is_fp_value_pos_inf(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3943*
     */
    external fun bitwuzla_term_is_fp_value_pos_inf(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point negative infinity (-oo) value.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a floating-point -oo value.<br></br>
     * Original signature : `bool bitwuzla_term_is_fp_value_neg_inf(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3952*
     */
    external fun bitwuzla_term_is_fp_value_neg_inf(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point NaN value.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a floating-point NaN value.<br></br>
     * Original signature : `bool bitwuzla_term_is_fp_value_nan(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3961*
     */
    external fun bitwuzla_term_is_fp_value_nan(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode RNA value.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a roundindg mode RNA value.<br></br>
     * Original signature : `bool bitwuzla_term_is_rm_value_rna(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3970*
     */
    external fun bitwuzla_term_is_rm_value_rna(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode RNE value.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a rounding mode RNE value.<br></br>
     * Original signature : `bool bitwuzla_term_is_rm_value_rne(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3979*
     */
    external fun bitwuzla_term_is_rm_value_rne(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode RTN value.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a rounding mode RTN value.<br></br>
     * Original signature : `bool bitwuzla_term_is_rm_value_rtn(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3988*
     */
    external fun bitwuzla_term_is_rm_value_rtn(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode RTP value.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a rounding mode RTP value.<br></br>
     * Original signature : `bool bitwuzla_term_is_rm_value_rtp(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:3997*
     */
    external fun bitwuzla_term_is_rm_value_rtp(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode RTZ value.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a rounding mode RTZ value.<br></br>
     * Original signature : `bool bitwuzla_term_is_rm_value_rtz(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:4006*
     */
    external fun bitwuzla_term_is_rm_value_rtz(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a constant array.<br></br>
     * * @param term The term.<br></br>
     * * @return True if `term` is a constant array.<br></br>
     * Original signature : `bool bitwuzla_term_is_const_array(const BitwuzlaTerm*)`<br></br>
     * *native declaration : bitwuzla.h:4015*
     */
    external fun bitwuzla_term_is_const_array(term: BitwuzlaTerm): Boolean

    /**
     * Print term .<br></br>
     * * @param term The term.<br></br>
     *
     * @param format The output format for printing the term. Either `"btor"` for the<br></br>
     * BTOR format, or `"smt2"` for the SMT-LIB v2 format.<br></br>
     * @param file   The file to print the term to.<br></br>
     * Original signature : `void bitwuzla_term_dump(const BitwuzlaTerm*, const char*, FILE*)`<br></br>
     * *native declaration : bitwuzla.h:4025*
     */
    fun bitwuzla_term_dump(term: BitwuzlaTerm, format: String, file: File) {
        throw UnsupportedOperationException("operation is not implemented")
    }

    class FpValue(val sign: String, val exponent: String, val significand: String)
    class ArrayValue(
        val size: Int,
        val indices: Array<BitwuzlaTerm>,
        val values: Array<BitwuzlaTerm>,
        val defaultValue: BitwuzlaTerm
    )

    class FunValue(
        val size: Int,
        val arity: Int,
        val args: Array<Array<BitwuzlaTerm>>,
        val values: Array<BitwuzlaTerm>
    )

    private fun <T : Pointer> Array<T>.mkPtr(): Pointer {
        val memory = Memory(Native.POINTER_SIZE.toLong() * size)
        for (i in indices) {
            memory.setPointer(Native.POINTER_SIZE.toLong() * i, this[i])
        }
        return memory
    }

}
