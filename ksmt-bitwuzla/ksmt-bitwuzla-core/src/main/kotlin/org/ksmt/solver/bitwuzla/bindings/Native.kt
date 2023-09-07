package org.ksmt.solver.bitwuzla.bindings

import io.ksmt.solver.bitwuzla.KBitwuzlaNativeLibraryLoader
import io.ksmt.utils.library.NativeLibraryLoaderUtils


typealias Bitwuzla = Long
typealias BitwuzlaTerm = Long
typealias BitwuzlaSort = Long
typealias BitwuzlaTermArray = LongArray
typealias BitwuzlaSortArray = LongArray

typealias BitwuzlaOptionsNative = Long
typealias BitwuzlaOptionNative = Int
typealias BitwuzlaRoundingModeNative = Int
typealias BitwuzlaResultNative = Int
typealias BitwuzlaKindNative = Int

object Native {
    init {
        NativeLibraryLoaderUtils.load<KBitwuzlaNativeLibraryLoader>()
        bitwuzlaInit()
    }

    /**
     * Initialize Bitwuzla native library.
     */
    @JvmStatic
    private external fun bitwuzlaInit()

    /**
     * decrement external references count to sort
     */
    @JvmStatic
    external fun bitwuzlaSortDecRef(sort: BitwuzlaSort)

    /**
     * decrement external references count to term
     */
    @JvmStatic
    external fun bitwuzlaTermDecRef(term: BitwuzlaTerm)


    /* -------------------------------------------------------------------------- */
    /* Library info                                                               */
    /* -------------------------------------------------------------------------- */

    /**
     * Get copyright information.
     */
    @JvmStatic
    external fun bitwuzlaCopyright(): String

    /**
     * Get version information.
     */
    @JvmStatic
    external fun bitwuzlaVersion(): String

    /**
     * Get git information.
     */
    @JvmStatic
    external fun bitwuzlaGitId(): String


    /* -------------------------------------------------------------------------- */
    /* Options                                                                    */
    /* -------------------------------------------------------------------------- */

    /**
     * Create a new BitwuzlaOptionsNative instance.
     *
     * The returned instance must be deleted via [bitwuzlaOptionsDelete].
     *
     * @return A pointer to the created BitwuzlaOptionsNative instance.
     *
     * @see bitwuzlaOptionsDelete
     */
    @JvmStatic
    external fun bitwuzlaOptionsNew(): BitwuzlaOptionsNative

    /**
     * Delete a BitwuzlaOptionsNative instance.
     *
     * The given instance must have been created via [bitwuzlaOptionsNew].
     *
     * @param options The BitwuzlaOptionsNative instance to delete.
     *
     * @see [bitwuzlaOptionsNew]
     */
    @JvmStatic
    external fun bitwuzlaOptionsDelete(options: BitwuzlaOptionsNative)

    /**
     * Determine if a given option is a numeric (or [Boolean]) option.
     * @return `true` if the given option is a numeric or [Boolean] option.
     */
    @JvmStatic
    fun bitwuzlaOptionIsNumeric(options: BitwuzlaOptionsNative, option: BitwuzlaOption) = bitwuzlaOptionIsNumeric(
        options, option.value
    )

    @JvmStatic
    private external fun bitwuzlaOptionIsNumeric(options: BitwuzlaOptionsNative, option: BitwuzlaOptionNative): Boolean

    /**
     * Determine if a given option is an option with a mode
     * @return `true` if the given option is an option with a mode.
     */
    @JvmStatic
    fun bitwuzlaOptionIsMode(options: BitwuzlaOptionsNative, option: BitwuzlaOption): Boolean = bitwuzlaOptionIsMode(
        options, option.value
    )

    @JvmStatic
    private external fun bitwuzlaOptionIsMode(options: BitwuzlaOptionsNative, option: BitwuzlaOptionNative): Boolean

    /**
     * Set option.
     *
     * @param options The Bitwuzla options instance.
     * @param option The option.
     * @param value The option value.
     *
     * @see BitwuzlaOption
     * @see bitwuzlaGetOption
     */
    @JvmStatic
    fun bitwuzlaSetOption(options: BitwuzlaOptionsNative, option: BitwuzlaOption, value: Long) = bitwuzlaSetOption(
        options, option.value, value
    )

    @JvmStatic
    private external fun bitwuzlaSetOption(options: BitwuzlaOptionsNative, option: BitwuzlaOptionNative, value: Long)

    /**
     * Set option mode for options with modes.
     *
     * @param options The Bitwuzla options instance.
     * @param option The option.
     * @param value The option string value.
     *
     * @see BitwuzlaOption
     * @see bitwuzlaGetOptionMode
     */
    @JvmStatic
    fun bitwuzlaSetOptionMode(options: BitwuzlaOptionsNative, option: BitwuzlaOption, value: String) = bitwuzlaSetOptionMode(
        options, option.value, value
    )

    @JvmStatic
    private external fun bitwuzlaSetOptionMode(options: BitwuzlaOptionsNative, option: BitwuzlaOptionNative, value: String)

    /**
     * Get the current value of an option.
     *
     * @param options The Bitwuzla options instance.
     * @param option The option.
     *
     * @return The option value.
     *
     * @see BitwuzlaOption
     */
    @JvmStatic
    fun bitwuzlaGetOption(options: BitwuzlaOptionsNative, option: BitwuzlaOption) = bitwuzlaGetOption(options, option.value)

    @JvmStatic
    private external fun bitwuzlaGetOption(options: BitwuzlaOptionsNative, option: BitwuzlaOptionNative): Long

    /**
     * Get the current mode of an option as a string if option has modes.
     *
     * @param options The Bitwuzla options instance.
     * @param option The option.
     *
     * @return The option value.
     *
     * @see BitwuzlaOption
     * @see bitwuzlaSetOptionMode
     */
    @JvmStatic
    fun bitwuzlaGetOptionMode(options: BitwuzlaOptionsNative, option: BitwuzlaOption) = bitwuzlaGetOptionMode(
        options, option.value
    )

    @JvmStatic
    private external fun bitwuzlaGetOptionMode(options: BitwuzlaOptionsNative, option: BitwuzlaOptionNative): String


    /* -------------------------------------------------------------------------- */
    /* BitwuzlaSort                                                               */
    /* -------------------------------------------------------------------------- */

    /**
     * Compute the hash value for a sort.
     *
     * @param sort The sort.
     *
     * @return The hash value of the sort.
     */
    @JvmStatic
    external fun bitwuzlaSortHash(sort: BitwuzlaSort): Long

    /**
     * Get the size of a bit-vector sort.
     *
     * Requires that given sort is a bit-vector sort.
     *
     * @param sort The sort.
     *
     * @return The size of the bit-vector sort.
     */
    @JvmStatic
    external fun bitwuzlaSortBvGetSize(sort: BitwuzlaSort): Long

    /**
     * Get the exponent size of a floating-point sort.
     *
     * Requires that given sort is a floating-point sort.
     *
     * @param sort The sort.
     *
     * @return The exponent size of the floating-point sort.
     */
    @JvmStatic
    external fun bitwuzlaSortFpGetExpSize(sort: BitwuzlaSort): Long

    /**
     * Get the significand size of a floating-point sort.
     *
     * Requires that given sort is a floating-point sort.
     *
     * @param sort The sort.
     *
     * @return The significand size of the floating-point sort.
     */
    @JvmStatic
    external fun bitwuzlaSortFpGetSigSize(sort: BitwuzlaSort): Long

    /**
     * Get the index sort of an array sort.
     *
     * Requires that given sort is an array sort.
     *
     * @param sort The sort.
     *
     * @return The index sort of the array sort.
     */
    @JvmStatic
    external fun bitwuzlaSortArrayGetIndex(sort: BitwuzlaSort): BitwuzlaSort

    /**
     * Get the element sort of an array sort.
     *
     * Requires that given sort is an array sort.
     *
     * @param sort The sort.
     *
     * @return The element sort of the array sort.
     */
    @JvmStatic
    external fun bitwuzlaSortArrayGetElement(sort: BitwuzlaSort): BitwuzlaSort

    /**
     * Get the domain sorts of a function sort.
     *
     * The domain sorts are returned as an array of sorts.
     * Requires that given sort is a function sort.
     *
     * @param sort The sort.
     *
     * @return The domain sorts of the function sort.
     */
    @JvmStatic
    external fun bitwuzlaSortFunGetDomainSorts(sort: BitwuzlaSort): BitwuzlaSortArray

    /**
     * Get the codomain sort of a function sort.
     *
     * Requires that given sort is a function sort.
     *
     * @param sort The sort.
     *
     * @return The codomain sort of the function sort.
     */
    @JvmStatic
    external fun bitwuzlaSortFunGetCodomain(sort: BitwuzlaSort): BitwuzlaSort

    /**
     * Get the arity of a function sort.
     *
     * @param sort The sort.
     *
     * @return The number of arguments of the function sort.
     */
    @JvmStatic
    external fun bitwuzlaSortFunGetArity(sort: BitwuzlaSort): Long

    /**
     * Get the symbol of an uninterpreted sort.
     * @param sort The sort.
     * @return The symbol; *null* if no symbol is defined.
     * @note The returned string is only valid until the next [bitwuzlaSortGetUninterpretedSymbol] call.
     */
    @JvmStatic
    external fun bitwuzlaSortGetUninterpretedSymbol(sort: BitwuzlaSort): String?

    /**
     * Determine if two sorts are equal.
     *
     * @param sort0 The first sort.
     * @param sort1 The second sort.
     *
     * @return `true` if the given sorts are equal.
     */
    @JvmStatic
    external fun bitwuzlaSortIsEqual(sort0: BitwuzlaSort, sort1: BitwuzlaSort): Boolean

    /**
     * Determine if a sort is an array sort.
     *
     * @param sort The sort.
     *
     * @return `true` if [sort] is an array sort.
     */
    @JvmStatic
    external fun bitwuzlaSortIsArray(sort: BitwuzlaSort): Boolean

    /**
     * Determine if a sort is a Boolean sort.
     *
     * @param sort The sort.
     *
     * @return `true` if [sort] is a Boolean sort.
     */
    @JvmStatic
    external fun bitwuzlaSortIsBool(sort: BitwuzlaSort): Boolean

    /**
     * Determine if a sort is a bit-vector sort.
     *
     * @param sort The sort.
     *
     * @return true if [sort] is a bit-vector sort.
     */
    @JvmStatic
    external fun bitwuzlaSortIsBv(sort: BitwuzlaSort): Boolean

    /**
     * Determine if a sort is a floating-point sort.
     *
     * @param sort The sort.
     *
     * @return `true` if [sort] is a floating-point sort.
     */
    @JvmStatic
    external fun bitwuzlaSortIsFp(sort: BitwuzlaSort): Boolean

    /**
     * Determine if a sort is a function sort.
     *
     * @param sort The sort.
     *
     * @return `true` if [sort] is a function sort.
     */
    @JvmStatic
    external fun bitwuzlaSortIsFun(sort: BitwuzlaSort): Boolean

    /**
     * Determine if a sort is a Roundingmode sort.
     *
     * @param sort The sort.
     *
     * @return `true` if [sort] is a Roundingmode sort.
     */
    @JvmStatic
    external fun bitwuzlaSortIsRm(sort: BitwuzlaSort): Boolean

    /**
     * Determine if a sort is an uninterpreted sort.
     *
     * @param sort The sort.
     *
     * @return `true` if [sort] is a uninterpreted sort.
     */
    @JvmStatic
    external fun bitwuzlaSortIsUninterpreted(sort: BitwuzlaSort): Boolean

    /**
     * Get the SMT-LIBV v2 string representation of a sort.
     * @return A string representation of the given sort.
     * @note The returned [String] is only valid until the next [bitwuzlaSortToString] call.
     */
    @JvmStatic
    external fun bitwuzlaSortToString(sort: BitwuzlaSort): String

    /* -------------------------------------------------------------------------- */
    /* BitwuzlaTerm                                                               */
    /* -------------------------------------------------------------------------- */

    /**
     * Compute the hash value for a term.
     *
     * @param term The term.
     *
     * @return The hash value of the term.
     */
    @JvmStatic
    external fun bitwuzlaTermHash(term: BitwuzlaTerm): Long

    /**
     * Get the kind of a term.
     *
     * @param term The term.
     *
     * @return The kind of the given term.
     *
     * @see BitwuzlaKind
     */
    @JvmStatic
    fun bitwuzlaTermGetKind(term: BitwuzlaTerm) = BitwuzlaKind.fromValue(bitwuzlaTermGetKindNative(term))

    @JvmStatic
    private external fun bitwuzlaTermGetKindNative(term: BitwuzlaTerm): BitwuzlaKindNative

    /**
     * Get the child terms of a term.
     *
     * Returns `null` if given term does not have children.
     *
     * @param term The term.
     *
     * @return The children of [term] as an array of terms.
     */
    @JvmStatic
    external fun bitwuzlaTermGetChildren(term: BitwuzlaTerm): BitwuzlaTermArray

    /**
     * Get the indices of an indexed term.
     *
     * Requires that given term is an indexed term.
     *
     * @param term The term.
     *
     * @return The indices of [term] as an array of indices.
     */
    @JvmStatic
    external fun bitwuzlaTermGetIndices(term: BitwuzlaTerm): LongArray

    /**
     * Determine if a term is an indexed term.
     *
     * @param term The term.
     *
     * @return `true` if [term] is an indexed term.
     */
    @JvmStatic
    external fun bitwuzlaTermIsIndexed(term: BitwuzlaTerm): Boolean

    /**
     * Get the sort of a term.
     *
     * @param term The term.
     *
     * @return The sort of the term.
     */
    @JvmStatic
    external fun bitwuzlaTermGetSort(term: BitwuzlaTerm): BitwuzlaSort

    /**
     * Get the index sort of an array term.
     *
     * Requires that given term is an array or an array store term.
     *
     * @param term The term.
     *
     * @return The index sort of the array term.
     */
    @JvmStatic
    external fun bitwuzlaTermArrayGetIndexSort(term: BitwuzlaTerm): BitwuzlaSort

    /**
     * Get the element sort of an array term.
     *
     * Requires that given term is an array or an array store term.
     *
     * @param term The term.
     *
     * @return The element sort of the array term.
     */
    @JvmStatic
    external fun bitwuzlaTermArrayGetElementSort(term: BitwuzlaTerm): BitwuzlaSort

    /**
     * Get the domain sorts of a function term.
     *
     * The domain sorts are returned as an array of sorts.
     * Requires that given term is an uninterpreted function, a lambda term, an
     * array store term, or an ite term over function terms.
     *
     *
     * @param term The term.
     *
     * @return The domain sorts of the function term.
     */
    @JvmStatic
    external fun bitwuzlaTermFunGetDomainSorts(term: BitwuzlaTerm): BitwuzlaSortArray

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
    @JvmStatic
    external fun bitwuzlaTermFunGetCodomainSort(term: BitwuzlaTerm): BitwuzlaSort

    /**
     * Get the bit-width of a bit-vector term.
     *
     * Requires that given term is a bit-vector term.
     *
     * @param term The term.
     *
     * @return The bit-width of the bit-vector term.
     */
    @JvmStatic
    external fun bitwuzlaTermBvGetSize(term: BitwuzlaTerm): Long

    /**
     * Get the bit-width of the exponent of a floating-point term.
     *
     * Requires that given term is a floating-point term.
     *
     * @param term The term.
     *
     * @return The bit-width of the exponent of the floating-point term.
     */
    @JvmStatic
    external fun bitwuzlaTermFpGetExpSize(term: BitwuzlaTerm): Long

    /**
     * Get the bit-width of the significand of a floating-point term.
     *
     * Requires that given term is a floating-point term.
     *
     * @param term The term.
     *
     * @return The bit-width of the significand of the floating-point term.
     */
    @JvmStatic
    external fun bitwuzlaTermFpGetSigSize(term: BitwuzlaTerm): Long

    /**
     * Get the arity of a function term.
     *
     * Requires that given term is a function term.
     *
     * @param term The term.
     *
     * @return The arity of the function term.
     */
    @JvmStatic
    external fun bitwuzlaTermFunGetArity(term: BitwuzlaTerm): Long

    /**
     * Get the symbol of a term.
     *
     * @param term The term.
     *
     * @return The symbol of [term]. `null` if no symbol is defined.
     */
    @JvmStatic
    external fun bitwuzlaTermGetSymbol(term: BitwuzlaTerm): String?

    /**
     * Determine if the sorts of two terms are equal.
     *
     * @param term0 The first term.
     * @param term1 The second term.
     *
     * @return `true` if the sorts of [term0] and [term1] are equal.
     */
    @JvmStatic
    external fun bitwuzlaTermIsEqualSort(term0: BitwuzlaTerm, term1: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is an array term.
     *
     * @param term The term.
     *
     * @return `true` if [term] is an array term.
     */
    @JvmStatic
    external fun bitwuzlaTermIsArray(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a constant.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a constant.
     */
    @JvmStatic
    external fun bitwuzlaTermIsConst(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a function.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a function.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFun(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a variable.
     *
     * @param term The term.
     *
     * @return True if `term` is a variable.
     */
    @JvmStatic
    external fun bitwuzlaTermIsVar(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a value.
     *
     * @param term The term.
     *
     * @return True if `term` is a value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsValue(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector value.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a bit-vector value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBvValue(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point value.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a floating-point value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFpValue(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode value.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a rounding mode value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRmValue(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a Boolean term.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a Boolean term.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBool(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector term.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a bit-vector term.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBv(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point term.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a floating-point term.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFp(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode term.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a rounding mode term.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRm(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a term of uninterpreted sort.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a term of uninterpreted sort.
     */
    @JvmStatic
    external fun bitwuzlaTermIsUninterpreted(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is Boolean value true.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a Boolean value true.
     */
    @JvmStatic
    external fun bitwuzlaTermIsTrue(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is Boolean value false.
     *
     * @param term The term.
     *
     * @return `false` if [term] is a Boolean value false.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFalse(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector value representing zero.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a bit-vector zero value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBvValueZero(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector value representing one.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a bit-vector one value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBvValueOne(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector value with all bits set to one.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a bit-vector value with all bits set to one.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBvValueOnes(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector minimum signed value.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a bit-vector value with the most significant bit
     * set to 1 and all other bits set to 0.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBvValueMinSigned(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector maximum signed value.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a bit-vector value with the most significant bit
     * set to 0 and all other bits set to 1.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBvValueMaxSigned(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point positive zero (+zero) value.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a floating-point +zero value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFpValuePosZero(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point value negative zero (-zero).
     *
     * @param term The term.
     *
     * @return `true` if [term] is a floating-point value negative zero.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFpValueNegZero(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point positive infinity (+oo) value.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a floating-point +oo value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFpValuePosInf(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point negative infinity (-oo) value.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a floating-point -oo value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFpValueNegInf(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point NaN value.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a floating-point NaN value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFpValueNan(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode RNA value.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a rounding mode RNA value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRmValueRna(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode RNE value.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a rounding mode RNE value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRmValueRne(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode RTN value.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a rounding mode RTN value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRmValueRtn(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode RTP value.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a rounding mode RTP value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRmValueRtp(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode RTZ value.
     *
     * @param term The term.
     *
     * @return `true` if [term] is a rounding mode RTZ value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRmValueRtz(term: BitwuzlaTerm): Boolean

    /**
     * Get Boolean representation of given Boolean value term.
     *
     * @param term The Boolean value term.
     * @return Boolean representation of value [term].
     */
    @JvmStatic
    external fun bitwuzlaTermValueGetBool(term: BitwuzlaTerm): Boolean

    /**
     * Get string representation of Boolean, bit-vector, floating-point, or
     * rounding mode value term.
     *
     * This returns the raw value string (as opposed to [bitwuzlaTermToString],
     * which returns the SMT-LIB v2 representation of a term).
     * For example, this function returns "010" for a bit-vector value 2 of size
     * 3, while [bitwuzlaTermToString] returns "#b010".
     *
     * @note This uses default binary format for bit-vector value strings.
     *
     * @param term The value term.
     * @return The string representation of the value term.
     *
     * @note Return value is valid until next [bitwuzlaTermValueGetStr] call.
     */
    @JvmStatic
    external fun bitwuzlaTermValueGetStr(term: BitwuzlaTerm): String

    /**
     * Get string representation of Boolean, bit-vector, floating-point, or
     * rounding mode value term. String representations of bit-vector values are
     * printed in the given base.
     *
     * This returns the raw value string (as opposed to [bitwuzlaTermToString],
     * which returns the SMT-LIB v2 representation of a term).
     * For example, this function returns "010" for a bit-vector value 2 of size
     * 3, while [bitwuzlaTermToString] returns "#b010".
     *
     * @param term The value term.
     * @param base The base of the string representation of bit-vector values; `2`
     *             for binary, `10` for decimal, and `16` for hexadecimal. Always
     *             ignored for Boolean and RoundingMode values.
     * @return String representation of the value term.
     *
     * @note The returned string for floating-point values is always the binary
     *       IEEE-754 representation of the value (parameter `base` is ignored).
     *       Parameter `base` always configures the numeric base for bit-vector
     *       values, and for floating-point values in case of the tuple of strings
     *       instantiation. It is always ignored for Boolean and RoundingMode
     *       values.
     *
     * @note Return value is valid until next [bitwuzlaTermValueGetStrFmt]
     * call.
     */
    @JvmStatic
    external fun bitwuzlaTermValueGetStrFmt(term: BitwuzlaTerm, /*uint8_t*/ base: Byte): String

//    /**
//     * Get (the raw) string representation of a given floating-point value
//     * term in IEEE 754 standard representation.
//     *
//     * @param term        The floating-point value term.
//     * @param sign        Output parameter. String representation of the
//     *                    sign bit.
//     * @param exponent    Output parameter. String representation of the exponent
//     *                    bit-vector value.
//     * @param significand Output parameter. String representation of the
//     *                    significand bit-vector value.
//     * @param base        The base in which the component bit-vector strings are
//     *                    given; `2` for binary, `10` for decimal, and `16` for
//     *                    hexadecimal.
//     *
//     * @note Return values sign, exponent and significand are valid until next
//     *       `bitwuzla_term_value_get_fp_ieee` call.
//     */
//    @JvmStatic
//    external fun bitwuzla_term_value_get_fp_ieee(term: BitwuzlaTerm,
//    const char **sign,
//    const char **exponent,
//    const char **significand,
//    uint8_t base)

    /**
     * Get representation of given rounding mode value term.
     * @param term The rounding mode value term.
     * @return The [BitwuzlaRoundingMode] representation of the given rounding mode value.
     */
    @JvmStatic
    fun bitwuzlaTermValueGetRoundingMode(term: BitwuzlaTerm): BitwuzlaRoundingMode = BitwuzlaRoundingMode.fromValue(
        bitwuzlaTermValueGetRm(term)
    )

    @JvmStatic
    private external fun bitwuzlaTermValueGetRm(term: BitwuzlaTerm): BitwuzlaRoundingModeNative

    /**
     * Get the SMT-LIB v2 string representation of a term.
     * @note This uses default binary format for bit-vector value strings.
     * @return A string representation of the given term.
     * @note The returned [String] is only valid until the next [bitwuzlaTermToString] call.
     */
    @JvmStatic
    external fun bitwuzlaTermToString(term: BitwuzlaTerm): String

    /**
     * Get the SMT-LIB v2 string representation of a term. String representations
     * of bit-vector values are printed in the given base.
     * @param base The base of the string representation of bit-vector values; `2`
     *             for binary, `10` for decimal, and `16` for hexadecimal. Always
     *             ignored for Boolean and RoundingMode values.
     * @return A string representation of the given term.
     *
     * @note The returned char* pointer is only valid until the next [bitwuzlaTermToString] call.
     */
    @JvmStatic
    external fun bitwuzlaTermToStringFmt(term: BitwuzlaTerm, /*uint8_t*/ base: Byte): String


    /* -------------------------------------------------------------------------- */
    /* Bitwuzla                                                                   */
    /* -------------------------------------------------------------------------- */

    /**
     * Create a new Bitwuzla instance.
     *
     * The returned instance must be deleted via [bitwuzlaDelete].
     *
     * @param options The associated options (optional).
     * @return A pointer to the created Bitwuzla instance.
     *
     * @see [bitwuzlaDelete]
     */
    @JvmStatic
    external fun bitwuzlaNew(options: BitwuzlaOptionsNative): Bitwuzla

    /**
     * Delete a Bitwuzla instance.
     *
     * The given instance must have been created via [bitwuzlaNew].
     *
     * @param bitwuzla The Bitwuzla instance to delete.
     *
     * @see [bitwuzlaNew]
     */
    @JvmStatic
    external fun bitwuzlaDelete(bitwuzla: Bitwuzla)

//    /**
//     * Configure a termination callback function.
//     *
//     * The `state` of the callback can be retrieved via
//     * `bitwuzla_get_termination_callback_state()`.
//     *
//     * @param bitwuzla The Bitwuzla instance.
//     * @param fun The callback function, returns a value != 0 if `bitwuzla` should
//     *            be terminated.
//     * @param state The argument to the callback function.
//     *
//     * @see
//     *   * `bitwuzla_terminate`
//     *   * `bitwuzla_get_termination_callback_state`
//     */
//    bitwuzla_set_termination_callback(Bitwuzla *bitwuzla,
//    int32_t (*fun)(void *),
//    void *state);

//    /**
//     * Get the state of the termination callback function.
//     *
//     * The returned object representing the state of the callback corresponds to
//     * the `state` configured as argument to the callback function via
//     * `bitwuzla_set_termination_callback()`.
//     *
//     * @param bitwuzla The Bitwuzla instance.
//     *
//     * @return The object passed as argument `state` to the callback function.
//     *
//     * @see
//     *   * `bitwuzla_terminate`
//     *   * `bitwuzla_set_termination_callback`
//     */
//    void *bitwuzla_get_termination_callback_state(Bitwuzla *bitwuzla);

//    /**
//     * Configure an abort callback function, which is called instead of exit
//     * on abort conditions.
//     *
//     * @note If the abort callback function throws a C++ exception, this must be
//     *       thrown via std::rethrow_if_nested.
//     *
//     * @param fun The callback function. Argument `msg` explains the reason for the
//     *            abort.
//     */
//    void bitwuzla_set_abort_callback(void (*fun)(const char *msg));

    /**
     * Push context levels.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param nlevels The number of context levels to push.
     *
     * @see [bitwuzlaSetOption]
     */
    @JvmStatic
    external fun bitwuzlaPush(bitwuzla: Bitwuzla, nlevels: Long)

    /**
     * Pop context levels.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param nlevels The number of context levels to pop.
     *
     * @see [bitwuzlaSetOption]
     */
    @JvmStatic
    external fun bitwuzlaPop(bitwuzla: Bitwuzla, nlevels: Long)

    /**
     * Assert formula.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param term The formula to assert.
     */
    @JvmStatic
    external fun bitwuzlaAssert(bitwuzla: Bitwuzla, term: BitwuzlaTerm)

    /**
     * Get the set of currently asserted formulas.
     * @return The asserted formulas.
     * @return An array with the set of asserted formulas. Only
     * valid until the next [bitwuzlaGetAssertions] call.
     */
    @JvmStatic
    external fun bitwuzlaGetAssertions(bitwuzla: Bitwuzla): BitwuzlaTermArray

    /**
     * Determine if an assumption is an unsat assumption.
     *
     * Unsat assumptions are assumptions that force an input formula to become
     * unsatisfiable. Unsat assumptions handling in Bitwuzla is analogous to
     * failed assumptions in MiniSAT.
     *
     * Requires that the last [bitwuzlaCheckSat] query returned `::BITWUZLA_UNSAT`.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param term The assumption to check for.
     *
     * @return `true` if given assumption is an unsat assumption.
     *
     * @see bitwuzlaSetOption
     * @see bitwuzlaCheckSat
     */
    @JvmStatic
    external fun bitwuzlaIsUnsatAssumption(bitwuzla: Bitwuzla, term: BitwuzlaTerm): Boolean

    /**
     * Get the set of unsat assumptions.
     *
     * Unsat assumptions are assumptions that force an input formula to become
     * unsatisfiable. Unsat assumptions handling in Bitwuzla is analogous to
     * failed assumptions in MiniSAT.
     *
     * Requires that the last `bitwuzla_check_sat()` query returned [BitwuzlaResult.BITWUZLA_UNSAT].
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return An array with unsat assumptions. Only valid until
     * the next [bitwuzlaGetUnsatAssumptions] call.
     *
     * @see bitwuzlaSetOption
     * @see bitwuzlaCheckSat
     */
    @JvmStatic
    external fun bitwuzlaGetUnsatAssumptions(bitwuzla: Bitwuzla): BitwuzlaTermArray

    /**
     * Get the unsat core (unsat assertions).
     *
     * The unsat core consists of the set of assertions that force an input formula
     * to become unsatisfiable.
     *
     * Requires that the last [bitwuzlaCheckSat] query returned [BitwuzlaResult.BITWUZLA_UNSAT].
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return An array with unsat assertions. Only valid until
     * the next [bitwuzlaGetUnsatCore] call.
     *
     * @see bitwuzlaAssert
     * @see bitwuzlaCheckSat
     */
    @JvmStatic
    external fun bitwuzlaGetUnsatCore(bitwuzla: Bitwuzla): BitwuzlaTermArray

    /**
     * Simplify the current input formula.
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @note Each call to [bitwuzlaCheckSat] and [bitwuzlaCheckSatAssuming]
     * simplifies the input formula as a preprocessing step. It is not
     * necessary to call this function explicitly in the general case.
     *
     * @see bitwuzlaAssert
     */
    @JvmStatic
    external fun bitwuzlaSimplify(bitwuzla: Bitwuzla)

    /**
     * Check satisfiability of current input formula.
     *
     * An input formula consists of assertions added via [bitwuzlaAssert].
     * The search for a solution can by guided by additionally making assumptions
     * (see [bitwuzlaCheckSatAssuming]).
     *
     * @note Assertions and assumptions are combined via Boolean and.
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return [BitwuzlaResult.BITWUZLA_SAT] if the input formula is satisfiable and
     * [BitwuzlaResult.BITWUZLA_UNSAT] if it is unsatisfiable,
     * and [BitwuzlaResult.BITWUZLA_UNKNOWN] when neither satisfiability nor unsatisfiability was determined.
     * This can happen when [bitwuzla] was terminated via a termination callback.
     *
     * @see bitwuzlaAssert
     * @see bitwuzlaSetOption
     * @see BitwuzlaResult
     */
    @JvmStatic
    fun bitwuzlaCheckSat(bitwuzla: Bitwuzla): BitwuzlaResult = BitwuzlaResult.fromValue(
        bitwuzlaCheckSatNative(bitwuzla)
    )

    /**
    @return `::BITWUZLA_SAT` if the input formula is satisfiable and
     * `::BITWUZLA_UNSAT` if it is unsatisfiable, and `::BITWUZLA_UNKNOWN`
     * when neither satisfiability nor unsatisfiability was determined.
     * This can happen when `bitwuzla` was terminated via a termination
     * callback.
     *
     * *note*: api call: bitwuzla_check_sat
    */
    @JvmStatic
    external fun bitwuzlaCheckSatNative(bitwuzla: Bitwuzla): BitwuzlaResultNative

    /**
     * Check satisfiability of current input formula wrt to the given set of
     * assumptions.
     *
     * An input formula consists of assertions added via [bitwuzlaAssert].
     * The search for a solution can by guided by additionally making assumptions
     * (the given set of assumptions [args]).
     *
     * @note Assertions and assumptions are combined via Boolean and.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param args     The assumptions.
     *
     * @return [BitwuzlaResult.BITWUZLA_SAT] if the input formula is satisfiable and [BitwuzlaResult.BITWUZLA_UNSAT]
     * if it is unsatisfiable, and [BitwuzlaResult.BITWUZLA_UNKNOWN]
     * when neither satisfiability nor unsatisfiability was determined.
     * This can happen when [bitwuzla] was terminated via a termination
     * callback.
     *
     * @see bitwuzlaAssert
     * @see bitwuzlaSetOption
     * @see BitwuzlaResult
     */
    @JvmStatic
    fun bitwuzlaCheckSatAssuming(bitwuzla: Bitwuzla, args: BitwuzlaTermArray) = BitwuzlaResult.fromValue(
        bitwuzlaCheckSatAssumingNative(bitwuzla, args)
    )

    /**
     * *note:* api call: bitwuzla_check_sat_assuming
     */
    @JvmStatic
    external fun bitwuzlaCheckSatAssumingNative(bitwuzla: Bitwuzla, args: BitwuzlaTermArray): BitwuzlaResultNative

    /**
     * Get a term representing the model value of a given term.
     *
     * Requires that the last [bitwuzlaCheckSat] query returned [BitwuzlaResult.BITWUZLA_SAT].
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param term The term to query a model value for.
     *
     * @return A term representing the model value of term [term].
     *
     * @see bitwuzlaCheckSat
     */
    @JvmStatic
    external fun bitwuzlaGetValue(bitwuzla: Bitwuzla, term: BitwuzlaTerm): BitwuzlaTerm

    @JvmStatic
    external fun bitwuzlaGetBvValueUInt64(term: BitwuzlaTerm): Long

    @JvmStatic
    external fun bitwuzlaGetBvValueString(term: BitwuzlaTerm): String

    @JvmStatic
    external fun bitwuzlaGetFpValue(term: BitwuzlaTerm): FpValue


    // TODO: mb add statistics?
//    /**
//     * Get current statistics.
//     *
//     * The statistics are retrieved as a mapping from statistic name (`keys`)
//     * to statistic value (`values`).
//     *
//     * @param bitwuzla The Bitwuzla instance.
//     * @param keys     The resulting set of statistic names.
//     * @param values   The resulting set of statistic values.
//     * @param size     The resulting size of `keys` and `values`.
//     *
//     * @note Output parameters `keys` and `values` are only valid until the
//     *       next call to `bitwuzla_get_statistics()`.
//     */
//    void bitwuzla_get_statistics(Bitwuzla *bitwuzla,
//    const char ***keys,
//    const char ***values,
//    size_t *size);


    /* -------------------------------------------------------------------------- */
    /* Sort creation                                                              */
    /* -------------------------------------------------------------------------- */

    /**
     * Create an array sort.
     *
     * @param index The index sort of the array sort.
     * @param element The element sort of the array sort.
     * @return An array sort which maps sort [index] to sort [element].
     *
     * @see bitwuzlaSortIsArray
     * @see bitwuzlaSortArrayGetIndex
     * @see bitwuzlaSortArrayGetElement
     * @see bitwuzlaTermIsArray
     * @see bitwuzlaTermArrayGetIndexSort
     * @see bitwuzlaTermArrayGetElementSort
     */
    @JvmStatic
    external fun bitwuzlaMkArraySort(index: BitwuzlaSort, element: BitwuzlaSort): BitwuzlaSort

    /**
     * Create a Boolean sort.
     * @return A Boolean sort.
     */
    @JvmStatic
    external fun bitwuzlaMkBoolSort(): BitwuzlaSort

    /**
     * Create a bit-vector sort of given size.
     *
     * @param size The size of the bit-vector sort.
     * @return A bit-vector sort of given size.
     *
     * @see bitwuzlaSortIsBv
     * @see bitwuzlaSortBvGetSize
     * @see bitwuzlaTermIsBv
     * @see bitwuzlaTermBvGetSize
     */
    @JvmStatic
    external fun bitwuzlaMkBvSort(size: Long): BitwuzlaSort

    /**
     * Create a floating-point sort of given exponent and significand size.
     *
     * @param expSize The size of the exponent.
     * @param sigSize The size of the significand (including sign bit).
     * @return A floating-point sort of given format.
     *
     * @see bitwuzlaSortIsFp
     * @see bitwuzlaSortFpGetExpSize
     * @see bitwuzlaSortFpGetSigSize
     * @see bitwuzlaTermIsFp
     * @see bitwuzlaTermFpGetExpSize
     * @see bitwuzlaTermFpGetSigSize
     */
    @JvmStatic
    external fun bitwuzlaMkFpSort(expSize: Long, sigSize: Long): BitwuzlaSort

    /**
     * Create a function sort.
     *
     * @param domain The domain sorts (the sorts of the arguments).
     * @param codomain The codomain sort (the sort of the return value).
     * @return A function sort of given domain and codomain sorts.
     *
     * @see bitwuzlaSortIsFun
     * @see bitwuzlaSortFunGetArity
     * @see bitwuzlaSortFunGetDomainSorts
     * @see bitwuzlaSortFunGetCodomain
     * @see bitwuzlaTermIsFun
     * @see bitwuzlaTermFunGetArity
     * @see bitwuzlaTermFunGetDomainSorts
     * @see bitwuzlaTermFunGetCodomainSort
     */
    @JvmStatic
    external fun bitwuzlaMkFunSort(domain: BitwuzlaSortArray, codomain: BitwuzlaSort): BitwuzlaSort

    /**
     * Create a Roundingmode sort.
     * @return A RoundingMode sort.
     * @see bitwuzlaSortIsRm
     * @see bitwuzlaTermIsRm
     */
    @JvmStatic
    external fun bitwuzlaMkRmSort(): BitwuzlaSort

    /**
     * Create an uninterpreted sort.
     * @param symbol The symbol of the sort. May be `null`.
     * @return A uninterpreted sort.
     * @see bitwuzlaSortIsUninterpreted
     * @see bitwuzlaTermIsUninterpreted
     */
    @JvmStatic
    external fun bitwuzlaMkUninterpretedSort(symbol: String?): BitwuzlaSort


    /* -------------------------------------------------------------------------- */
    /* Term creation                                                              */
    /* -------------------------------------------------------------------------- */

    /**
     * Create a true value.
     * @return A term representing true.
     */
    @JvmStatic
    external fun bitwuzlaMkTrue(): BitwuzlaTerm

    /**
     * Create a false value.
     * @return A term representing false.
     */
    @JvmStatic
    external fun bitwuzlaMkFalse(): BitwuzlaTerm

    /**
     * Create a bit-vector value zero.
     *
     * @param sort The sort of the value.
     * @return A term representing the bit-vector value 0 of given sort.
     *
     * @see bitwuzlaMkBvSort
     */
    @JvmStatic
    external fun bitwuzlaMkBvZero(sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a bit-vector value one.
     *
     * @param sort The sort of the value.
     * @return A term representing the bit-vector value 1 of given sort.
     *
     * @see bitwuzlaMkBvSort
     */
    @JvmStatic
    external fun bitwuzlaMkBvOne(sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a bit-vector value where all bits are set to 1.
     *
     * @param sort The sort of the value.
     * @return A term representing the bit-vector value of given sort where all bits are set to 1.
     * @see bitwuzlaMkBvSort
     */
    @JvmStatic
    external fun bitwuzlaMkBvOnes(sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a bit-vector minimum signed value.
     *
     * @param sort The sort of the value.
     * @return A term representing the bit-vector value of given sort where the MSB
     * is set to 1 and all remaining bits are set to 0.
     * @see bitwuzlaMkBvSort
     */
    @JvmStatic
    external fun bitwuzlaMkBvMinSigned(sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a bit-vector maximum signed value.
     *
     * @param sort The sort of the value.
     * @return A term representing the bit-vector value of given sort where the MSB
     * is set to 0 and all remaining bits are set to 1.
     * @see bitwuzlaMkBvSort
     */
    @JvmStatic
    external fun bitwuzlaMkBvMaxSigned(sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a floating-point positive zero value (SMT-LIB: `+zero`).
     *
     * @param sort The sort of the value.
     * @return A term representing the floating-point positive zero value of given floating-point sort.
     * @see bitwuzlaMkFpSort
     */
    @JvmStatic
    external fun bitwuzlaMkFpPosZero(sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a floating-point negative zero value (SMT-LIB: `-zero`).
     *
     * @param sort The sort of the value.
     * @return A term representing the floating-point negative zero value of given floating-point sort.
     * @see bitwuzlaMkFpSort
     */
    @JvmStatic
    external fun bitwuzlaMkFpNegZero(sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a floating-point positive infinity value (SMT-LIB: `+oo`).
     *
     * @param sort The sort of the value.
     * @return A term representing the floating-point positive infinity value of given floating-point sort.
     * @see bitwuzlaMkFpSort
     */
    @JvmStatic
    external fun bitwuzlaMkFpPosInf(sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a floating-point negative infinity value (SMT-LIB: `-oo`).
     *
     * @param sort The sort of the value.
     * @return A term representing the floating-point negative infinity value of given floating-point sort.
     * @see bitwuzlaMkFpSort
     */
    @JvmStatic
    external fun bitwuzlaMkFpNegInf(sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a floating-point NaN value.
     *
     * @param sort The sort of the value.
     * @return A term representing the floating-point NaN value of given floating-point sort.
     * @see bitwuzlaMkFpSort
     */
    @JvmStatic
    external fun bitwuzlaMkFpNan(sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a bit-vector value from its string representation.
     *
     * Parameter [base] determines the base of the string representation.
     *
     * @note Given value must fit into a bit-vector of given size (sort).
     *
     * @param sort The sort of the value.
     * @param value A string representing the value.
     * @param base The base in which the string is given; `2` for binary, `10` for decimal, and `16` for hexadecimal.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VALUE], representing the bit-vector value
     * of given sort.
     *
     * @see bitwuzlaMkBvSort
     */
    @JvmStatic
    fun bitwuzlaMkBvValue(sort: BitwuzlaSort, value: String, /*uint8_t*/ base: BitwuzlaBVBase) = bitwuzlaMkBvValue(
        sort, value, base.nativeValue
    )

    @JvmStatic
    private external fun bitwuzlaMkBvValue(sort: BitwuzlaSort, value: String, /*uint8_t*/ base: Byte): BitwuzlaTerm

    /**
     * Create a bit-vector value from its unsigned integer representation.
     *
     * @note Given value must fit into a bit-vector of given size (sort).
     *
     * @param sort The sort of the value.
     * @param value The unsigned integer representation of the bit-vector value.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VALUE], representing the bit-vector value
     * of given sort.
     *
     * @see bitwuzlaMkBvSort
     */
    @JvmStatic
    external fun bitwuzlaMkBvValueUint64(sort: BitwuzlaSort, value: Long): BitwuzlaTerm

    /**
     * Create a bit-vector value from its signed integer representation.
     *
     * @note Given value must fit into a bit-vector of given size (sort).
     *
     * @param sort The sort of the value.
     * @param value The unsigned integer representation of the bit-vector value.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VALUE], representing the bit-vector value of given sort.
     *
     * @see bitwuzlaMkBvSort
     */
    @JvmStatic
    external fun bitwuzlaMkBvValueInt64(sort: BitwuzlaSort, value: Long): BitwuzlaTerm

    /**
     * Create a floating-point value from its IEEE 754 standard representation
     * given as three bit-vector values representing the sign bit, the exponent and the significand.
     *
     * @param bvSign The sign bit.
     * @param bvExponent The exponent bit-vector value.
     * @param bvSignificand The significand bit-vector value.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VALUE], representing the floating-point value.
     */
    @JvmStatic
    external fun bitwuzlaMkFpValue(
        bvSign: BitwuzlaTerm,
        bvExponent: BitwuzlaTerm,
        bvSignificand: BitwuzlaTerm
    ): BitwuzlaTerm

    /**
     * Create a floating-point value from its real representation, given as a
     * decimal string, with respect to given rounding mode.
     *
     * @note Given rounding mode may be an arbitrary, non-value rounding mode term.
     *       If it is a value, the returned term will be a floating-point value,
     *       else a non-value floating-point term.
     *
     * @param sort The sort of the value.
     * @param rm The rounding mode.
     * @param real The decimal string representing a real value.
     *
     * @return A floating-point representation of the given real string. If [rm]
     * is of kind [BitwuzlaKind.BITWUZLA_KIND_VALUE] the floating-point will be of kind
     * [BitwuzlaKind.BITWUZLA_KIND_VALUE], else it will be a non-value term.
     *
     * @see bitwuzlaMkFpSort
     */
    @JvmStatic
    external fun bitwuzlaMkFpFromReal(sort: BitwuzlaSort, rm: BitwuzlaTerm, real: String): BitwuzlaTerm

    /**
     * Create a floating-point value from its rational representation, given as
     * two decimal strings representing the numerator and denominator, with respect
     * to given rounding mode.
     *
     * @note Given rounding mode may be an arbitrary, non-value rounding mode term.
     *       If it is a value, the returned term will be a floating-point value,
     *       else a non-value floating-point term.
     *
     * @param sort The sort of the value.
     * @param rm The rounding mode.
     * @param num The decimal string representing the numerator.
     * @param den The decimal string representing the denominator.
     *
     * @return A floating-point representation of the given rational string. If
     * [rm] is of kind [BitwuzlaKind.BITWUZLA_KIND_VALUE] the floating-point will be of
     * kind [BitwuzlaKind.BITWUZLA_KIND_VALUE], else it will be a non-value term.
     *
     * @see bitwuzlaMkFpSort
     */
    @JvmStatic
    external fun bitwuzlaMkFpFromRational(sort: BitwuzlaSort, rm: BitwuzlaTerm, num: String, den: String): BitwuzlaTerm

    /**
     * Create a rounding mode value.
     *
     * @param rm The rounding mode value.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VALUE], representing the rounding mode value.
     *
     * @see BitwuzlaRoundingMode
     */
    @JvmStatic
    fun bitwuzlaMkRmValue(rm: BitwuzlaRoundingMode): BitwuzlaTerm = bitwuzlaMkRmValue(rm.value)

    @JvmStatic
    private external fun bitwuzlaMkRmValue(rm: BitwuzlaRoundingModeNative): BitwuzlaTerm

    /**
     * Create a term of given kind with one argument term.
     *
     * @param kind The operator kind.
     * @param arg The argument to the operator.
     *
     * @return A term representing an operation of given kind.
     *
     * @see [BitwuzlaKind]
     */
    @JvmStatic
    fun bitwuzlaMkTerm(kind: BitwuzlaKind, arg: BitwuzlaTerm): BitwuzlaTerm = bitwuzlaMkTerm1(kind.value, arg)

    @JvmStatic
    private external fun bitwuzlaMkTerm1(kind: BitwuzlaKindNative, arg: BitwuzlaTerm): BitwuzlaTerm

    /**
     * Create a term of given kind with two argument terms.
     *
     * @param kind The operator kind.
     * @param arg0 The first argument to the operator.
     * @param arg1 The second argument to the operator.
     *
     * @return A term representing an operation of given kind.
     *
     * @see [BitwuzlaKind]
     */
    @JvmStatic
    fun bitwuzlaMkTerm(kind: BitwuzlaKind, arg0: BitwuzlaTerm, arg1: BitwuzlaTerm): BitwuzlaTerm = bitwuzlaMkTerm2(
        kind.value, arg0, arg1
    )

    @JvmStatic
    private external fun bitwuzlaMkTerm2(kind: BitwuzlaKindNative, arg0: BitwuzlaTerm, arg1: BitwuzlaTerm): BitwuzlaTerm

    /**
     * Create a term of given kind with three argument terms.
     *
     * @param kind The operator kind.
     * @param arg0 The first argument to the operator.
     * @param arg1 The second argument to the operator.
     * @param arg2 The third argument to the operator.
     *
     * @return A term representing an operation of given kind.
     *
     * @see BitwuzlaKind
     */
    @JvmStatic
    fun bitwuzlaMkTerm(
        kind: BitwuzlaKind,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        arg2: BitwuzlaTerm
    ): BitwuzlaTerm = bitwuzlaMkTerm3(kind.value, arg0, arg1, arg2)

    @JvmStatic
    private external fun bitwuzlaMkTerm3(
        kind: BitwuzlaKindNative,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        arg2: BitwuzlaTerm
    ): BitwuzlaTerm

    /**
     * Create a term of given kind with the given argument terms.
     *
     * @param kind The operator kind.
     * @param args The argument terms.
     *
     * @return A term representing an operation of given kind.
     *
     * @see BitwuzlaKind
     */
    @JvmStatic
    fun bitwuzlaMkTerm(kind: BitwuzlaKind, args: BitwuzlaTermArray): BitwuzlaTerm =  bitwuzlaMkTerm(kind.value, args)

    @JvmStatic
    private external fun bitwuzlaMkTerm(kind: BitwuzlaKindNative, args: BitwuzlaTermArray): BitwuzlaTerm

    /**
     * Create an indexed term of given kind with one argument term and one index.
     *
     * @param kind The operator kind.
     * @param arg The argument term.
     * @param idx The index.
     *
     * @return A term representing an indexed operation of given kind.
     *
     * @see BitwuzlaKind
     */
    @JvmStatic
    fun bitwuzlaMkTermIndexed(kind: BitwuzlaKind, arg: BitwuzlaTerm, idx: Long): BitwuzlaTerm = bitwuzlaMkTerm1Indexed1(
        kind.value, arg, idx
    )

    @JvmStatic
    private external fun bitwuzlaMkTerm1Indexed1(kind: BitwuzlaKindNative, arg: BitwuzlaTerm, idx: Long): BitwuzlaTerm

    /**
     * Create an indexed term of given kind with one argument term and two indices.
     *
     * @param kind The operator kind.
     * @param arg The argument term.
     * @param idx0 The first index.
     * @param idx1 The second index.
     *
     * @return A term representing an indexed operation of given kind.
     *
     * @see BitwuzlaKind
     */
    @JvmStatic
    fun bitwuzlaMkTermIndexed(
        kind: BitwuzlaKind,
        arg: BitwuzlaTerm,
        idx0: Long,
        idx1: Long
    ): BitwuzlaTerm = bitwuzlaMkTerm1Indexed2(kind.value, arg, idx0, idx1)

    @JvmStatic
    private external fun bitwuzlaMkTerm1Indexed2(
        kind: BitwuzlaKindNative,
        arg: BitwuzlaTerm,
        idx0: Long,
        idx1: Long
    ): BitwuzlaTerm

    /**
     * Create an indexed term of given kind with two argument terms and one index.
     *
     * @param kind The operator kind.
     * @param arg0 The first argument term.
     * @param arg1 The second argument term.
     * @param idx The index.
     *
     * @return A term representing an indexed operation of given kind.
     *
     * @see BitwuzlaKind
     */
    @JvmStatic
    fun bitwuzlaMkTerm2Indexed(
        kind: BitwuzlaKind,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        idx: Long
    ): BitwuzlaTerm = bitwuzlaMkTerm2Indexed1(kind.value, arg0, arg1, idx)

    @JvmStatic
    private external fun bitwuzlaMkTerm2Indexed1(
        kind: BitwuzlaKindNative,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        idx: Long
    ): BitwuzlaTerm

    /**
     * Create an indexed term of given kind with two argument terms and two indices.
     *
     * @param kind The operator kind.
     * @param arg0 The first argument term.
     * @param arg1 The second argument term.
     * @param idx0 The first index.
     * @param idx1 The second index.
     *
     * @return A term representing an indexed operation of given kind.
     *
     * @see BitwuzlaKind
     */
    @JvmStatic
    fun bitwuzlaMkTermIndexed(
        kind: BitwuzlaKind,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        idx0: Long,
        idx1: Long
    ): BitwuzlaTerm = bitwuzlaMkTerm2Indexed2(kind.value, arg0, arg1, idx0, idx1)

    @JvmStatic
    private external fun bitwuzlaMkTerm2Indexed2(
        kind: BitwuzlaKindNative,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        idx0: Long,
        idx1: Long
    ): BitwuzlaTerm

    /**
     * Create an indexed term of given kind with the given argument terms and indices.
     *
     * @param kind The operator kind.
     * @param args The argument terms.
     * @param idxs The indices.
     *
     * @return A term representing an indexed operation of given kind.
     *
     * @see BitwuzlaKind
     */
    @JvmStatic
    fun bitwuzlaMkTermIndexed(
        kind: BitwuzlaKind,
        args: BitwuzlaTermArray,
        idxs: LongArray
    ): BitwuzlaTerm = bitwuzlaMkTermIndexed(kind.value, args, idxs)

    @JvmStatic
    private external fun bitwuzlaMkTermIndexed(
        kind: BitwuzlaKindNative,
        args: BitwuzlaTermArray,
        idxs: LongArray
    ): BitwuzlaTerm

    /**
     * Create a (first-order) constant of given sort with given symbol.
     *
     * @param sort The sort of the constant.
     * @param symbol The symbol of the constant.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_CONSTANT], representing the constant.
     *
     * @see bitwuzlaMkArraySort
     * @see bitwuzlaMkBoolSort
     * @see bitwuzlaMkBvSort
     * @see bitwuzlaMkFpSort
     * @see bitwuzlaMkFunSort
     * @see bitwuzlaMkRmSort
     */
    @JvmStatic
    external fun bitwuzlaMkConst(sort: BitwuzlaSort, symbol: String): BitwuzlaTerm

    /**
     * Create a one-dimensional constant array of given sort, initialized with
     * given value.
     *
     * @param sort The sort of the array.
     * @param value The term to initialize the elements of the array with.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_CONST_ARRAY], representing a constant array of given sort.
     *
     * @see bitwuzlaMkArraySort
     */
    @JvmStatic
    external fun bitwuzlaMkConstArray(sort: BitwuzlaSort, value: BitwuzlaTerm): BitwuzlaTerm

    /**
     * Create a variable of given sort with given symbol.
     *
     * @note This creates a variable to be bound by quantifiers or lambdas.
     *
     * @param sort The sort of the variable.
     * @param symbol The symbol of the variable.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VARIABLE], representing the variable.
     *
     * @see bitwuzlaMkBoolSort
     * @see bitwuzlaMkBvSort
     * @see bitwuzlaMkFpSort
     * @see bitwuzlaMkFunSort
     * @see bitwuzlaMkRmSort
     */
    @JvmStatic
    external fun bitwuzlaMkVar(sort: BitwuzlaSort, symbol: String): BitwuzlaTerm


    /* -------------------------------------------------------------------------- */
    /* Term substitution                                                          */
    /* -------------------------------------------------------------------------- */

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
    @JvmStatic
    external fun bitwuzlaSubstituteTerm(
        bitwuzla: Bitwuzla,
        term: BitwuzlaTerm,
        mapKeys: BitwuzlaTermArray,
        mapValues: BitwuzlaTermArray
    ): BitwuzlaTerm

    /**
     * Substitute a set of keys with their corresponding values in the set of given
     * terms.
     *
     * The terms in [terms] are replaced with the terms resulting from this substitutions.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param terms The terms in which the keys are to be substituted.
     * @param mapKeys The keys.
     * @param mapValues The mapped values.
     */
    @JvmStatic
    external fun bitwuzlaSubstituteTerms(
        bitwuzla: Bitwuzla,
        terms: BitwuzlaTermArray,
        mapKeys: BitwuzlaTermArray,
        mapValues: BitwuzlaTermArray
    )
}
