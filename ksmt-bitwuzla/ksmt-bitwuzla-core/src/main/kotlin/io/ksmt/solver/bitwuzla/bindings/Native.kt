package io.ksmt.solver.bitwuzla.bindings

typealias BitwuzlaTermManager = Long
typealias Bitwuzla = Long
typealias BitwuzlaOptions = Long
typealias BitwuzlaTerm = Long
typealias BitwuzlaSort = Long
typealias BitwuzlaTermArray = LongArray
typealias BitwuzlaSortArray = LongArray

object Native {
    init {
        System.load("/home/sobol/CLionProjects/bitwuzla/dist/lib/libbitwuzlabv.so")
        System.load("/home/sobol/CLionProjects/bitwuzla/dist/lib/libbitwuzlabb.so")
        System.load("/home/sobol/CLionProjects/bitwuzla/dist/lib/libbitwuzlals.so")
        System.load("/home/sobol/CLionProjects/bitwuzla/dist/lib/libbitwuzla.so")
        System.load("/home/sobol/IdeaProjects/ksmt/ksmt-bitwuzla/bindings-native/cmake-build-debug/libbitwuzla_jni.so")

//        NativeLibraryLoaderUtils.load<KBitwuzlaNativeLibraryLoader>()
    }

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

    /**
     * Create a new Bitwuzla term manager.
     *
     * The returned instance must be deleted via [bitwuzlaTermManagerDelete].
     */
    @JvmStatic
    external fun bitwuzlaTermManagerNew(): BitwuzlaTermManager

    /**
     * Release memory but don't deleter term manager.
     *
     * @see [bitwuzlaTermManagerDelete]
     */
    @JvmStatic
    external fun bitwuzlaTermManagerRelease(termManager: BitwuzlaTermManager)

    /**
     * Delete term manager and release memory.
     */
    @JvmStatic
    external fun bitwuzlaTermManagerDelete(termManager: BitwuzlaTermManager)

    /**
     * Decrement [sort] native pointer counter.
     */
    @JvmStatic
    external fun bitwuzlaSortRelease(sort: BitwuzlaSort)

    /**
     * Decrement [term] native pointer counter.
     */
    @JvmStatic
    external fun bitwuzlaTermRelease(term: BitwuzlaTerm)

    /**
     * Create a new Bitwuzla instance.
     *
     * The returned instance must be deleted via [bitwuzlaDelete].
     *
     * @return A pointer to the created Bitwuzla instance.
     *
     * @see bitwuzlaDelete
     */
    @JvmStatic
    external fun bitwuzlaNew(termManager: BitwuzlaTermManager, options: BitwuzlaOptions): Bitwuzla

    /**
     * Delete a Bitwuzla instance.
     *
     * The given instance must have been created via [bitwuzlaNew].
     *
     * @param bitwuzla The Bitwuzla instance to delete.
     *
     * @see bitwuzlaNew
     */
    @JvmStatic
    external fun bitwuzlaDelete(bitwuzla: Bitwuzla)

    /**
     * Create new Bitwuzla options.
     *
     * @see [bitwuzlaOptionsDelete]
     * */
    @JvmStatic
    external fun bitwuzlaOptionsNew(): BitwuzlaOptions

    /**
     * Delete Bitwuzla options instance.
     * */
    @JvmStatic
    external fun bitwuzlaOptionsDelete(options: BitwuzlaOptions)

    /**
     * Set option.
     *
     * @param options The Bitwuzla options instance.
     * @param option The option.
     * @param value The option value.
     *
     * @see BitwuzlaOption
     */
    @JvmStatic
    fun bitwuzlaSetOption(options: BitwuzlaOptions, option: BitwuzlaOption, value: Int) =
        bitwuzlaSetOption(options, option.ordinal, value.toLong())

    @JvmStatic
    external fun bitwuzlaSetOption(options: BitwuzlaOptions, option: Int, value: Long)

    /**
     * Set option value for string options.
     *
     * @param options The Bitwuzla options instance.
     * @param option The option.
     * @param value The option string value.
     *
     * @see BitwuzlaOption
     */
    @JvmStatic
    fun bitwuzlaSetOptionMode(options: BitwuzlaOptions, option: BitwuzlaOption, value: String) =
        bitwuzlaSetOptionMode(options, option.ordinal, value)

    @JvmStatic
    external fun bitwuzlaSetOptionMode(optsions: BitwuzlaOptions, option: Int, value: String)

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
    fun bitwuzlaGetOption(options: BitwuzlaOptions, option: BitwuzlaOption): Int =
        bitwuzlaGetOption(options, option.ordinal).toInt()

    @JvmStatic
    external fun bitwuzlaGetOption(options: BitwuzlaOptions, option: Int): Long

    /**
     * Get the current value of an option as a string if option can be configured
     * via a string value.
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
    fun bitwuzlaGetOptionMode(options: BitwuzlaOptions, option: BitwuzlaOption): String =
        bitwuzlaGetOptionMode(options, option.ordinal)

    @JvmStatic
    external fun bitwuzlaGetOptionMode(options: BitwuzlaOptions, option: Int): String

    /**
     * Create an array sort.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param index The index sort of the array sort.
     * @param element The element sort of the array sort.
     *
     * @return An array sort which maps sort `index` to sort `element`.
     *
     * @see bitwuzlaSortIsArray
     * @see bitwuzlaSortArrayGetIndex
     * @see bitwuzlaSortArrayGetElement
     * @see bitwuzlaTermIsArray
     * @see bitwuzlaTermArrayGetIndexSort
     * @see bitwuzlaTermArrayGetElementSort
     */
    @JvmStatic
    external fun bitwuzlaMkArraySort(
        termManager: BitwuzlaTermManager,
        index: BitwuzlaSort,
        element: BitwuzlaSort
    ): BitwuzlaSort


    /**
     * Create a Boolean sort.
     *
     * @param termManager The Bitwuzla term manager instance.
     *
     * @return A Boolean sort.
     */
    @JvmStatic
    external fun bitwuzlaMkBoolSort(termManager: BitwuzlaTermManager): BitwuzlaSort

    /**
     * Create a bit-vector sort of given size.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param size The size of the bit-vector sort.
     *
     * @return A bit-vector sort of given size.
     *
     * @see bitwuzlaSortIsBv
     * @see bitwuzlaSortBvGetSize
     * @see bitwuzlaTermIsBv
     * @see bitwuzlaTermBvGetSize
     */
    @JvmStatic
    external fun bitwuzlaMkBvSort(termManager: BitwuzlaTermManager, size: Long): BitwuzlaSort

    /**
     * Create a floating-point sort of given exponent and significand size.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param expSize The size of the exponent.
     * @param sigSize The size of the significand (including sign bit).
     *
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
    external fun bitwuzlaMkFpSort(termManager: BitwuzlaTermManager, expSize: Long, sigSize: Long): BitwuzlaSort

    /**
     * Create a function sort.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param arity The number of arguments to the function.
     * @param domain The domain sorts (the sorts of the arguments). The number of
     * sorts in this vector must match `arity`.
     * @param codomain The codomain sort (the sort of the return value).
     *
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
    external fun bitwuzlaMkFunSort(
        termManager: BitwuzlaTermManager,
        arity: Long,
        domain: BitwuzlaSortArray,
        codomain: BitwuzlaSort
    ): BitwuzlaSort

    /**
     * Create a Roundingmode sort.
     *
     * @param termManager The Bitwuzla term manager instance.
     *
     * @return A Roundingmode sort.
     *
     * @see bitwuzlaSortIsRm
     * @see bitwuzlaTermIsRm
     */
    @JvmStatic
    external fun bitwuzlaMkRmSort(termManager: BitwuzlaTermManager): BitwuzlaSort

    /**
     * Create an uninterpreted sort.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param name The sort name.
     *
     * @return An Uninterpreted sort.
     *
     * @see bitwuzlaSortIsUninterpreted
     */
    @JvmStatic
    external fun bitwuzlaMkUninterpretedSort(termManager: BitwuzlaTermManager, name: String): BitwuzlaSort

    /**
     * Create a true value.
     *
     * Note: This creates a bit-vector value 1 of size 1.
     *
     * @param termManager The Bitwuzla term manager instance.
     *
     * @return A term representing the bit-vector value 1 of size 1.
     */
    @JvmStatic
    external fun bitwuzlaMkTrue(termManager: BitwuzlaTermManager): BitwuzlaTerm

    /**
     * Create a false value.
     *
     * Note: This creates a bit-vector value 0 of size 1.
     *
     * @param termManager The Bitwuzla term manager instance.
     *
     * @return A term representing the bit-vector value 0 of size 1.
     */
    @JvmStatic
    external fun bitwuzlaMkFalse(termManager: BitwuzlaTermManager): BitwuzlaTerm

    /**
     * Create a bit-vector value zero.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the bit-vector value 0 of given sort.
     *
     * @see bitwuzlaMkBvSort
     */
    @JvmStatic
    external fun bitwuzlaMkBvZero(termManager: BitwuzlaTermManager, sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a bit-vector value one.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the bit-vector value 1 of given sort.
     *
     * @see bitwuzlaMkBvSort
     */
    @JvmStatic
    external fun bitwuzlaMkBvOne(termManager: BitwuzlaTermManager, sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a bit-vector value where all bits are set to 1.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the bit-vector value of given sort
     * where all bits are set to 1.
     *
     * @see bitwuzlaMkBvSort
     */
    @JvmStatic
    external fun bitwuzlaMkBvOnes(termManager: BitwuzlaTermManager, sort: BitwuzlaSort): BitwuzlaTerm


    /**
     * Create a bit-vector minimum signed value.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the bit-vector value of given sort where the MSB
     * is set to 1 and all remaining bits are set to 0.
     *
     * @see bitwuzlaMkBvSort
     */
    @JvmStatic
    external fun bitwuzlaMkBvMinSigned(termManager: BitwuzlaTermManager, sort: BitwuzlaSort): BitwuzlaTerm


    /**
     * Create a bit-vector maximum signed value.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the bit-vector value of given sort where the MSB
     * is set to 0 and all remaining bits are set to 1.
     *
     * @see bitwuzlaMkBvSort
     */
    @JvmStatic
    external fun bitwuzlaMkBvMaxSigned(termManager: BitwuzlaTermManager, sort: BitwuzlaSort): BitwuzlaTerm


    /**
     * Create a floating-point positive zero value (SMT-LIB: `+zero`).
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the floating-point positive zero value of given
     * floating-point sort.
     *
     * @see bitwuzlaMkFpSort
     */
    @JvmStatic
    external fun bitwuzlaMkFpPosZero(termManager: BitwuzlaTermManager, sort: BitwuzlaSort): BitwuzlaTerm


    /**
     * Create a floating-point negative zero value (SMT-LIB: `-zero`).
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the floating-point negative zero value of given
     * floating-point sort.
     *
     * @see bitwuzlaMkFpSort
     */
    @JvmStatic
    external fun bitwuzlaMkFpNegZero(termManager: BitwuzlaTermManager, sort: BitwuzlaSort): BitwuzlaTerm


    /**
     * Create a floating-point positive infinity value (SMT-LIB: `+oo`).
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the floating-point positive infinity value of
     * given floating-point sort.
     *
     * @see bitwuzlaMkFpSort
     */
    @JvmStatic
    external fun bitwuzlaMkFpPosInf(termManager: BitwuzlaTermManager, sort: BitwuzlaSort): BitwuzlaTerm


    /**
     * Create a floating-point negative infinity value (SMT-LIB: `-oo`).
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the floating-point negative infinity value of
     * given floating-point sort.
     *
     * @see bitwuzlaMkFpSort
     */
    @JvmStatic
    external fun bitwuzlaMkFpNegInf(termManager: BitwuzlaTermManager, sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a floating-point NaN value.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param sort The sort of the value.
     *
     * @return A term representing the floating-point NaN value of given
     * floating-point sort.
     *
     * @see bitwuzlaMkFpSort
     */
    @JvmStatic
    external fun bitwuzlaMkFpNan(termManager: BitwuzlaTermManager, sort: BitwuzlaSort): BitwuzlaTerm

    /**
     * Create a bit-vector value from its string representation.
     *
     * Parameter `base` determines the base of the string representation (e.g. 2, 10, 16).
     *
     * Note: Given value must fit into a bit-vector of given size (sort).
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param sort The sort of the value.
     * @param value A string representing the value.
     * @param base The base in which the string is given.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VAL], representing the bit-vector value
     * of given sort.
     *
     * @see bitwuzlaMkBvSort
     */
    @JvmStatic
    external fun bitwuzlaMkBvValue(
        termManager: BitwuzlaTermManager,
        sort: BitwuzlaSort,
        value: String,
        base: Byte
    ): BitwuzlaTerm

    /**
     * Create a bit-vector value from its unsigned integer representation.
     *
     * Note: If given value does not fit into a bit-vector of given size (sort),
     * the value is truncated to fit.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param sort The sort of the value.
     * @param value The unsigned integer representation of the bit-vector value.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VAL], representing the bit-vector value
     * of given sort.
     *
     * @see bitwuzlaMkBvSort
     */
    @JvmStatic
    external fun bitwuzlaMkBvValueUint64(
        termManager: BitwuzlaTermManager,
        sort: BitwuzlaSort,
        value: Long
    ): BitwuzlaTerm

    /**
     * Create Bv value of width [bvWidth] using bits from [value] array.
     * Array should match Bv bits representation. array[0] = bv[0:63], array[1] = bv[64:128], ...
     * */
    @JvmStatic
    external fun bitwuzlaMkBvValueUint64Array(bitwuzla: Bitwuzla, bvWidth: Long, value: LongArray): BitwuzlaTerm

    /**
     * Create a floating-point value from its IEEE 754 standard representation
     * given as three bit-vector values representing the sign bit, the exponent and
     * the significand.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param bvSign The sign bit.
     * @param bvExponent The exponent bit-vector value.
     * @param bvSignificand The significand bit-vector value.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VAL], representing the floating-point
     * value.
     */
    @JvmStatic
    external fun bitwuzlaMkFpValue(
        termManager: BitwuzlaTermManager,
        bvSign: BitwuzlaTerm,
        bvExponent: BitwuzlaTerm,
        bvSignificand: BitwuzlaTerm
    ): BitwuzlaTerm

    /**
     * Create a floating-point value from its real representation, given as a
     * decimal string, with respect to given rounding mode.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param sort The sort of the value.
     * @param rm The rounding mode.
     * @param value The decimal string representing a real value.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VAL], representing the floating-point
     *         value of given sort.
     */
    @JvmStatic
    external fun bitwuzlaMkFpValueFromReal(
        termManager: BitwuzlaTermManager,
        sort: BitwuzlaSort,
        rm: BitwuzlaTerm,
        value: String
    ): BitwuzlaTerm

    /**
     * Create a floating-point value from its rational representation, given as
     * two decimal strings representing the numerator and denominator, with respect
     * to given rounding mode.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param sort The sort of the value.
     * @param rm The rounding mode.
     * @param numerator The decimal string representing the numerator.
     * @param denominator The decimal string representing the denominator.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VAL], representing the floating-point
     *         value of given sort.
     */
    @JvmStatic
    external fun bitwuzlaMkFpValueFromRational(
        termManager: BitwuzlaTermManager,
        sort: BitwuzlaSort,
        rm: BitwuzlaTerm,
        numerator: String,
        denominator: String
    ): BitwuzlaTerm

    /**
     * Create a rounding mode value.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param rm The rounding mode value.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VAL], representing the rounding mode
     * value.
     *
     * @see BitwuzlaRoundingMode
     */
    @JvmStatic
    fun bitwuzlaMkRmValue(termManager: BitwuzlaTermManager, rm: BitwuzlaRoundingMode): BitwuzlaTerm =
        bitwuzlaMkRmValue(termManager, rm.ordinal)

    @JvmStatic
    external fun bitwuzlaMkRmValue(termManager: BitwuzlaTermManager, rm: Int): BitwuzlaTerm

    /**
     * Create a term of given kind with one argument term.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param kind The operator kind.
     * @param arg The argument to the operator.
     *
     * @return A term representing an operation of given kind.
     *
     * @see  BitwuzlaKind
     */
    @JvmStatic
    fun bitwuzlaMkTerm1(termManager: BitwuzlaTermManager, kind: BitwuzlaKind, arg: BitwuzlaTerm): BitwuzlaTerm =
        bitwuzlaMkTerm1(termManager, kind.ordinal, arg)

    @JvmStatic
    external fun bitwuzlaMkTerm1(termManager: BitwuzlaTermManager, kind: Int, arg: BitwuzlaTerm): BitwuzlaTerm

    /**
     * Create a term of given kind with two argument terms.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param kind The operator kind.
     * @param arg0 The first argument to the operator.
     * @param arg1 The second argument to the operator.
     *
     * @return A term representing an operation of given kind.
     *
     * @see  BitwuzlaKind
     */
    @JvmStatic
    fun bitwuzlaMkTerm2(
        termManager: BitwuzlaTermManager,
        kind: BitwuzlaKind,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm
    ): BitwuzlaTerm = bitwuzlaMkTerm2(termManager, kind.ordinal, arg0, arg1)

    @JvmStatic
    external fun bitwuzlaMkTerm2(
        termManager: BitwuzlaTermManager,
        kind: Int,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm
    ): BitwuzlaTerm

    /**
     * Create a term of given kind with three argument terms.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param kind The operator kind.
     * @param arg0 The first argument to the operator.
     * @param arg1 The second argument to the operator.
     * @param arg2 The third argument to the operator.
     *
     * @return A term representing an operation of given kind.
     *
     * @see  BitwuzlaKind
     */
    @JvmStatic
    fun bitwuzlaMkTerm3(
        termManager: BitwuzlaTermManager,
        kind: BitwuzlaKind,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        arg2: BitwuzlaTerm
    ): BitwuzlaTerm = bitwuzlaMkTerm3(termManager, kind.ordinal, arg0, arg1, arg2)

    @JvmStatic
    external fun bitwuzlaMkTerm3(
        termManager: BitwuzlaTermManager,
        kind: Int,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        arg2: BitwuzlaTerm
    ): BitwuzlaTerm

    /**
     * Create a term of given kind with the given argument terms.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param kind The operator kind.
     * @param args The argument terms.
     *
     * @return A term representing an operation of given kind.
     *
     * @see  BitwuzlaKind
     */
    @JvmStatic
    fun bitwuzlaMkTerm(termManager: BitwuzlaTermManager, kind: BitwuzlaKind, args: BitwuzlaTermArray): BitwuzlaTerm =
        bitwuzlaMkTerm(termManager, kind.ordinal, args)

    @JvmStatic
    external fun bitwuzlaMkTerm(termManager: BitwuzlaTermManager, kind: Int, args: BitwuzlaTermArray): BitwuzlaTerm

    /**
     * Create an indexed term of given kind with one argument term and one index.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param kind The operator kind.
     * @param arg The argument term.
     * @param idx The index.
     *
     * @return A term representing an indexed operation of given kind.
     *
     * @see  BitwuzlaKind
     */
    @JvmStatic
    fun bitwuzlaMkTerm1Indexed1(
        termManager: BitwuzlaTermManager,
        kind: BitwuzlaKind,
        arg: BitwuzlaTerm,
        idx: Int
    ): BitwuzlaTerm = bitwuzlaMkTerm1Indexed1(termManager, kind.ordinal, arg, idx.toLong())

    @JvmStatic
    external fun bitwuzlaMkTerm1Indexed1(
        termManager: BitwuzlaTermManager,
        kind: Int,
        arg: BitwuzlaTerm,
        idx: Long
    ): BitwuzlaTerm


    /**
     * Create an indexed term of given kind with one argument term and two indices.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param kind The operator kind.
     * @param arg The argument term.
     * @param idx0 The first index.
     * @param idx1 The second index.
     *
     * @return A term representing an indexed operation of given kind.
     *
     * @see  BitwuzlaKind
     */
    @JvmStatic
    fun bitwuzlaMkTerm1Indexed2(
        termManager: BitwuzlaTermManager,
        kind: BitwuzlaKind,
        arg: BitwuzlaTerm,
        idx0: Int,
        idx1: Int
    ): BitwuzlaTerm = bitwuzlaMkTerm1Indexed2(termManager, kind.ordinal, arg, idx0.toLong(), idx1.toLong())

    @JvmStatic
    external fun bitwuzlaMkTerm1Indexed2(
        termManager: BitwuzlaTermManager,
        kind: Int,
        arg: BitwuzlaTerm,
        idx0: Long,
        idx1: Long
    ): BitwuzlaTerm


    /**
     * Create an indexed term of given kind with two argument terms and one index.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param kind The operator kind.
     * @param arg0 The first argument term.
     * @param arg1 The second argument term.
     * @param idx The index.
     *
     * @return A term representing an indexed operation of given kind.
     *
     * @see  BitwuzlaKind
     */
    @JvmStatic
    fun bitwuzlaMkTerm2Indexed1(
        termManager: BitwuzlaTermManager,
        kind: BitwuzlaKind,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        idx: Int
    ): BitwuzlaTerm = bitwuzlaMkTerm2Indexed1(termManager, kind.ordinal, arg0, arg1, idx.toLong())

    @JvmStatic
    external fun bitwuzlaMkTerm2Indexed1(
        termManager: BitwuzlaTermManager,
        kind: Int,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        idx: Long
    ): BitwuzlaTerm


    /**
     * Create an indexed term of given kind with two argument terms and two indices.
     *
     * @param termManager The Bitwuzla term manager instance.
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
    @Suppress("LongParameterList")
    @JvmStatic
    fun bitwuzlaMkTerm2Indexed2(
        termManager: BitwuzlaTermManager,
        kind: BitwuzlaKind,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        idx0: Int,
        idx1: Int
    ): BitwuzlaTerm = bitwuzlaMkTerm2Indexed2(termManager, kind.ordinal, arg0, arg1, idx0.toLong(), idx1.toLong())

    @Suppress("LongParameterList")
    @JvmStatic
    external fun bitwuzlaMkTerm2Indexed2(
        termManager: BitwuzlaTermManager,
        kind: Int,
        arg0: BitwuzlaTerm,
        arg1: BitwuzlaTerm,
        idx0: Long,
        idx1: Long
    ): BitwuzlaTerm

    /**
     * Create an indexed term of given kind with the given argument terms and
     * indices.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param kind The operator kind.
     * @param args The argument terms.
     * @param indices The indices.
     *
     * @return A term representing an indexed operation of given kind.
     */
    @JvmStatic
    fun bitwuzlaMkTermIndexed(
        termManager: BitwuzlaTermManager,
        kind: BitwuzlaKind,
        args: BitwuzlaTermArray,
        indices: LongArray
    ): BitwuzlaTerm = bitwuzlaMkTermIndexed(termManager, kind.ordinal, args, indices)

    @JvmStatic
    external fun bitwuzlaMkTermIndexed(
        termManager: BitwuzlaTermManager,
        kind: Int,
        args: BitwuzlaTermArray,
        indices: LongArray
    ): BitwuzlaTerm

    /**
     * Create a (first-order) constant of given sort with given symbol.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param sort The sort of the constant.
     * @param symbol The symbol of the constant.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_CONST], representing the constant.
     *
     * @see bitwuzlaMkArraySort
     * @see bitwuzlaMkBoolSort
     * @see bitwuzlaMkBvSort
     * @see bitwuzlaMkFpSort
     * @see bitwuzlaMkFunSort
     * @see bitwuzlaMkRmSort
     */
    @JvmStatic
    external fun bitwuzlaMkConst(termManager: BitwuzlaTermManager, sort: BitwuzlaSort, symbol: String): BitwuzlaTerm

    /**
     * Create a one-dimensional constant array of given sort, initialized with
     * given value.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param sort The sort of the array.
     * @param value The term to initialize the elements of the array with.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_CONST_ARRAY], representing a constant
     * array of given sort.
     *
     * @see bitwuzlaMkArraySort
     */
    @JvmStatic
    external fun bitwuzlaMkConstArray(
        termManager: BitwuzlaTermManager,
        sort: BitwuzlaSort,
        value: BitwuzlaTerm
    ): BitwuzlaTerm

    /**
     * Create a variable of given sort with given symbol.
     *
     * Note: This creates a variable to be bound by quantifiers or lambdas.
     *
     * @param termManager The Bitwuzla term manager instance.
     * @param sort The sort of the variable.
     * @param symbol The symbol of the variable.
     *
     * @return A term of kind [BitwuzlaKind.BITWUZLA_KIND_VAR], representing the variable.
     *
     * @see bitwuzlaMkBoolSort
     * @see bitwuzlaMkBvSort
     * @see bitwuzlaMkFpSort
     * @see bitwuzlaMkFunSort
     * @see bitwuzlaMkRmSort
     */
    @JvmStatic
    external fun bitwuzlaMkVar(termManager: BitwuzlaTermManager, sort: BitwuzlaSort, symbol: String): BitwuzlaTerm

    /**
     * Push context levels.
     *
     * Requires that incremental solving has been enabled via
     * [bitwuzlaSetOption].
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param nlevels The number of context levels to push.
     *
     * @see bitwuzlaSetOption
     * @see BitwuzlaOption.BITWUZLA_OPT_INCREMENTAL
     */
    @JvmStatic
    external fun bitwuzlaPush(bitwuzla: Bitwuzla, nlevels: Long)

    /**
     * Pop context levels.
     *
     * Requires that incremental solving has been enabled via
     * [bitwuzlaSetOption].
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param nlevels The number of context levels to pop.
     *
     * @see bitwuzlaSetOption
     * @see BitwuzlaOption.BITWUZLA_OPT_INCREMENTAL
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
     * Determine if an assumption is an unsat assumption.
     *
     * Unsat assumptions are assumptions that force an input formula to become
     * unsatisfiable. Unsat assumptions handling in Bitwuzla is analogous to
     * failed assumptions in MiniSAT.
     *
     * Requires that incremental solving has been enabled via
     * [bitwuzlaSetOption].
     *
     * Requires that the last [bitwuzlaCheckSat] query returned
     * [BitwuzlaResult.BITWUZLA_UNSAT].
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param term The assumption to check for.
     *
     * @return True if given assumption is an unsat assumption.
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
     * Requires that incremental solving has been enabled via
     * [bitwuzlaSetOption].
     *
     * Requires that the last [bitwuzlaCheckSat] query returned
     * [BitwuzlaResult.BITWUZLA_UNSAT].
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return An array with unsat assumptions.
     *
     * @see bitwuzlaSetOption
     * @see bitwuzlaCheckSat
     * @see BitwuzlaOption.BITWUZLA_OPT_INCREMENTAL
     */
    @JvmStatic
    external fun bitwuzlaGetUnsatAssumptions(bitwuzla: Bitwuzla): BitwuzlaTermArray


    /**
     * Get the set unsat core (unsat assertions).
     *
     * The unsat core consists of the set of assertions that force an input formula
     * to become unsatisfiable.
     *
     * Requires that the last [bitwuzlaCheckSat] query returned
     * [BitwuzlaResult.BITWUZLA_UNSAT].
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return An array with unsat assertions.
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
     */
    @JvmStatic
    external fun bitwuzlaSimplify(bitwuzla: Bitwuzla)

    /**
     * Check satisfiability of current input formula.
     *
     * An input formula consists of assertions added via [bitwuzlaAssert].
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return [BitwuzlaResult.BITWUZLA_SAT] if the input formula is satisfiable and
     * [BitwuzlaResult.BITWUZLA_UNSAT] if it is unsatisfiable, and [BitwuzlaResult.BITWUZLA_UNKNOWN]
     * when neither satisfiability nor unsatisfiability was determined.
     * This can happen when `bitwuzla` was terminated via a termination
     * callback.
     *
     * @see bitwuzlaAssert
     * @see bitwuzlaSetOption
     * @see BitwuzlaOption.BITWUZLA_OPT_INCREMENTAL
     * @see BitwuzlaResult
     */
    @JvmStatic
    fun bitwuzlaCheckSatResult(bitwuzla: Bitwuzla): BitwuzlaResult =
        bitwuzlaCheckSat(bitwuzla).let { BitwuzlaResult.fromValue(it) }

    @JvmStatic
    external fun bitwuzlaCheckSat(bitwuzla: Bitwuzla): Int

    /**
     * Check satisfiability of current input formula.
     *
     * An input formula consists of assertions added via [bitwuzlaAssert].
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param assumptions
     *
     * @return [BitwuzlaResult.BITWUZLA_SAT] if the input formula is satisfiable and
     * [BitwuzlaResult.BITWUZLA_UNSAT] if it is unsatisfiable, and [BitwuzlaResult.BITWUZLA_UNKNOWN]
     * when neither satisfiability nor unsatisfiability was determined.
     * This can happen when `bitwuzla` was terminated via a termination
     * callback.
     *
     * @see bitwuzlaAssert
     * @see bitwuzlaSetOption
     * @see BitwuzlaOption.BITWUZLA_OPT_INCREMENTAL
     * @see BitwuzlaResult
     */
    @JvmStatic
    fun bitwuzlaCheckSatAssumingResult(bitwuzla: Bitwuzla, assumptions: BitwuzlaTermArray): BitwuzlaResult =
        bitwuzlaCheckSatAssuming(bitwuzla, assumptions).let { BitwuzlaResult.fromValue(it) }

    @JvmStatic
    external fun bitwuzlaCheckSatAssuming(bitwuzla: Bitwuzla, assumptions: BitwuzlaTermArray): Int

    /**
     * Check formula satisfiability with timeout.
     *
     * @param timeout Timeout in milliseconds.
     *
     * @see bitwuzlaCheckSat
     * */
    @JvmStatic
    fun bitwuzlaCheckSatTimeoutResult(bitwuzla: Bitwuzla, timeout: Long): BitwuzlaResult =
        bitwuzlaCheckSatTimeout(bitwuzla, timeout).let { BitwuzlaResult.fromValue(it) }

    @JvmStatic
    external fun bitwuzlaCheckSatTimeout(bitwuzla: Bitwuzla, timeout: Long): Int

    /**
     * Check formula satisfiability with timeout.
     *
     * @param timeout Timeout in milliseconds.
     * @param assumptions
     *
     * @see bitwuzlaCheckSatAssuming
     * */
    @JvmStatic
    fun bitwuzlaCheckSatAssumingTimeoutResult(
        bitwuzla: Bitwuzla,
        assumptions: BitwuzlaTermArray,
        timeout: Long
    ): BitwuzlaResult =
        bitwuzlaCheckSatAssumingTimeout(bitwuzla, assumptions, timeout).let { BitwuzlaResult.fromValue(it) }

    @JvmStatic
    external fun bitwuzlaCheckSatAssumingTimeout(bitwuzla: Bitwuzla, assumptions: BitwuzlaTermArray, timeout: Long): Int

    /**
     * Cancel currently performing check.
     *
     * Note: in current implementation check is cancelable only
     * if it was a check with timeout [bitwuzlaCheckSatTimeout].
     * */
    @JvmStatic
    external fun bitwuzlaForceTerminate(bitwuzla: Bitwuzla)

    /**
     * Get a term representing the model value of a given term.
     *
     * Requires that the last [bitwuzlaCheckSat] query returned
     * [BitwuzlaResult.BITWUZLA_SAT].
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param term The term to query a model value for.
     *
     * @return A term representing the model value of term `term`.
     *
     * @see bitwuzlaCheckSat
     */
    @JvmStatic
    external fun bitwuzlaGetValue(bitwuzla: Bitwuzla, term: BitwuzlaTerm): BitwuzlaTerm

    /**
     * Get bool term value. Only safe if [bitwuzlaTermIsBool] is true for [term].
     * */
    @JvmStatic
    external fun bitwuzlaTermValueGetBool(term: BitwuzlaTerm): Boolean

    /**
     * Get bv const bits. Only safe if [bitwuzlaTermIsBvValue] is true for [term] and
     * bv width <= 64.
     * */
    @JvmStatic
    external fun bitwuzlaTermValueGetBvUint64(term: BitwuzlaTerm): Long

    /**
     * Get bv const bits. Only safe if [bitwuzlaTermIsBvValue] is true for [term].
     * Returned array matches Bv bits representation. array[0] = bv[0:63], array[1] = bv[64:127], ...
     * */
    @JvmStatic
    external fun bitwuzlaTermValueGetBvUint64Array(term: BitwuzlaTerm): LongArray

    /**
     * Get fp const bits. Only safe if [bitwuzlaTermIsFpValue] is true for [term] and
     * bits count <= 64.
     * */
    @JvmStatic
    external fun bitwuzlaTermValueGetFpUint64(term: BitwuzlaTerm): Long

    /**
     * Get fp const bits. Only safe if [bitwuzlaTermIsFpValue] is true for [term].
     * Returned array matches Fp bits representation. array[0] = fp[0:63], array[1] = fp[64:127], ...
     * */
    @JvmStatic
    external fun bitwuzlaTermValueGetFpUint64Array(term: BitwuzlaTerm): LongArray

    /**
     * Get rm term value. Only safe if [bitwuzlaTermIsRmValue] is true for [term].
     * */
    @JvmStatic
    fun bitwuzlaTermValueGetRoundingMode(term: BitwuzlaTerm): BitwuzlaRoundingMode =
        BitwuzlaRoundingMode.fromValue(bitwuzlaTermValueGetRm(term))

    @JvmStatic
    external fun bitwuzlaTermValueGetRm(term: BitwuzlaTerm): Int

    /**
     * Substitute a set of keys with their corresponding values in the given term.
     *
     * @param term The term in which the keys are to be substituted.
     * @param mapKeys The keys.
     * @param mapValues The mapped values.
     *
     * @return The resulting term from this substitution.
     */
    @JvmStatic
    external fun bitwuzlaSubstituteTerm(
        term: BitwuzlaTerm,
        mapKeys: BitwuzlaTermArray,
        mapValues: BitwuzlaTermArray
    ): BitwuzlaTerm

    /**
     * Substitute a set of keys with their corresponding values in the set of given
     * terms.
     *
     * @param terms The terms in which the keys are to be substituted.
     * @param mapKeys The keys.
     * @param mapValues The mapped values.
     *
     * @return The resulting terms from this substitution.
     */
    @JvmStatic
    external fun bitwuzlaSubstituteTerms(
        terms: BitwuzlaTermArray,
        mapKeys: BitwuzlaTermArray,
        mapValues: BitwuzlaTermArray
    ): BitwuzlaTermArray

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
     * Get the kind of a term.
     *
     * @param term The term.
     *
     * @return The kind of the given term.
     *
     * @see BitwuzlaKind
     */
    @JvmStatic
    fun bitwuzlaTermGetBitwuzlaKind(term: BitwuzlaTerm): BitwuzlaKind =
        bitwuzlaTermGetKind(term).let { BitwuzlaKind.fromValue(it) }

    @JvmStatic
    external fun bitwuzlaTermGetKind(term: BitwuzlaTerm): Int

    /**
     * Get the child terms of a term.
     *
     * Returns `null` if given term does not have children.
     *
     * @param term The term.
     *
     * @return The children of `term` as an array of terms.
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
     * @return The children of `term` as an array of terms.
     */
    @JvmStatic
    external fun bitwuzlaTermGetIndices(term: BitwuzlaTerm): LongArray

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
     * Get the name of an uninterpreted sort.
     *
     * @param sort The sort.
     *
     * @return The name of the uninterpreted sort.
     */
    @JvmStatic
    external fun bitwuzlaSortGetUninterpretedSymbol(sort: BitwuzlaSort): String

    /**
     * Determine if a sort is an array sort.
     *
     * @param sort The sort.
     *
     * @return True if `sort` is an array sort.
     */
    @JvmStatic
    external fun bitwuzlaSortIsArray(sort: BitwuzlaSort): Boolean

    /**
     * Determine if a sort is a bool sort.
     *
     * @param sort The sort.
     *
     * @return True if `sort` is a bool sort.
     */
    @JvmStatic
    external fun bitwuzlaSortIsBool(sort: BitwuzlaSort): Boolean

    /**
     * Determine if a sort is a bit-vector sort.
     *
     * @param sort The sort.
     *
     * @return True if `sort` is a bit-vector sort.
     */
    @JvmStatic
    external fun bitwuzlaSortIsBv(sort: BitwuzlaSort): Boolean

    /**
     * Determine if a sort is a floating-point sort.
     *
     * @param sort The sort.
     *
     * @return True if `sort` is a floating-point sort.
     */
    @JvmStatic
    external fun bitwuzlaSortIsFp(sort: BitwuzlaSort): Boolean

    /**
     * Determine if a sort is a function sort.
     *
     * @param sort The sort.
     *
     * @return True if `sort` is a function sort.
     */
    @JvmStatic
    external fun bitwuzlaSortIsFun(sort: BitwuzlaSort): Boolean

    /**
     * Determine if a sort is a Roundingmode sort.
     *
     * @param sort The sort.
     *
     * @return True if `sort` is a Roundingmode sort.
     */
    @JvmStatic
    external fun bitwuzlaSortIsRm(sort: BitwuzlaSort): Boolean

    /**
     * Determine if a sort is an uninterpreted sort.
     *
     * @param sort The sort.
     *
     * @return True if `sort` is an uninterpreted sort.
     */
    @JvmStatic
    external fun bitwuzlaSortIsUninterpreted(sort: BitwuzlaSort): Boolean

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
     * Determine if a term is an indexed term.
     *
     * @param term The term.
     *
     * @return True if `term` is an indexed term.
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
     * The domain sorts are returned as an array of sorts of size `size.
     * Requires that given term is an uninterpreted function, a lambda term, an
     * array store term, or an ite term over function terms.
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
     * @return The symbol of `term`. `null` if no symbol is defined.
     */
    @JvmStatic
    external fun bitwuzlaTermGetSymbol(term: BitwuzlaTerm): String?

    /**
     * Determine if the sorts of two terms are equal.
     *
     * @param term0 The first term.
     * @param term1 The second term.
     *
     * @return True if the sorts of `term0` and `term1` are equal.
     */
    @JvmStatic
    external fun bitwuzlaTermIsEqualSort(term0: BitwuzlaTerm, term1: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is an array term.
     *
     * @param term The term.
     *
     * @return True if `term` is an array term.
     */
    @JvmStatic
    external fun bitwuzlaTermIsArray(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a constant.
     *
     * @param term The term.
     *
     * @return True if `term` is a constant.
     */
    @JvmStatic
    external fun bitwuzlaTermIsConst(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a function.
     *
     * @param term The term.
     *
     * @return True if `term` is a function.
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
     * @return True if `term` is a bit-vector value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBvValue(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point value.
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFpValue(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode value.
     *
     * @param term The term.
     *
     * @return True if `term` is a rounding mode value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRmValue(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bool term.
     *
     * @param term The term.
     *
     * @return True if `term` is a bool term.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBool(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector term.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector term.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBv(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point term.
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point term.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFp(term: BitwuzlaTerm): Boolean


    /**
     * Determine if a term is a rounding mode term.
     *
     * @param term The term.
     *
     * @return True if `term` is a rounding mode term.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRm(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is an uninterpreted sort term.
     *
     * @param term The term.
     *
     * @return True if `term` is an uninterpreted sort term.
     */
    @JvmStatic
    external fun bitwuzlaTermIsUninterpreted(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bool `true` term.
     *
     * @param term The term.
     *
     * @return True if `term` is a bool `true` term.
     */
    @JvmStatic
    external fun bitwuzlaTermIsTrue(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bool `false` term.
     *
     * @param term The term.
     *
     * @return True if `term` is a bool `false` term.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFalse(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector value representing zero.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector zero value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBvValueZero(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector value representing one.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector one value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBvValueOne(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector value with all bits set to one.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector value with all bits set to one.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBvValueOnes(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector minimum signed value.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector value with the most significant bit
     * set to 1 and all other bits set to 0.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBvValueMinSigned(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a bit-vector maximum signed value.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector value with the most significant bit
     * set to 0 and all other bits set to 1.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBvValueMaxSigned(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point positive zero (+zero) value.
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point +zero value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFpValuePosZero(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point value negative zero (-zero).
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point value negative zero.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFpValueNegZero(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point positive infinity (+oo) value.
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point +oo value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFpValuePosInf(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point negative infinity (-oo) value.
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point -oo value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFpValueNegInf(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a floating-point NaN value.
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point NaN value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFpValueNan(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode RNA value.
     *
     * @param term The term.
     *
     * @return True if `term` is a roundindg mode RNA value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRmValueRna(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode RNE value.
     *
     * @param term The term.
     *
     * @return True if `term` is a rounding mode RNE value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRmValueRne(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode RTN value.
     *
     * @param term The term.
     *
     * @return True if `term` is a rounding mode RTN value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRmValueRtn(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode RTP value.
     *
     * @param term The term.
     *
     * @return True if `term` is a rounding mode RTP value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRmValueRtp(term: BitwuzlaTerm): Boolean

    /**
     * Determine if a term is a rounding mode RTZ value.
     *
     * @param term The term.
     *
     * @return True if `term` is a rounding mode RTZ value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRmValueRtz(term: BitwuzlaTerm): Boolean

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
     * @param outputFilePath The file to print the formula to.
     */
    @JvmStatic
    external fun bitwuzlaDumpFormula(bitwuzla: Bitwuzla, format: String, outputFilePath: String)

    /**
     * Print [term].
     * */
    @JvmStatic
    external fun bitwuzlaTermToString(term: BitwuzlaTerm): String

    /**
     * Print [sort].
     * */
    @JvmStatic
    external fun bitwuzlaSortToString(sort: BitwuzlaSort): String

    /**
     * Print rounding mode [rm].
     * */
    @JvmStatic
    fun bitwuzlaRmToString(rm: BitwuzlaRoundingMode): String =
        bitwuzlaRmToString(rm.ordinal)

    @JvmStatic
    external fun bitwuzlaRmToString(rm: Int): String

    /**
     * Print check-sat [result].
     * */
    @JvmStatic
    fun bitwuzlaResultToString(result: BitwuzlaResult): String =
        bitwuzlaResultToString(result.value)

    @JvmStatic
    external fun bitwuzlaResultToString(result: Int): String

    /**
     * Print application [kind].
     * */
    @JvmStatic
    fun bitwuzlaKindToString(kind: BitwuzlaKind): String =
        bitwuzlaKindToString(kind.ordinal)

    @JvmStatic
    external fun bitwuzlaKindToString(kind: Int): String

    /**
     * Print [option] name.
     * */
    @JvmStatic
    fun bitwuzlaOptionToString(option: BitwuzlaOption): String =
        bitwuzlaOptionToString(option.ordinal)

    @JvmStatic
    external fun bitwuzlaOptionToString(option: Int): String
}
