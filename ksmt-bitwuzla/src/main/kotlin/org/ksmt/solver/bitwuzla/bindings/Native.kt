package org.ksmt.solver.bitwuzla.bindings


typealias Bitwuzla = Long
typealias BitwuzlaTerm = Long
typealias BitwuzlaSort = Long
typealias BitwuzlaBitVector = Long

object Native {
    init {
        System.load("/home/sobol/IdeaProjects/ksmt/ksmt-bitwuzla/dist/libbitwuzla.so")
        System.load("/home/sobol/IdeaProjects/ksmt/ksmt-bitwuzla/dist/libbitwuzla_jni.so")
//        System.load("Z:\\IdeaProjects\\ksmt\\ksmt-bitwuzla\\dist\\libbitwuzla.dll")
//        System.load("Z:\\IdeaProjects\\ksmt\\ksmt-bitwuzla\\dist\\libbitwuzla_jni.dll")
//        NativeLibraryLoader.load { os ->
//            when (os) {
//                NativeLibraryLoader.OS.LINUX -> listOf()
//                NativeLibraryLoader.OS.WINDOWS -> TODO()
//                NativeLibraryLoader.OS.MACOS -> TODO()
//            }
//        }

        bitwuzlaInit()
    }

    @JvmStatic
    external fun bitwuzlaInit(): Bitwuzla

    /**
     * Create a new Bitwuzla instance.
     *
     * The returned instance must be deleted via [bitwuzla_delete].
     *
     * @return A pointer to the created Bitwuzla instance.
     *
     * @see bitwuzla_delete
     */
    @JvmStatic
    external fun bitwuzlaNew(): Bitwuzla

    /**
     * Delete a Bitwuzla instance.
     *
     * The given instance must have been created via [bitwuzla_new].
     *
     * @param bitwuzla The Bitwuzla instance to delete.
     *
     * @see bitwuzla_new
     */
    @JvmStatic
    external fun bitwuzlaDelete(bitwuzla: Bitwuzla)

    /**
     * Reset a Bitwuzla instance.
     *
     * This deletes the given instance and creates a new instance in place.
     * The given instance must have been created via [bitwuzla_new].
     *
     * Note: All sorts and terms associated with the given instance are released
     * and thus invalidated.
     *
     * @param bitwuzla The Bitwuzla instance to reset.
     *
     * @see bitwuzla_new
     */
    @JvmStatic
    external fun bitwuzlaReset(bitwuzla: Bitwuzla);

    /**
     * Get copyright information.
     *
     * @param bitwuzla The Bitwuzla instance.
     */
    @JvmStatic
    external fun bitwuzlaCopyright(bitwuzla: Bitwuzla): String

    /**
     * Get version information.
     *
     * @param bitwuzla The Bitwuzla instance.
     */
    @JvmStatic
    external fun bitwuzlaVersion(bitwuzla: Bitwuzla): String

    /**
     * Get git information.
     *
     * @param bitwuzla The Bitwuzla instance.
     */
    @JvmStatic
    external fun bitwuzlaGitId(bitwuzla: Bitwuzla): String

    /**
     * Set option.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param option The option.
     * @param value The option value.
     *
     * @see BitwuzlaOption
     */
    @JvmStatic
    external fun bitwuzlaSetOption(bitwuzla: Bitwuzla, option: Int, value: Int)

    /**
     * Set option value for string options.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param option The option.
     * @param value The option string value.
     *
     * @see BitwuzlaOption
     */
    @JvmStatic
    external fun bitwuzlaSetOptionStr(bitwuzla: Bitwuzla, option: Int, value: String)

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
    @JvmStatic
    external fun bitwuzlaGetOption(bitwuzla: Bitwuzla, option: Int): Int

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
    @JvmStatic
    external fun bitwuzlaGetOptionStr(bitwuzla: Bitwuzla, option: Int): String

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
    @JvmStatic
    external fun bitwuzlaMkArraySort(bitwuzla: Bitwuzla, index: BitwuzlaSort, element: BitwuzlaSort): BitwuzlaSort;


    /**
     * Create a Boolean sort.
     *
     * Note: A Boolean sort is a bit-vector sort of size 1.
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return A Boolean sort.
     */
    @JvmStatic
    external fun bitwuzlaMkBoolSort(bitwuzla: Bitwuzla): BitwuzlaSort;

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
    @JvmStatic
    external fun bitwuzlaMkBvSort(bitwuzla: Bitwuzla, size: Int): BitwuzlaSort;

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
    @JvmStatic
    external fun bitwuzlaMkFpSort(bitwuzla: Bitwuzla, expSize: Int, sigSize: Int): BitwuzlaSort

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
    @JvmStatic
    external fun bitwuzlaMkFunSort(bitwuzla: Bitwuzla, arity: Int, domain: LongArray, codomain: BitwuzlaSort): BitwuzlaSort

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
    @JvmStatic
    external fun bitwuzlaMkRmSort(bitwuzla: Bitwuzla): BitwuzlaSort;

    /**
     * Create a true value.
     *
     * Note: This creates a bit-vector value 1 of size 1.
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return A term representing the bit-vector value 1 of size 1.
     */
    @JvmStatic
    external fun bitwuzlaMkTrue(bitwuzla: Bitwuzla): BitwuzlaTerm;

    /**
     * Create a false value.
     *
     * Note: This creates a bit-vector value 0 of size 1.
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @return A term representing the bit-vector value 0 of size 1.
     */
    @JvmStatic
    external fun bitwuzlaMkFalse(bitwuzla: Bitwuzla): BitwuzlaTerm;

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
    @JvmStatic
    external fun bitwuzlaMkBvZero(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm;

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
    @JvmStatic
    external fun bitwuzlaMkBvOne(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm;

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
    @JvmStatic
    external fun bitwuzlaMkBvOnes(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm;


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
    @JvmStatic
    external fun bitwuzlaMkBvMinSigned(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm;


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
    @JvmStatic
    external fun bitwuzlaMkBvMaxSigned(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm;


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
    @JvmStatic
    external fun bitwuzlaMkFpPosZero(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm


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
    @JvmStatic
    external fun bitwuzlaMkFpNegZero(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm;


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
    @JvmStatic
    external fun bitwuzlaMkFpPosInf(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm;


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
    @JvmStatic
    external fun bitwuzlaMkFpNegInf(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm;

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
    @JvmStatic
    external fun bitwuzlaMkFpNan(bitwuzla: Bitwuzla, sort: BitwuzlaSort): BitwuzlaTerm;

    /**
     * Create a bit-vector value from its string representation.
     *
     * Parameter `base` determines the base of the string representation.
     *
     * Note: Given value must fit into a bit-vector of given size (sort).
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
    @JvmStatic
    external fun bitwuzlaMkBvValue(
            bitwuzla: Bitwuzla,
            sort: BitwuzlaSort,
            value: String,
            base: Int
    ): BitwuzlaTerm

    /**
     * Create a bit-vector value from its unsigned integer representation.
     *
     * Note: If given value does not fit into a bit-vector of given size (sort),
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
    @JvmStatic
    external fun bitwuzlaMkBvValueUint64(bitwuzla: Bitwuzla, sort: BitwuzlaSort, value: Long): BitwuzlaTerm

    /**
     * Create a bit-vector value from its unsigned integer representation.
     *
     * Note: If given value does not fit into a bit-vector of given size (sort),
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
    @JvmStatic
    fun bitwuzlaMkBvValueUint32(bitwuzla: Bitwuzla, sort: BitwuzlaSort, value: Int): BitwuzlaTerm {
        return bitwuzlaMkBvValueUint64(bitwuzla, sort, value.toLong())
    }

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
    @JvmStatic
    external fun bitwuzlaMkFpValue(
            bitwuzla: Bitwuzla,
            bvSign: BitwuzlaTerm,
            bvExponent: BitwuzlaTerm,
            bvSignificand: BitwuzlaTerm
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
    @JvmStatic
    external fun bitwuzlaMkRmValue(bitwuzla: Bitwuzla, rm: Int): BitwuzlaTerm

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
    @JvmStatic
    external fun bitwuzlaMkTerm1(bitwuzla: Bitwuzla, kind: Int, arg: BitwuzlaTerm): BitwuzlaTerm

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
    @JvmStatic
    external fun bitwuzlaMkTerm2(
            bitwuzla: Bitwuzla,
            kind: Int,
            arg0: BitwuzlaTerm,
            arg1: BitwuzlaTerm
    ): BitwuzlaTerm;

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
    @JvmStatic
    external fun bitwuzlaMkTerm3(
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
    @JvmStatic
    external fun bitwuzlaMkTerm(bitwuzla: Bitwuzla, kind: Int, args: LongArray/*Array<BitwuzlaTerm>*/): BitwuzlaTerm

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
    @JvmStatic
    external fun bitwuzlaMkTerm1Indexed1(
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
    @JvmStatic
    external fun bitwuzlaMkTerm1Indexed2(
            bitwuzla: Bitwuzla,
            kind: Int,
            arg: BitwuzlaTerm,
            idx0: Int,
            idx1: Int
    ): BitwuzlaTerm;


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
    @JvmStatic
    external fun bitwuzlaMkTerm2Indexed1(
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
    @JvmStatic
    external fun bitwuzlaMkTerm2Indexed2(
            bitwuzla: Bitwuzla,
            kind: Int,
            arg0: BitwuzlaTerm,
            arg1: BitwuzlaTerm,
            idx0: Int,
            idx1: Int
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
    @JvmStatic
    external fun bitwuzlaMkConst(bitwuzla: Bitwuzla, sort: BitwuzlaSort, symbol: String): BitwuzlaTerm

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
    @JvmStatic
    external fun bitwuzlaMkConstArray(bitwuzla: Bitwuzla, sort: BitwuzlaSort, value: BitwuzlaTerm): BitwuzlaTerm

    /**
     * Create a variable of given sort with given symbol.
     *
     * Note: This creates a variable to be bound by quantifiers or lambdas.
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
    @JvmStatic
    external fun bitwuzlaMkVar(bitwuzla: Bitwuzla, sort: BitwuzlaSort, symbol: String): BitwuzlaTerm;

    /**
     * Push context levels.
     *
     * Requires that incremental solving has been enabled via
     * [bitwuzla_set_option].
     *
     * Note: Assumptions added via this [bitwuzla_assume] are not affected by
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
    @JvmStatic
    external fun bitwuzlaPush(bitwuzla: Bitwuzla, nlevels: Int);

    /**
     * Pop context levels.
     *
     * Requires that incremental solving has been enabled via
     * [bitwuzla_set_option].
     *
     * Note: Assumptions added via this [bitwuzla_assume] are not affected by
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
    @JvmStatic
    external fun bitwuzlaPop(bitwuzla: Bitwuzla, nlevels: Int)

    /**
     * Assert formula.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param term The formula to assert.
     */
    @JvmStatic
    external fun bitwuzlaAssert(bitwuzla: Bitwuzla, term: BitwuzlaTerm);

    /**
     * Assume formula.
     *
     * Requires that incremental solving has been enabled via
     * [bitwuzla_set_option].
     *
     * Note: Assumptions added via this function are not affected by context level
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
    @JvmStatic
    external fun bitwuzlaAssume(bitwuzla: Bitwuzla, term: BitwuzlaTerm)


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
    @JvmStatic
    external fun bitwuzlaGetUnsatAssumptions(bitwuzla: Bitwuzla): LongArray


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
    @JvmStatic
    external fun bitwuzlaGetUnsatCore(bitwuzla: Bitwuzla): LongArray


    /**
     * Assert all added assumptions.
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @see bitwuzla_assume
     */
    @JvmStatic
    external fun bitwuzlaFixateAssumptions(bitwuzla: Bitwuzla);

    /**
     * Reset all added assumptions.
     *
     * @param bitwuzla The Bitwuzla instance.
     *
     * @see bitwuzla_assume
     */
    @JvmStatic
    external fun bitwuzlaResetAssumptions(bitwuzla: Bitwuzla)


    /**
     * Check satisfiability of current input formula.
     *
     * An input formula consists of assertions added via [bitwuzla_assert].
     * The search for a solution can by guided by making assumptions via
     * [bitwuzla_assume].
     *
     * Note: Assertions and assumptions are combined via Boolean and.  Multiple
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
    @JvmStatic
    external fun bitwuzlaCheckSat(bitwuzla: Bitwuzla): Int

    @JvmStatic
    external fun bitwuzlaCheckSatTimeout(bitwuzla: Bitwuzla, timeout: Long): Int

    @JvmStatic
    external fun bitwuzlaForceTerminate(bitwuzla: Bitwuzla): Unit

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
    @JvmStatic
    external fun bitwuzlaGetValue(bitwuzla: Bitwuzla, term: BitwuzlaTerm): BitwuzlaTerm


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
    @JvmStatic
    external fun bitwuzlaGetBvValue(bitwuzla: Bitwuzla, term: BitwuzlaTerm): String

    /**
     * Get string of IEEE 754 standard representation of the current model value of
     * given floating-point term.
     *
     * @param bitwuzla The Bitwuzla instance.
     * @param term The term to query a model value for.
     *
     * @see FpValue
     */
    @JvmStatic
    external fun bitwuzlaGetFpValue(bitwuzla: Bitwuzla, term: BitwuzlaTerm): FpValue

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
    @JvmStatic
    external fun bitwuzlaGetArrayValue(bitwuzla: Bitwuzla, term: BitwuzlaTerm): ArrayValue

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
    @JvmStatic
    external fun bitwuzlaGetFunValue(bitwuzla: Bitwuzla, term: BitwuzlaTerm): FunValue

    /**
     * Compute the hash value for a sort.
     *
     * @param sort The sort.
     *
     * @return The hash value of the sort.
     */
    @JvmStatic
    external fun bitwuzlaSortHash(sort: BitwuzlaSort): Long;

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
    external fun bitwuzlaTermGetChildren(term: BitwuzlaTerm): LongArray;

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
    external fun bitwuzlaTermGetIndices(term: BitwuzlaTerm): IntArray

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
    external fun bitwuzlaSortBvGetSize(sort: BitwuzlaSort): Int

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
    external fun bitwuzlaSortFpGetExpSize(sort: BitwuzlaSort): Int

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
    external fun bitwuzlaSortFpGetSigSize(sort: BitwuzlaSort): Int

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
     * Determine if two sorts are equal.
     *
     * @param sort0 The first sort.
     * @param sort1 The second sort.
     *
     * @return True if the given sorts are equal.
     */
    @JvmStatic
    external fun bitwuzlaSortIsEqual(sort0: BitwuzlaSort, sort1: BitwuzlaSort): Boolean

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
     * Get the associated Bitwuzla instance of a term.
     *
     * @param term The term.
     *
     * @return The associated Bitwuzla instance.
     */
    @JvmStatic
    external fun bitwuzlaTermGetBitwuzla(term: BitwuzlaTerm): Bitwuzla

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
    external fun bitwuzlaTermFunGetDomainSorts(term: BitwuzlaTerm): LongArray

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
    external fun bitwuzlaTermFunGetCodomainSort(term: BitwuzlaTerm): BitwuzlaSort;

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
    external fun bitwuzlaTermBvGetSize(term: BitwuzlaTerm): Int;

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
    external fun bitwuzlaTermFpGetExpSize(term: BitwuzlaTerm): Int;

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
    external fun bitwuzlaTermFpGetSigSize(term: BitwuzlaTerm): Int;

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
    external fun bitwuzlaTermFunGetArity(term: BitwuzlaTerm): Int;

    /**
     * Get the symbol of a term.
     *
     * @param term The term.
     *
     * @return The symbol of `term`. `null` if no symbol is defined.
     */
    @JvmStatic
    external fun bitwuzlaTermGetSymbol(term: BitwuzlaTerm): String?;

    /**
     * Set the symbol of a term.
     *
     * @param term The term.
     * @param symbol The symbol.
     */
    @JvmStatic
    external fun bitwuzlaTermSetSymbol(term: BitwuzlaTerm, symbol: String);

    /**
     * Determine if the sorts of two terms are equal.
     *
     * @param term0 The first term.
     * @param term1 The second term.
     *
     * @return True if the sorts of `term0` and `term1` are equal.
     */
    @JvmStatic
    external fun bitwuzlaTermIsEqualSort(term0: BitwuzlaTerm, term1: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is an array term.
     *
     * @param term The term.
     *
     * @return True if `term` is an array term.
     */
    @JvmStatic
    external fun bitwuzlaTermIsArray(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a constant.
     *
     * @param term The term.
     *
     * @return True if `term` is a constant.
     */
    @JvmStatic
    external fun bitwuzlaTermIsConst(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a function.
     *
     * @param term The term.
     *
     * @return True if `term` is a function.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFun(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a variable.
     *
     * @param term The term.
     *
     * @return True if `term` is a variable.
     */
    @JvmStatic
    external fun bitwuzlaTermIsVar(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a bound variable.
     *
     * @param term The term.
     *
     * @return True if `term` is a variable and bound.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBoundVar(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a value.
     *
     * @param term The term.
     *
     * @return True if `term` is a value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsValue(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a bit-vector value.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBvValue(term: BitwuzlaTerm): Boolean;

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
     * Determine if a term is a bit-vector term.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector term.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBv(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a floating-point term.
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point term.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFp(term: BitwuzlaTerm): Boolean;


    /**
     * Determine if a term is a rounding mode term.
     *
     * @param term The term.
     *
     * @return True if `term` is a rounding mode term.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRm(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a bit-vector value representing zero.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector zero value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBvValueZero(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a bit-vector value representing one.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector one value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBvValueOne(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a bit-vector value with all bits set to one.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector value with all bits set to one.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBvValueOnes(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a bit-vector minimum signed value.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector value with the most significant bit
     * set to 1 and all other bits set to 0.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBvValueMinSigned(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a bit-vector maximum signed value.
     *
     * @param term The term.
     *
     * @return True if `term` is a bit-vector value with the most significant bit
     * set to 0 and all other bits set to 1.
     */
    @JvmStatic
    external fun bitwuzlaTermIsBvValueMaxSigned(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a floating-point positive zero (+zero) value.
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point +zero value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFpValuePosZero(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a floating-point value negative zero (-zero).
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point value negative zero.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFpValueNegZero(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a floating-point positive infinity (+oo) value.
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point +oo value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFpValuePosInf(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a floating-point negative infinity (-oo) value.
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point -oo value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFpValueNegInf(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a floating-point NaN value.
     *
     * @param term The term.
     *
     * @return True if `term` is a floating-point NaN value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsFpValueNan(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a rounding mode RNA value.
     *
     * @param term The term.
     *
     * @return True if `term` is a roundindg mode RNA value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRmValueRna(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a rounding mode RNE value.
     *
     * @param term The term.
     *
     * @return True if `term` is a rounding mode RNE value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRmValueRne(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a rounding mode RTN value.
     *
     * @param term The term.
     *
     * @return True if `term` is a rounding mode RTN value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRmValueRtn(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a rounding mode RTP value.
     *
     * @param term The term.
     *
     * @return True if `term` is a rounding mode RTP value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRmValueRtp(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a rounding mode RTZ value.
     *
     * @param term The term.
     *
     * @return True if `term` is a rounding mode RTZ value.
     */
    @JvmStatic
    external fun bitwuzlaTermIsRmValueRtz(term: BitwuzlaTerm): Boolean;

    /**
     * Determine if a term is a constant array.
     *
     * @param term The term.
     *
     * @return True if `term` is a constant array.
     */
    @JvmStatic
    external fun bitwuzlaTermIsConstArray(term: BitwuzlaTerm): Boolean;

    /**
     * Get bv const bits. Only safe if [bitwuzlaTermIsBvValue] is true for [term]
     * */
    @JvmStatic
    external fun bitwuzlaBvConstNodeGetBits(term: BitwuzlaTerm): BitwuzlaBitVector

    /**
     * Get width (bv size) of bv const bits.
     * */
    @JvmStatic
    external fun bitwuzlaBvBitsGetWidth(bv: BitwuzlaBitVector): Int

    /**
     * Convert bv const bits to uint64.
     * Only safe if [bitwuzlaBvBitsGetWidth] <= 64.
     *
     * Note: doesn't work on Windows because JNA truncates Long to uint32.
     * */
    @JvmStatic
    external fun bitwuzlaBvBitsToUInt64(bv: BitwuzlaBitVector): Long

    /**
     * Convert bv const bits to uint32.
     * Only safe if [bitwuzlaBvBitsGetWidth] <= 32.
     * */
    @JvmStatic
    fun bitwuzlaBvBitsToUInt32(bv: BitwuzlaBitVector): Int {
        return bitwuzlaBvBitsToUInt64(bv).toInt();
    }

    /**
     * Get a single bit (0 or 1) from bv const bits.
     * Only safe if 0 <= [pos] < [bitwuzlaBvBitsGetWidth].
     * */
    @JvmStatic
    external fun bitwuzlaBvBitsGetBit(bv: BitwuzlaBitVector, pos: Int): Int

    /**
     * Get fp const bits. Only safe if [bitwuzlaTermIsFpValue] is true for [term]
     * */
    @JvmStatic
    external fun bitwuzlaFpConstNodeGetBits(bitwuzla: Bitwuzla, term: BitwuzlaTerm): BitwuzlaBitVector



}
