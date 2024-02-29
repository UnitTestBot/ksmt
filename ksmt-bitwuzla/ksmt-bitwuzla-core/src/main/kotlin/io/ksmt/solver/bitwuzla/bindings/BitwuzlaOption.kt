package io.ksmt.solver.bitwuzla.bindings

/**
 * The configuration options supported by Bitwuzla.
 *
 * Options that list string values can be configured via
 * [Native.bitwuzlaSetOptionMode]. Options with integer configuration values are
 * configured via [Native.bitwuzlaSetOption].
 *
 * For all options, the current configuration value can be queried via
 * [Native.bitwuzlaGetOption].
 * Options with string configuration values internally represent these
 * values as enum values.
 * For these options, [Native.bitwuzlaGetOption] will return such an enum value.
 * Use [Native.bitwuzlaGetOptionMode] to query enum options for the corresponding
 * string representation.
 */
enum class BitwuzlaOption {
    /* ----------------- General Options -------------------------------------- */
    /**  Log level.
     *
     * Values:
     *  * An unsigned integer value. [default: 0]
     */
    BITWUZLA_OPT_LOG_LEVEL,

    /**  Model generation.
     *
     * Values:
     *  * 1: enable, generate model for assertions only
     *  * 2: enable, generate model for all created terms
     *  * 0: disable [default]
     *
     * @note This option cannot be enabled in combination with option
     *       `::EVALUE(PP_UNCONSTRAINED_OPTIMIZATION`.
     */
    BITWUZLA_OPT_PRODUCE_MODELS,

    /**  Unsat assumptions generation.
     *
     * Values:
     *  * 1: enable
     *  * 0: disable [default]
     */
    BITWUZLA_OPT_PRODUCE_UNSAT_ASSUMPTIONS,

    /**  Unsat core generation.
     *
     * Values:
     *  * 1: enable
     *  * 0: disable [default]
     */
    BITWUZLA_OPT_PRODUCE_UNSAT_CORES,

    /**  Seed for random number generator.
     *
     * Values:
     *  * An unsigned integer value. [default: 0]
     */
    BITWUZLA_OPT_SEED,

    /**  Verbosity level.
     *
     * Values:
     *  * An unsigned integer value <= 4. [default: 0]
     */
    BITWUZLA_OPT_VERBOSITY,

    /**  Time limit in milliseconds per satisfiability check.
     *
     * Values:
     *  * An unsigned integer for the time limit in milliseconds. [default: 0]
     */
    BITWUZLA_OPT_TIME_LIMIT_PER,

    /**   Memory limit in MB.
     *
     * Values:
     *  * An unsigned integer for the memory limit in MB. [default: 0]
     */
    BITWUZLA_OPT_MEMORY_LIMIT,

    /* ---------------- Bitwuzla-specific Options ----------------------------- */

    /**  Configure the bit-vector solver engine.
     *
     * Values:
     *  * bitblast: The classical bit-blasting approach. [default]
     *  * prop: Propagation-based local search (sat only).
     *  * preprop: Sequential portfolio combination of bit-blasting and
     *                 propagation-based local search.
     */
    BITWUZLA_OPT_BV_SOLVER,

    /**  Rewrite level.
     *
     * Values:
     * * 0: no rewriting
     * * 1: term level rewriting
     * * 2: term level rewriting and basic preprocessing
     * * 3: term level rewriting and full preprocessing [default]
     *
     * @note Configuring the rewrite level after terms have been created
     *       is not allowed.
     *
     *  @warning This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_REWRITE_LEVEL,

    /**  Configure the SAT solver engine.
     *
     * Values:
     *  * cadical:
     *    [CaDiCaL](https://github.com/arminbiere/cadical) [default]
     *  * cms:
     *    [CryptoMiniSat](https://github.com/msoos/cryptominisat)
     *  * kissat:
     *    [Kissat](https://github.com/arminbiere/kissat)
     *  * lingeling:
     *    [Lingeling](https://github.com/arminbiere/lingeling)
     */
    BITWUZLA_OPT_SAT_SOLVER,

    /* ---------------- BV: Prop Engine Options (Expert) ---------------------- */

    /**  Propagation-based local search solver engine:
     *    Constant bits.
     *
     * Configure constant bit propagation (requires bit-blasting to AIG).
     *
     * Values:
     *  * 1: enable [default]
     *  * 0: disable
     *
     *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_CONST_BITS,

    /**  Propagation-based local search solver engine:
     *    Infer bounds for inequalities for value computation.
     *
     * When enabled, infer bounds for value computation for inequalities based on
     * satisfied top level inequalities.
     *
     * Values:
     *  * 1: enable
     *  * 0: disable [default]
     *
     *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_INEQ_BOUNDS,

    /**  Propagation-based local search solver engine:
     *    Number of propagations.
     *
     * Configure the number of propagations used as a limit for the
     * propagation-based local search solver engine. No limit if 0.
     *
     * Values:
     *  * An unsigned integer value. [default: 0]
     *
     *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_NPROPS,

    /**  Propagation-based local search solver engine:
     *    Number of updates.
     *
     * Configure the number of model value updates used as a limit for the
     * propagation-based local search solver engine. No limit if 0.
     *
     * Values:
     *  * An unsigned integer value. [default: 0]
     *
     *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_NUPDATES,

    /**  Propagation-based local search solver engine:
     *    Optimization for inverse value computation of inequalities over
     *    concat and sign extension operands.
     *
     * When enabled, use optimized inverse value value computation for
     * inequalities over concats.
     *
     * Values:
     *  * 1: enable
     *  * 0: disable [default]
     *
     *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_OPT_LT_CONCAT_SEXT,

    /**  Propagation-based local search solver engine:
     *    Path selection.
     *
     * Configure mode for path selection.
     *
     * Values:
     *  * essential:
     *    Select path based on essential inputs. [default]
     *  * random:
     *    Select path randomly.
     *
     *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PATH_SEL,

    /**  Propagation-based local search solver engine:
     *    Probability for selecting random input.
     *
     * Configure the probability with which to select a random input instead of
     * an essential input when selecting the path.
     *
     * Values:
     *  * An unsigned integer value <= 1000 (= 100%). [default: 0]
     *
     *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_PICK_RAND_INPUT,

    /**  Propagation-based local search solver engine:
     *    Probability for inverse values.
     *
     * Configure the probability with which to choose an inverse value over a
     * consistent value when aninverse value exists.
     *
     * Values:
     *  * An unsigned integer value <= 1000 (= 100%). [default: 990]
     *
     *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_PICK_INV_VALUE,

    /**  Propagation-based local search solver engine:
     *    Value computation for sign extension.
     *
     * When enabled, detect sign extension operations (are rewritten on
     * construction) and use value computation for sign extension.
     *
     * Values:
     *  * 1: enable
     *  * 0: disable [default]
     *
     *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_SEXT,

    /**  Propagation-based local search solver engine:
     *    Local search specific normalization.
     *
     * When enabled, perform normalizations for local search, on the local search
     * layer (does not affect node layer).
     *
     * Values:
     *  * 1: enable
     *  * 0: disable [default]
     *
     *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_NORMALIZE,

    /**  Preprocessing
     *
     * When enabled, applies all enabled preprocessing passes.
     *
     * Values:
     *  * 1: enable [default]
     *  * 0: disable
     */
    BITWUZLA_OPT_PREPROCESS,

    /**  Preprocessing: Find contradicting bit-vector ands
     *
     * When enabled, substitutes contradicting nodes of kind #BV_AND with zero.
     *
     * Values:
     *  * 1: enable [default]
     *  * 0: disable
     */
    BITWUZLA_OPT_PP_CONTR_ANDS,

    /**  Preprocessing: Eliminate bit-vector extracts on bit-vector constants
     *
     * When enabled, eliminates bit-vector extracts on constants.
     *
     * Values:
     *  * 1: enable
     *  * 0: disable [default]
     */
    BITWUZLA_OPT_PP_ELIM_EXTRACTS,

    /**  Preprocessing: Embedded constraint substitution
     *
     * When enabled, substitutes assertions that occur as sub-expression in the
     * formula with their respective Boolean value.
     *
     * Values:
     *  * 1: enable [default]
     *  * 0: disable
     */
    BITWUZLA_OPT_PP_EMBEDDED,

    /**  Preprocessing: AND flattening
     *
     * Values:
     *  * 1: enable [default]
     *  * 0: disable
     */
    BITWUZLA_OPT_PP_FLATTEN_AND,

    /**  Preprocessing: Normalization
     *
     * Values:
     *  * 1: enable [default]
     *  * 0: disable
     */
    BITWUZLA_OPT_PP_NORMALIZE,

    /**  Preprocessing: Normalization: Enable share awareness normlization
     *
     * When enabled, this disables normalizations that may yield blow-up on the
     * bit-level.
     *
     * Values:
     *  * 1: enable [default]
     *  * 0: disable
     */
    BITWUZLA_OPT_PP_NORMALIZE_SHARE_AWARE,

    /**  Preprocessing: Boolean skeleton preprocessing
     *
     * Values:
     *  * 1: enable [default]
     *  * 0: disable
     */
    BITWUZLA_OPT_PP_SKELETON_PREPROC,

    /**  Preprocessing: Variable substitution
     *
     * Values:
     *  * 1: enable [default]
     *  * 0: disable
     */
    BITWUZLA_OPT_PP_VARIABLE_SUBST,

    /**  Preprocessing: Variable substitution: Equality Normalization
     *
     * Values:
     *  * 1: enable [default]
     *  * 0: disable
     */
    BITWUZLA_OPT_PP_VARIABLE_SUBST_NORM_EQ,

    /**  Preprocessing: Variable substitution: Disequality Normalization
     *
     * Values:
     *  * 1: enable [default]
     *  * 0: disable
     */
    BITWUZLA_OPT_PP_VARIABLE_SUBST_NORM_DISEQ,

    /**  Preprocessing: Variable substitution: Bit-Vector Inequality
     * Normalization
     *
     * Values:
     *  * 1: enable [default]
     *  * 0: disable
     */
    BITWUZLA_OPT_PP_VARIABLE_SUBST_NORM_BV_INEQ,

    /**  Debug:
     *    Threshold for number of new nodes introduced for recursive call of
     *    rewrite().
     *
     *  Prints a warning number of newly introduced nodes is above threshold.
     *
     *  @warning This is an expert debug option.
     */
    BITWUZLA_OPT_DBG_RW_NODE_THRESH,

    /**  Debug:
     *    Threshold for formula size increase after preprocessing in percent.
     *
     *  Prints a warning if formula size increase is above threshold.
     *
     *  @warning This is an expert debug option.
     */
    BITWUZLA_OPT_DBG_PP_NODE_THRESH,

    /**  Debug: Check models for each satisfiable query.
     *
     *  @warning This is an expert debug option.
     */
    BITWUZLA_OPT_CHECK_MODEL,

    /**  Debug: Check unsat core for each unsatisfiable query.
     *
     *  @warning This is an expert debug option.
     */
    BITWUZLA_OPT_CHECK_UNSAT_CORE;

    companion object {
        private val valueMapping = BitwuzlaOption.values().associateBy { it.ordinal }
        private val nameMapping = BitwuzlaOption.values().associateBy { it.name }
        fun fromValue(value: Int): BitwuzlaOption = valueMapping.getValue(value)
        fun forName(name: String): BitwuzlaOption? = nameMapping[name]
    }
}
