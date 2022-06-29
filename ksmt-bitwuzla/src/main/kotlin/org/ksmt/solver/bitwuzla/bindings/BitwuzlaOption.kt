package org.ksmt.solver.bitwuzla.bindings

/**
 * The configuration options supported by Bitwuzla.<br>
 * * Options that list string values can be configured via<br>
 * `bitwuzla_set_option_str`. Options with integer configuration values are<br>
 * configured via `bitwuzla_set_option`.<br>
 * * For all options, the current configuration value can be queried via<br>
 * `bitwuzla_get_option`.<br>
 * Options with string configuration values internally represent these<br>
 * values as enum values.<br>
 * For these options, `bitwuzla_get_opiton` will return such an enum value.<br>
 * Use `bitwuzla_get_option_str` to query enum options for the corresponding<br>
 * string representation.<br>
 * enum values<br>
 * <i>native declaration : bitwuzla.h:51</i>
 */
enum class BitwuzlaOption(val value: Int) {
    /**
     * **Configure the solver engine.**<br></br>
     * * Values:<br></br>
     * * **aigprop**:<br></br>
     * The propagation-based local search QF_BV engine that operates on the<br></br>
     * bit-blasted formula (the AIG circuit layer).<br></br>
     * * **fun** [**default**]:<br></br>
     * The default engine for all combinations of QF_AUFBVFP, uses lemmas on<br></br>
     * demand for QF_AUFBVFP, and eager bit-blasting (optionally with local<br></br>
     * searchin a sequential portfolio) for QF_BV.<br></br>
     * * **prop**:<br></br>
     * The propagation-based local search QF_BV engine.<br></br>
     * * **sls**:<br></br>
     * The stochastic local search QF_BV engine.<br></br>
     * * **quant**:<br></br>
     * The quantifier engine.
     */
    BITWUZLA_OPT_ENGINE(0),

    /**
     * **Use non-zero exit codes for sat and unsat results.**<br></br>
     * * When enabled, use Bitwuzla exit codes:<br></br>
     * * `::BITWUZLA_SAT`<br></br>
     * * `::BITWUZLA_UNSAT`<br></br>
     * * `::BITWUZLA_UNKNOWN`<br></br>
     * * When disabled, return 0 on success (sat, unsat, unknown), and a non-zero<br></br>
     * exit code otherwise.<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable
     */
    BITWUZLA_OPT_EXIT_CODES(1),

    /**
     * **Configure input file format.**<br></br>
     * * If unspecified, Bitwuzla will autodetect the input file format.<br></br>
     * * Values:<br></br>
     * * **none** [**default**]:<br></br>
     * Auto-detect input file format.<br></br>
     * * **btor**:<br></br>
     * \verbatim embed:rst:leading-asterisk<br></br>
     * BTOR format :cite:`btor`<br></br>
     * \endverbatim<br></br>
     * * **btor2**:<br></br>
     * \verbatim embed:rst:leading-asterisk<br></br>
     * BTOR2 format :cite:`btor2`<br></br>
     * \endverbatim<br></br>
     * * **smt2**:<br></br>
     * \verbatim embed:rst:leading-asterisk<br></br>
     * SMT-LIB v2 format :cite:`smtlib2`<br></br>
     * \endverbatim
     */
    BITWUZLA_OPT_INPUT_FORMAT(2),

    /**
     * **Incremental solving.**<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * * @note<br></br>
     * * Enabling this option turns off some optimization techniques.<br></br>
     * * Enabling/disabling incremental solving after bitwuzla_check_sat()<br></br>
     * has been called is not supported.<br></br>
     * * This option cannot be enabled in combination with option<br></br>
     * `::BITWUZLA_OPT_PP_UNCONSTRAINED_OPTIMIZATION`.
     */
    BITWUZLA_OPT_INCREMENTAL(3),

    /**
     * **Log level.**<br></br>
     * * Values:<br></br>
     * * An unsigned integer value (**default**: 0).
     */
    BITWUZLA_OPT_LOGLEVEL(4),

    /**
     * **Configure output number format for bit-vector values.**<br></br>
     * * If unspecified, Bitwuzla will use BTOR format.<br></br>
     * * Values:<br></br>
     * * **aiger**:<br></br>
     * \verbatim embed:rst:leading-asterisk<br></br>
     * AIGER ascii format :cite:`aiger`<br></br>
     * \endverbatim<br></br>
     * * **aigerbin**:<br></br>
     * \verbatim embed:rst:leading-asterisk<br></br>
     * AIGER binary format :cite:`aiger`<br></br>
     * \endverbatim<br></br>
     * * **btor** [**default**]:<br></br>
     * \verbatim embed:rst:leading-asterisk<br></br>
     * BTOR format :cite:`btor`<br></br>
     * \endverbatim<br></br>
     * * **smt2**:<br></br>
     * \verbatim embed:rst:leading-asterisk<br></br>
     * SMT-LIB v2 format :cite:`smtlib2`<br></br>
     * \endverbatim
     */
    BITWUZLA_OPT_OUTPUT_FORMAT(5),

    /**
     * **Configure output number format for bit-vector values.**<br></br>
     * * If unspecified, Bitwuzla will use binary representation.<br></br>
     * * Values:<br></br>
     * * **bin** [**default**]:<br></br>
     * Binary number format.<br></br>
     * * **hex**:<br></br>
     * Hexadecimal number format.<br></br>
     * * **dec**:<br></br>
     * Decimal number format.
     */
    BITWUZLA_OPT_OUTPUT_NUMBER_FORMAT(6),

    /**
     * **Pretty printing.**<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable
     */
    BITWUZLA_OPT_PRETTY_PRINT(7),

    /**
     * **Print DIMACS.**<br></br>
     * * Print the CNF sent to the SAT solver in DIMACS format to stdout.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]
     */
    BITWUZLA_OPT_PRINT_DIMACS(8),

    /**
     * **Model generation.**<br></br>
     * * Values:<br></br>
     * * **1**: enable, generate model for assertions only<br></br>
     * * **2**: enable, generate model for all created terms<br></br>
     * * **0**: disable [**default**]<br></br>
     * * @note This option cannot be enabled in combination with option<br></br>
     * `::BITWUZLA_OPT_PP_UNCONSTRAINED_OPTIMIZATION`.
     */
    BITWUZLA_OPT_PRODUCE_MODELS(9),

    /**
     * **Unsat core generation.**<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]
     */
    BITWUZLA_OPT_PRODUCE_UNSAT_CORES(10),

    /**
     * **Configure the SAT solver engine.**<br></br>
     * * Values:<br></br>
     * * **cadical** [**default**]:<br></br>
     * [CaDiCaL](https://github.com/arminbiere/cadical)<br></br>
     * * **cms**:<br></br>
     * [CryptoMiniSat](https://github.com/msoos/cryptominisat)<br></br>
     * * **gimsatul**:<br></br>
     * [Gimsatul](https://github.com/arminbiere/gimsatul)<br></br>
     * * **kissat**:<br></br>
     * [Kissat](https://github.com/arminbiere/kissat)<br></br>
     * * **lingeling**:<br></br>
     * [Lingeling](https://github.com/arminbiere/lingeling)<br></br>
     * * **minisat**:<br></br>
     * [MiniSat](https://github.com/niklasso/minisat)<br></br>
     * * **picosat**:<br></br>
     * [PicoSAT](http://fmv.jku.at/picosat/)
     */
    BITWUZLA_OPT_SAT_ENGINE(11),

    /**
     * **Seed for random number generator.**<br></br>
     * * Values:<br></br>
     * * An unsigned integer value (**default**: 0).
     */
    BITWUZLA_OPT_SEED(12),

    /**
     * **Verbosity level.**<br></br>
     * * Values:<br></br>
     * * An unsigned integer value <= 4 (**default**: 0).
     */
    BITWUZLA_OPT_VERBOSITY(13),

    /**
     * **Ackermannization preprocessing.**<br></br>
     * * Eager addition of Ackermann constraints for function applications.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_ACKERMANN(14),

    /**
     * **Beta reduction preprocessing.**<br></br>
     * * Eager elimination of lambda terms via beta reduction.<br></br>
     * * Values:<br></br>
     * * **none** [**default**]:<br></br>
     * Disable beta reduction preprocessing.<br></br>
     * * **fun**:<br></br>
     * Only beta reduce functions that do not represent array stores.<br></br>
     * * **all**:<br></br>
     * Only beta reduce all functions, including array stores.<br></br>
     * *  @warning This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_BETA_REDUCE(15),

    /**
     * **Eliminate bit-vector extracts (preprocessing).**<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_ELIMINATE_EXTRACTS(16),

    /**
     * **Eliminate ITEs (preprocessing).**<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_ELIMINATE_ITES(17),

    /**
     * **Extract lambdas (preprocessing).**<br></br>
     * * Extraction of common array patterns as lambda terms.<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_EXTRACT_LAMBDAS(18),

    /**
     * **Merge lambda terms (preprocessing).**<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_MERGE_LAMBDAS(19),

    /**
     * **Non-destructive term substitutions.**<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_NONDESTR_SUBST(20),

    /**
     * **Normalize bit-vector addition (global).**<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_NORMALIZE_ADD(21),

    /**
     * **Boolean skeleton preprocessing.**<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_SKELETON_PREPROC(22),

    /**
     * **Unconstrained optimization (preprocessing).**<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_UNCONSTRAINED_OPTIMIZATION(23),

    /**
     * **Variable substitution preprocessing.**<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_VAR_SUBST(24),

    /**
     * **Propagate bit-vector extracts over arithmetic bit-vector operators.**<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_RW_EXTRACT_ARITH(25),

    /**
     * **Rewrite level.**<br></br>
     * * Values:<br></br>
     * * **0**: no rewriting<br></br>
     * * **1**: term level rewriting<br></br>
     * * **2**: term level rewriting and basic preprocessing<br></br>
     * * **3**: term level rewriting and full preprocessing [**default**]<br></br>
     * * @note Configuring the rewrite level after terms have been created<br></br>
     * is not allowed.<br></br>
     * *  @warning This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_RW_LEVEL(26),

    /**
     * **Normalize bit-vector operations.**<br></br>
     * * Normalize bit-vector addition, multiplication and bit-wise and.<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_RW_NORMALIZE(27),

    /**
     * **Normalize bit-vector addition (local).**<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_RW_NORMALIZE_ADD(28),

    /**
     * **Simplify constraints on construction.**<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_RW_SIMPLIFY_CONSTRAINTS(29),

    /**
     * **Eliminate bit-vector SLT nodes.**<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_RW_SLT(30),

    /**
     * **Sort the children of AIG nodes by id.**<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_RW_SORT_AIG(31),

    /**
     * **Sort the children of adder and multiplier circuits by id.**<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_RW_SORT_AIGVEC(32),

    /**
     * **Sort the children of commutative operations by id.**<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_RW_SORT_EXP(33),

    /**
     * **Function solver engine:<br></br>
     * Dual propagation optimization.**<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the fun solver engine.
     */
    BITWUZLA_OPT_FUN_DUAL_PROP(34),

    /**
     * **Function solver engine:<br></br>
     * Assumption order for dual propagation optimization.**<br></br>
     * * Set order in which inputs are assumed in the dual propagation clone.<br></br>
     * * Values:<br></br>
     * * **just** [**default**]:<br></br>
     * Order by score, highest score first.<br></br>
     * * **asc**:<br></br>
     * Order by input id, ascending.<br></br>
     * * **desc**:<br></br>
     * Order by input id, descending.<br></br>
     * *  @warning This is an expert option to configure the func solver engine.
     */
    BITWUZLA_OPT_FUN_DUAL_PROP_QSORT(35),

    /**
     * **Function solver engine:<br></br>
     * Eager lemmas.**<br></br>
     * * Configure mode for eager lemma generation.<br></br>
     * * Values:<br></br>
     * * **none**:<br></br>
     * Do not generate lemmas eagerly (generate one single lemma per<br></br>
     * refinement iteration).<br></br>
     * * **conf** [**default**]:<br></br>
     * Only generate lemmas eagerly until the first conflict dependent on<br></br>
     * another conflict is found.<br></br>
     * * **all**:<br></br>
     * In each refinement iteration, generate lemmas for all conflicts.<br></br>
     * *  @warning This is an expert option to configure the func solver engine.
     */
    BITWUZLA_OPT_FUN_EAGER_LEMMAS(36),

    /**
     * **Function solver engine:<br></br>
     * Lazy synthesis.**<br></br>
     * * Configure lazy synthesis (to bit-level) of bit-vector expressions.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the func solver engine.
     */
    BITWUZLA_OPT_FUN_LAZY_SYNTHESIZE(37),

    /**
     * **Function solver engine:<br></br>
     * Justification optimization.**<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the func solver engine.
     */
    BITWUZLA_OPT_FUN_JUST(38),

    /**
     * **Function solver engine:<br></br>
     * Justification optimization heuristic.**<br></br>
     * * Configure heuristic to determine path selection for justification<br></br>
     * optimization.<br></br>
     * * Values:<br></br>
     * * **applies** [**default**]:<br></br>
     * Choose branch with minimum number of applies.<br></br>
     * * **depth**:<br></br>
     * Choose branch with minimum depth.<br></br>
     * * **left**:<br></br>
     * Always choose left branch.<br></br>
     * *  @warning This is an expert option to configure the func solver engine.
     */
    BITWUZLA_OPT_FUN_JUST_HEURISTIC(39),

    /**
     * **Function solver engine:<br></br>
     * Propagation-based local search sequential portfolio.**<br></br>
     * * When function solver engine is enabled, configure propagation-based local<br></br>
     * search solver engine as preprocessing step within sequential portfolio<br></br>
     * setting.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the func solver engine.
     */
    BITWUZLA_OPT_FUN_PREPROP(40),

    /**
     * **Function solver engine:<br></br>
     * Stochastic local search sequential portfolio.**<br></br>
     * * When function solver engine is enabled, configure stochastic local<br></br>
     * search solver engine as preprocessing step within sequential portfolio<br></br>
     * setting.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the func solver engine.
     */
    BITWUZLA_OPT_FUN_PRESLS(41),

    /**
     * **Function solver engine:<br></br>
     * Represent store as lambda.**<br></br>
     * * Represent array stores as lambdas.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the func solver engine.
     */
    BITWUZLA_OPT_FUN_STORE_LAMBDAS(42),

    /**
     * **Stochastic local search solver engine:<br></br>
     * Justification-based path selection.**<br></br>
     * * Configure justification-based path selection for SLS engine.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_JUST(43),

    /**
     * **Stochastic local search solver engine:<br></br>
     * Group-wise moves.**<br></br>
     * * Configure group-wise moves for SLS engine. When enabled, rather than<br></br>
     * changing the assignment of one single candidate variable, all candidates<br></br>
     * are set at the same time (using the same strategy).<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_GW(44),

    /**
     * **Stochastic local search solver engine:<br></br>
     * Incremental move test.**<br></br>
     * * Configure that during best move selection, the previous best neighbor<br></br>
     * for the current candidate is used for neighborhood exploration rather<br></br>
     * than its current assignment.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_INC_MOVE_TEST(45),

    /**
     * **Stochastic local search solver engine:<br></br>
     * Propagation moves.**<br></br>
     * * Configure propagation moves, chosen with a ratio of number of propagation<br></br>
     * moves `::BITWUZLA_OPT_SLS_MOVE_PROP_NPROPS` to regular SLS moves<br></br>
     * `::BITWUZLA_OPT_SLS_MOVE_PROP_NSLSS`.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_PROP(46),

    /**
     * **Stochastic local search solver engine:<br></br>
     * Force random walks.**<br></br>
     * * Configure that random walks are forcibly chosen as recovery moves in case<br></br>
     * of conflicts when a propagation move is performed (rather than performing<br></br>
     * a regular SLS move).<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_PROP_FORCE_RW(47),

    /**
     * **Stochastic local search solver engine:<br></br>
     * Number of propagation moves.**<br></br>
     * * Configure the number of propagation moves to be performed when propagation<br></br>
     * moves are enabled. Propagation moves are chosen with a ratio of<br></br>
     * `::BITWUZLA_OPT_SLS_MOVE_PROP_NPROPS` to<br></br>
     * `::BITWUZLA_OPT_SLS_MOVE_PROP_NSLSS`.<br></br>
     * * Values:<br></br>
     * * An unsigned integer value (**default**: 1)<br></br>
     * * @see<br></br>
     * * BITWUZLA_OPT_SLS_MOVE_PROP<br></br>
     * * BITWUZLA_OPT_SLS_MOVE_PROP_NSLSS<br></br>
     * *  @warning This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_PROP_NPROPS(48),

    /**
     * **Stochastic local search solver engine:<br></br>
     * Number of regular SLS moves.**<br></br>
     * * Configure the number of regular SLS moves to be performed when propagation<br></br>
     * moves are enabled. Propagation moves are chosen with a ratio of<br></br>
     * `::BITWUZLA_OPT_SLS_MOVE_PROP_NPROPS` to<br></br>
     * `::BITWUZLA_OPT_SLS_MOVE_PROP_NSLSS`.<br></br>
     * * Values:<br></br>
     * * An unsigned integer value (**default**: 1)<br></br>
     * * @see<br></br>
     * * BITWUZLA_OPT_SLS_MOVE_PROP<br></br>
     * * BITWUZLA_OPT_SLS_MOVE_PROP_NPROPS<br></br>
     * *  @warning This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_PROP_NSLSS(49),

    /**
     * **Stochastic local search solver engine:<br></br>
     * Randomize all candidates.**<br></br>
     * * Configure the randomization of all candidate variables (rather than just<br></br>
     * a single randomly selected one) in case no best move has been found.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_RAND_ALL(50),

    /**
     * **Stochastic local search solver engine:<br></br>
     * Randomize bit ranges.**<br></br>
     * * Configure the randomization of bit ranges (rather than all bits) of<br></br>
     * candidate variable(s) in case no best move has been found.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_RAND_RANGE(51),

    /**
     * **Stochastic local search solver engine:<br></br>
     * Random walk.**<br></br>
     * * Configure random walk moves, where one out of all possible neighbors is<br></br>
     * randomly selected (with given probability<br></br>
     * `::BITWUZLA_OPT_SLS_PROB_MOVE_RAND_WALK`) for a randomly selected<br></br>
     * candidate variable.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * * @see<br></br>
     * * BITWUZLA_OPT_SLS_MOVE_PROB_RAND_WALK<br></br>
     * *  @warning This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_RAND_WALK(52),

    /**
     * **Stochastic local search solver engine:<br></br>
     * Range-wise bit-flip moves.**<br></br>
     * * Configure range-wise bit-flip moves for SLS engine. When enabled, try<br></br>
     * range-wise bit-flips when selecting moves, where bits within all ranges<br></br>
     * from 2 to the bit-width (starting from the LSB) are flipped at once.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_RANGE(53),

    /**
     * **Stochastic local search solver engine:<br></br>
     * Segment-wise bit-flip moves.**<br></br>
     * * Configure range-wise bit-flip moves for SLS engine. When enabled, try<br></br>
     * segment-wise bit-flips when selecting moves, where bits within segments<br></br>
     * of multiples of 2 are flipped at once.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_SEGMENT(54),

    /**
     * **Stochastic local search solver engine:<br></br>
     * Probability for random walks.**<br></br>
     * * Configure the probability with which a random walk is chosen if random<br></br>
     * walks are enabled.<br></br>
     * * Values:<br></br>
     * * An unsigned integer value <= 1000 (= 100%) (**default**: 100)<br></br>
     * * @see<br></br>
     * * BITWUZLA_OPT_SLS_MOVE_RAND_WALK<br></br>
     * *  @warning This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_PROB_MOVE_RAND_WALK(55),

    /**
     * **Stochastic local search solver engine:<br></br>
     * Number of bit flips.**<br></br>
     * * Configure the number of bit flips used as a limit for the SLS engine.<br></br>
     * * Values:<br></br>
     * * An unsigned integer value, no limit if 0 (**default**: 0).<br></br>
     * *  @warning This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_NFLIPS(56),

    /**
     * **Stochastic local search solver engine:<br></br>
     * Move strategy.**<br></br>
     * * Configure the move selection strategy for the SLS engine.<br></br>
     * * Values:<br></br>
     * * **best** [**default**]:<br></br>
     * Choose best score improving move.<br></br>
     * * **walk**:<br></br>
     * Choose random walk weighted by score.<br></br>
     * * **first**:<br></br>
     * Choose first best move (no matter if any other move is better).<br></br>
     * * **same**:<br></br>
     * Determine move as best move even if its score is not better but the<br></br>
     * same as the score of the previous best move.<br></br>
     * * **prop**:<br></br>
     * Choose propagation move (and recover with SLS move in case of conflict).<br></br>
     * *  @warning This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_STRATEGY(57),

    /**
     * **Stochastic local search solver engine:<br></br>
     * Restarts.**<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_USE_RESTARTS(58),

    /**
     * **Stochastic local search solver engine:<br></br>
     * Bandit scheme.**<br></br>
     * * Configure bandit scheme heuristic for selecting root constraints.<br></br>
     * If disabled, root constraints are selected randomly.<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_USE_BANDIT(59),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Value computation for xor.**<br></br>
     * * When enabled, detect arithmetic right shift operations (are rewritten on<br></br>
     * construction) and use value computation for ashr.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_ASHR(60),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Constant bits.**<br></br>
     * * Configure constant bit propagation (requries bit-blasting to AIG).<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_CONST_BITS(61),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Domain propagators.**<br></br>
     * * Configure the use of domain propagators for determining constant bits<br></br>
     * (instead of bit-blastin to AIG).<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_CONST_DOMAINS(62),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Entailed propagations.**<br></br>
     * * Maintain a work queue with entailed propagations.<br></br>
     * If enabled, propagations from this queue are propagated before randomly<br></br>
     * choosing a yet unsatisfied path from the root.<br></br>
     * * Values:<br></br>
     * *  * **off** [**default**]:<br></br>
     * Disable strategy.<br></br>
     * * **all**:<br></br>
     * Propagate all entailed propagations.<br></br>
     * * **first**:<br></br>
     * Process only the first entailed propagation.<br></br>
     * * **last**:<br></br>
     * Process only the last entailed propagation.<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_ENTAILED(63),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Delta for flipping ite conditions with constant branches.**<br></br>
     * * Configure the delta by which `::BITWUZLA_OPT_PROP_PROB_FLIP_COND_CONST` is<br></br>
     * decreased or increased after a limit<br></br>
     * `::BITWUZLA_OPT_PROP_FLIP_COND_CONST_NPATHSEL` is reached.<br></br>
     * * Values:<br></br>
     * * A signed integer value (**default**: 100).<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_FLIP_COND_CONST_DELTA(64),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Limit for flipping ite conditions with constant branches.**<br></br>
     * * Configure the limit for how often the path to the condition for ite<br></br>
     * operations with constant branches may be selected before<br></br>
     * `::BITWUZLA_OPT_PROP_PROB_FLIP_COND_CONST` is decreased or increased by<br></br>
     * `::BITWUZLA_OPT_PROP_FLIP_COND_CONST_DELTA`.<br></br>
     * * Values:<br></br>
     * * A signed integer value (**default**: 500).<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_FLIP_COND_CONST_NPATHSEL(65),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Infer bounds for inequalities for value computation.**<br></br>
     * * When enabled, infer bounds for value computation for inequalities based on<br></br>
     * satisfied top level inequalities.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_INFER_INEQ_BOUNDS(66),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * No move on conflict.**<br></br>
     * * When enabled, no move is performed when running into a conflict during<br></br>
     * value computation.<br></br>
     * * @note This is the default behavior for the SLS engine when propagation<br></br>
     * moves are enabled, where a conflict triggers a recovery by means<br></br>
     * of a regular SLS move.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_NO_MOVE_ON_CONFLICT(67),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Number of propagations.**<br></br>
     * * Configure the number of propagations used as a limit for the<br></br>
     * propagation-based local search solver engine. No limit if 0.<br></br>
     * * Values:<br></br>
     * * An unsigned integer value (**default**: 0).<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_NPROPS(68),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Number of updates.**<br></br>
     * * Configure the number of model value updates used as a limit for the<br></br>
     * propagation-based local search solver engine. No limit if 0.<br></br>
     * * Values:<br></br>
     * * An unsigned integer value (**default**: 0).<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_NUPDATES(69),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Path selection.**<br></br>
     * * Configure mode for path selection.<br></br>
     * * Values:<br></br>
     * * **essential** [default]:<br></br>
     * Select path based on essential inputs.<br></br>
     * * **random**:<br></br>
     * Select path randomly.<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PATH_SEL(70),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Probability for producing inverse rather than consistent values.**<br></br>
     * * Values:<br></br>
     * * An unsigned integer value <= 1000 (= 100%) (**default**: 0).<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_FALLBACK_RANDOM_VALUE(71),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Probability for flipping one of the don't care bits for ands.**<br></br>
     * * Configure the probability with which to keep the current assignement of<br></br>
     * the operand to a bit-vector and with max one bit flipped (rather than<br></br>
     * fully randomizing the assignment) when selecting an inverse or consistent<br></br>
     * value.<br></br>
     * * Values:<br></br>
     * * An unsigned integer value <= 1000 (= 100%) (**default**: 0).<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_AND_FLIP(72),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Probability for using the current assignment with one bit flipped for<br></br>
     * equalities.**<br></br>
     * * Configure the probability with which the current assignment of an operand<br></br>
     * to a disequality is kept with just a single bit flipped (rather than fully<br></br>
     * randomizing the assignment) when selecting an inverse or consistent value.<br></br>
     * * Values:<br></br>
     * * An unsigned integer value <= 1000 (= 100%) (**default**: 0).<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_EQ_FLIP(73),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Probability for flipping ite condition.**<br></br>
     * * Configure the probability with which to select the path to the condition<br></br>
     * (in case of an ite operation) rather than the enabled branch during down<br></br>
     * propagation).<br></br>
     * * Values:<br></br>
     * * An unsigned integer value <= 1000 (= 100%) (**default**: 100).<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_FLIP_COND(74),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Probability for flipping ite condition with constant branches.**<br></br>
     * * Configure the probability with which to select the path to the condition<br></br>
     * (in case of an ite operation) rather than the enabled branch during down<br></br>
     * propagation) if either the 'then' or 'else' branch is constant.<br></br>
     * * Values:<br></br>
     * * An unsigned integer value <= 1000 (= 100%) (**default**: 100).<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_FLIP_COND_CONST(75),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Probability for selecting random input.**<br></br>
     * * Configure the probability with which to select a random input instead of<br></br>
     * an essential input when selecting the path.<br></br>
     * * Values:<br></br>
     * * An unsigned integer value <= 1000 (= 100%) (**default**: 0).<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_RANDOM_INPUT(76),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Probability for flipping one of the don't care bits for extracts.**<br></br>
     * * Configure the probability with which to flip one of the don't care bits of<br></br>
     * the current assignment of the operand to a bit-vector extract (when the<br></br>
     * asignment is kept) when selecting an inverse or consistent value.<br></br>
     * * Values:<br></br>
     * * An unsigned integer value <= 1000 (= 100%) (**default**: 0).<br></br>
     * * @see<br></br>
     * * BITWUZLA_OPT_PROP_PROB_SLICE_KEEP_DC<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_SLICE_FLIP(77),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Probability for keeping the value of don't care bits for extracts.**<br></br>
     * * Configure the probability with which to keep the current value of don't<br></br>
     * care bits of an extract operation (rather than fully randomizing them)<br></br>
     * when selecting an inverse or consistent value.<br></br>
     * * Values:<br></br>
     * * An unsigned integer value <= 1000 (= 100%) (**default**: 500).<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_SLICE_KEEP_DC(78),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Probability for inverse values.**<br></br>
     * * Configure the probability with which to choose an inverse value over a<br></br>
     * consistent value when aninverse value exists.<br></br>
     * * Values:<br></br>
     * * An unsigned integer value <= 1000 (= 100%) (**default**: 990).<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_USE_INV_VALUE(79),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Bandit scheme.**<br></br>
     * * Configure bandit scheme heuristic for selecting root constraints.<br></br>
     * If enabled, root constraint selection via bandit scheme is based on a<br></br>
     * scoring scheme similar to the one used in the SLS engine.<br></br>
     * If disabled, root constraints are selected randomly.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_USE_BANDIT(80),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Inverse value computation for inequalities over concats.**<br></br>
     * * When enabled, use special inverse value computation for inequality over<br></br>
     * concats.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_USE_INV_LT_CONCAT(81),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Restarts.**<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_USE_RESTARTS(82),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Value computation for sign extension.**<br></br>
     * * When enabled, detect sign extension operations (are rewritten on<br></br>
     * construction) and use value computation for sign extension.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_SEXT(83),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Skip if no progress.**<br></br>
     * * When enabled, moves that make no progress, that is, that produce a target<br></br>
     * value that is the seame as the current assignment of a variable, are<br></br>
     * skipped.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_SKIP_NO_PROGRESS(84),

    /**
     * **Propagation-based local search solver engine:<br></br>
     * Value computation for xor.**<br></br>
     * * When enabled, detect xor operations (are rewritten on construction) and<br></br>
     * use value computation for xor.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_XOR(85),

    /**
     * **AIG-level propagation-based local search solver engine:<br></br>
     * Number of propagations.**<br></br>
     * * Configure the number of propagations used as a limit for the<br></br>
     * propagation-based local search solver engine. No limit if 0.<br></br>
     * * Values:<br></br>
     * * An unsigned integer value (**default**: 0).<br></br>
     * *  @warning This is an expert option to configure the aigprop solver engine.
     */
    BITWUZLA_OPT_AIGPROP_NPROPS(86),

    /**
     * **AIG-level propagation-based local search solver engine:<br></br>
     * Bandit scheme.**<br></br>
     * * Configure bandit scheme heuristic for selecting root constraints.<br></br>
     * If enabled, root constraint selection via bandit scheme is based on a<br></br>
     * scoring scheme similar to the one used in the SLS engine.<br></br>
     * If disabled, root constraints are selected randomly.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the aigprop solver engine.
     */
    BITWUZLA_OPT_AIGPROP_USE_BANDIT(87),

    /**
     * **AIG-level propagation-based local search solver engine:<br></br>
     * Restarts.**<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option to configure the aigprop solver engine.
     */
    BITWUZLA_OPT_AIGPROP_USE_RESTARTS(88),

    /**
     * **Quantifier solver engine:<br></br>
     * Constructive Equality Resolution.**<br></br>
     * * Configure the use of Constructive Equality Resolution simplification in<br></br>
     * the quantifier solver engine.<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure the quantifier solver<br></br>
     * engine.
     */
    BITWUZLA_OPT_QUANT_CER(89),

    /**
     * **Quantifier solver engine:<br></br>
     * Destructive Equality Resolution.**<br></br>
     * * Configure the use of Destructive Equality Resolution simplification in<br></br>
     * the quantifier solver engine.<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure the quantifier solver<br></br>
     * engine.
     */
    BITWUZLA_OPT_QUANT_DER(90),

    /**
     * **Quantifier solver engine:<br></br>
     * Dual solver.**<br></br>
     * * Configure the use of the dual (negated) version of the quantified formula.<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure the quantifier solver<br></br>
     * engine.
     */
    BITWUZLA_OPT_QUANT_DUAL_SOLVER(91),

    /**
     * **Quantifier solver engine:<br></br>
     * Miniscoping.**<br></br>
     * * Configure the use of miniscoping in the quantifier solver engine.<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure the quantifier solver<br></br>
     * engine.
     */
    BITWUZLA_OPT_QUANT_MINISCOPE(92),

    /**
     * **Quantifier solver engine:<br></br>
     * Synthesis mode.**<br></br>
     * * Configure mode for synthesizing Skolem functions.<br></br>
     * * Values:<br></br>
     * * **none**:<br></br>
     * Do not synthesize skolem functions (use model values for instantiation).<br></br>
     * * **el**:<br></br>
     * Use enumerative learning to synthesize skolem functions.<br></br>
     * * **elmc**:<br></br>
     * Use enumerative learning modulo the predicates in the cone of influence<br></br>
     * of the existential variables to synthesize skolem functions.<br></br>
     * * **elelmc**:<br></br>
     * Chain `el` and `elmc` approaches to synthesize skolem functions.<br></br>
     * * **elmr** [**default**]:<br></br>
     * Use enumerative learning modulo the given root constraints to synthesize<br></br>
     * skolem functions.<br></br>
     * *  @warning This is an expert option to configure the quantifier solver<br></br>
     * engine.
     */
    BITWUZLA_OPT_QUANT_SYNTH(93),

    /**
     * **Quantifier solver engine:<br></br>
     * Update model with respect to synthesized skolem.**<br></br>
     * * Configure to update the current model with respect to the synthesized<br></br>
     * skolem function if enabled.<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure the quantifier solver<br></br>
     * engine.
     */
    BITWUZLA_OPT_QUANT_FIXSYNTH(94),

    /**
     * **Quantifier solver engine:<br></br>
     * Base case for ITE model.**<br></br>
     * * Configure the base case of a concrete model for ITEs. Constant if enabled,<br></br>
     * else undefined.<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure the quantifier solver<br></br>
     * engine.
     */
    BITWUZLA_OPT_QUANT_SYNTH_ITE_COMPLETE(95),

    /**
     * **Quantifier solver engine:<br></br>
     * Limit for synthesis.**<br></br>
     * * Configure the limit of enumerated expressions for the enumerative learning<br></br>
     * synthesis algorithm implemented in the quantified solver engine.<br></br>
     * * Values:<br></br>
     * * An unsigned integer value (**default**: 10000).<br></br>
     * *  @warning This is an expert option to configure the quantifier solver<br></br>
     * engine.
     */
    BITWUZLA_OPT_QUANT_SYNTH_LIMIT(96),

    /**
     * **Quantifier solver engine:<br></br>
     * Quantifier instantiation.**<br></br>
     * * Configure the generalization of quantifier instantiations via enumerative<br></br>
     * learning.<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option to configure the quantifier solver<br></br>
     * engine.
     */
    BITWUZLA_OPT_QUANT_SYNTH_QI(97),

    /**
     * **Check model (debug only).**<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option.
     */
    BITWUZLA_OPT_CHECK_MODEL(98),

    /**
     * **Check result when unconstrained optimization is enabled (debug only).**<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option.
     */
    BITWUZLA_OPT_CHECK_UNCONSTRAINED(99),

    /**
     * **Check unsat assumptions (debug only).**<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option.
     */
    BITWUZLA_OPT_CHECK_UNSAT_ASSUMPTIONS(100),

    /**
     * **Interpret sorts introduced with declare-sort as bit-vectors of given<br></br>
     * width.**<br></br>
     * * Disabled if zero.<br></br>
     * * Values:<br></br>
     * * An unsigned integer value (**default**: 0).<br></br>
     * *  @warning This is an expert option.
     */
    BITWUZLA_OPT_DECLSORT_BV_WIDTH(101),

    /**
     * **Share partial models determined via local search with bit-blasting<br></br>
     * engine.**<br></br>
     * * This option is only effective when local search engines are combined with<br></br>
     * the bit-blasting engine in a sequential portfolio.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option.
     */
    BITWUZLA_OPT_LS_SHARE_SAT(102),

    /**
     * **Interactive parsing mode.**<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option.
     */
    BITWUZLA_OPT_PARSE_INTERACTIVE(103),

    /**
     * **Use CaDiCaL's freeze/melt.**<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option.
     */
    BITWUZLA_OPT_SAT_ENGINE_CADICAL_FREEZE(104),

    /**
     * **Lingeling fork mode.**<br></br>
     * * Values:<br></br>
     * * **1**: enable [**default**]<br></br>
     * * **0**: disable<br></br>
     * *  @warning This is an expert option.
     */
    BITWUZLA_OPT_SAT_ENGINE_LGL_FORK(105),

    /**
     * **Number of threads to use in the SAT solver.**<br></br>
     * * This option is only effective for SAT solvers with support for<br></br>
     * multi-threading.<br></br>
     * * Values:<br></br>
     * * An unsigned integer value > 0 (**default**: 1).<br></br>
     * *  @warning This is an expert option.
     */
    BITWUZLA_OPT_SAT_ENGINE_N_THREADS(106),

    /**
     * **Enable SMT-COMP mode.**<br></br>
     * * Parser only option. Only effective when an SMT2 input file is parsed.<br></br>
     * * Values:<br></br>
     * * **1**: enable<br></br>
     * * **0**: disable [**default**]<br></br>
     * *  @warning This is an expert option.
     */
    BITWUZLA_OPT_SMT_COMP_MODE(107),

    /** this MUST be the last entry!  */
    BITWUZLA_OPT_NUM_OPTS(108);

    companion object {
        private val valueMapping = BitwuzlaOption.values().associateBy { it.value }
        fun fromValue(value: Int): BitwuzlaOption = valueMapping.getValue(value)
    }
}
