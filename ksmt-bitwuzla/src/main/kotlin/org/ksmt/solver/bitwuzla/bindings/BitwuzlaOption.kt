@file:Suppress("MagicNumber")

package org.ksmt.solver.bitwuzla.bindings

/**
 * The configuration options supported by Bitwuzla.
 *
 * Options that list string values can be configured via
 * [Native.bitwuzlaSetOptionStr]. Options with integer configuration values are
 * configured via [Native.bitwuzlaSetOption].
 *
 * For all options, the current configuration value can be queried via
 * [Native.bitwuzlaGetOption].
 * Options with string configuration values internally represent these
 * values as enum values.
 * For these options, [Native.bitwuzlaGetOption] will return such an enum value.
 * Use [Native.bitwuzlaGetOptionStr] to query enum options for the corresponding
 * string representation.
 */
enum class BitwuzlaOption(val value: Int) {

    /* --------------------------- General Options --------------------------- */

    /** 
     * **Configure the solver engine.**
     *
     * Values:
     *  * **aigprop**:
     *    The propagation-based local search QF_BV engine that operates on the
     *    bit-blasted formula (the AIG circuit layer).
     *  * **fun** **(default)**:
     *    The default engine for all combinations of QF_AUFBVFP, uses lemmas on
     *    demand for QF_AUFBVFP, and eager bit-blasting (optionally with local
     *    searchin a sequential portfolio) for QF_BV.
     *  * **prop**:
     *    The propagation-based local search QF_BV engine.
     *  * **sls**:
     *     The stochastic local search QF_BV engine.
     *  * **quant**:
     *    The quantifier engine.
     */
    BITWUZLA_OPT_ENGINE(0),

    /** 
     * **Use non-zero exit codes for sat and unsat results.**
     *
     * When enabled, use Bitwuzla exit codes:
     * * [BitwuzlaResult.BITWUZLA_SAT]
     * * [BitwuzlaResult.BITWUZLA_UNSAT]
     * * [BitwuzlaResult.BITWUZLA_UNKNOWN]
     *
     * When disabled, return 0 on success (sat, unsat, unknown), and a non-zero
     * exit code otherwise.
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     */
    BITWUZLA_OPT_EXIT_CODES(1),

    /** 
     * **Configure input file format.**
     *
     * If unspecified, Bitwuzla will autodetect the input file format.
     *
     * Values:
     *  * **none** **(default)**: Auto-detect input file format.
     *  * **btor**: BTOR format
     *  * **btor2**: BTOR2 format
     *  * **smt2**: SMT-LIB v2 format
     */
    BITWUZLA_OPT_INPUT_FORMAT(2),

    /** 
     * **Incremental solving.**
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     * Note:
     * * Enabling this option turns off some optimization techniques.
     * * Enabling/disabling incremental solving after bitwuzla_check_sat()
     *   has been called is not supported.
     * * This option cannot be enabled in combination with option [BITWUZLA_OPT_PP_UNCONSTRAINED_OPTIMIZATION].
     */
    BITWUZLA_OPT_INCREMENTAL(3),

    /** 
     * **Log level.**
     *
     * Values:
     *  * An unsigned integer value (**default**: 0).
     */
    BITWUZLA_OPT_LOGLEVEL(4),

    /** 
     * **Configure output number format for bit-vector values.**
     *
     * If unspecified, Bitwuzla will use BTOR format.
     *
     * Values:
     *  * **aiger**: AIGER ascii format
     *  * **aigerbin**: AIGER binary format
     *  * **btor** **(default)**: BTOR format
     *  * **smt2**: SMT-LIB v2 format
     */
    BITWUZLA_OPT_OUTPUT_FORMAT(5),

    /** 
     * **Configure output number format for bit-vector values.**
     *
     * If unspecified, Bitwuzla will use binary representation.
     *
     * Values:
     *  * **bin** **(default)**:
     *  Binary number format.
     *  * **hex**:
     *  Hexadecimal number format.
     *  * **dec**:
     *  Decimal number format.
     */
    BITWUZLA_OPT_OUTPUT_NUMBER_FORMAT(6),

    /** 
     * **Pretty printing.**
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     */
    BITWUZLA_OPT_PRETTY_PRINT(7),

    /** 
     * **Print DIMACS.**
     *
     * Print the CNF sent to the SAT solver in DIMACS format to stdout.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     */
    BITWUZLA_OPT_PRINT_DIMACS(8),

    /** 
     * **Model generation.**
     *
     * Values:
     *  * **1**: enable, generate model for assertions only
     *  * **2**: enable, generate model for all created terms
     *  * **0**: disable **(default)**
     *
     * **Note**: This option cannot be enabled in combination with option [BITWUZLA_OPT_PP_UNCONSTRAINED_OPTIMIZATION].
     */
    BITWUZLA_OPT_PRODUCE_MODELS(9),

    /** 
     * **Unsat core generation.**
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     */
    BITWUZLA_OPT_PRODUCE_UNSAT_CORES(10),

    /** 
     * **Configure the SAT solver engine.**
     *
     * Values:
     *  * **cadical** **(default)**:
     *    [CaDiCaL](https://github.com/arminbiere/cadical)
     *  * **cms**:
     *    [CryptoMiniSat](https://github.com/msoos/cryptominisat)
     *  * **gimsatul**:
     *    [Gimsatul](https://github.com/arminbiere/gimsatul)
     *  * **kissat**:
     *    [Kissat](https://github.com/arminbiere/kissat)
     *  * **lingeling**:
     *    [Lingeling](https://github.com/arminbiere/lingeling)
     *  * **minisat**:
     *    [MiniSat](https://github.com/niklasso/minisat)
     *  * **picosat**:
     *    [PicoSAT](http://fmv.jku.at/picosat/)
     */
    BITWUZLA_OPT_SAT_ENGINE(11),

    /** 
     * **Seed for random number generator.**
     *
     * Values:
     *  * An unsigned integer value (**default**: 0).
     */
    BITWUZLA_OPT_SEED(12),

    /** 
     * **Verbosity level.**
     *
     * Values:
     *  * An unsigned integer value <= 4 (**default**: 0).
     */
    BITWUZLA_OPT_VERBOSITY(13),

    /* -------------- Rewriting/Preprocessing Options (Expert) --------------- */

    /** 
     * **Ackermannization preprocessing.**
     *
     * Eager addition of Ackermann constraints for function applications.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_ACKERMANN(14),

    /** 
     * **Beta reduction preprocessing.**
     *
     * Eager elimination of lambda terms via beta reduction.
     *
     * Values:
     *  * **none** **(default)**:
     *    Disable beta reduction preprocessing.
     *  * **fun**:
     *    Only beta reduce functions that do not represent array stores.
     *  * **all**:
     *    Only beta reduce all functions, including array stores.
     *
     *  **Warning**: This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_BETA_REDUCE(15),

    /** 
     * **Eliminate bit-vector extracts (preprocessing).**
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_ELIMINATE_EXTRACTS(16),

    /** 
     * **Eliminate ITEs (preprocessing).**
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_ELIMINATE_ITES(17),

    /** 
     * **Extract lambdas (preprocessing).**
     *
     * Extraction of common array patterns as lambda terms.
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_EXTRACT_LAMBDAS(18),

    /** 
     * **Merge lambda terms (preprocessing).**
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_MERGE_LAMBDAS(19),

    /** 
     * **Non-destructive term substitutions.**
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_NONDESTR_SUBST(20),

    /** 
     * **Normalize bit-vector addition (global).**
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_NORMALIZE_ADD(21),

    /** 
     * **Boolean skeleton preprocessing.**
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_SKELETON_PREPROC(22),

    /** 
     * **Unconstrained optimization (preprocessing).**
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_UNCONSTRAINED_OPTIMIZATION(23),

    /** 
     * **Variable substitution preprocessing.**
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure preprocessing.
     */
    BITWUZLA_OPT_PP_VAR_SUBST(24),

    /** 
     * **Propagate bit-vector extracts over arithmetic bit-vector operators.**
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_RW_EXTRACT_ARITH(25),

    /** 
     * **Rewrite level.**
     *
     * Values:
     * * **0**: no rewriting
     * * **1**: term level rewriting
     * * **2**: term level rewriting and basic preprocessing
     * * **3**: term level rewriting and full preprocessing **(default)**
     *
     * **Note**: Configuring the rewrite level after terms have been created
     *       is not allowed.
     *
     *  **Warning**: This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_RW_LEVEL(26),

    /** 
     * **Normalize bit-vector operations.**
     *
     * Normalize bit-vector addition, multiplication and bit-wise and.
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_RW_NORMALIZE(27),

    /** 
     * **Normalize bit-vector addition (local).**
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_RW_NORMALIZE_ADD(28),

    /** 
     * **Simplify constraints on construction.**
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_RW_SIMPLIFY_CONSTRAINTS(29),

    /** 
     * **Eliminate bit-vector SLT nodes.**
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_RW_SLT(30),

    /** 
     * **Sort the children of AIG nodes by id.**
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_RW_SORT_AIG(31),

    /** 
     * **Sort the children of adder and multiplier circuits by id.**
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_RW_SORT_AIGVEC(32),

    /** 
     * **Sort the children of commutative operations by id.**
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure rewriting.
     */
    BITWUZLA_OPT_RW_SORT_EXP(33),

/* --------------------- Fun Engine Options (Expert) --------------------- */

    /** 
     * **Function solver engine:
     *    Dual propagation optimization.**
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the fun solver engine.
     */
    BITWUZLA_OPT_FUN_DUAL_PROP(34),

    /** 
     * **Function solver engine:
     *    Assumption order for dual propagation optimization.**
     *
     * Set order in which inputs are assumed in the dual propagation clone.
     *
     * Values:
     *  * **just** **(default)**:
     *    Order by score, highest score first.
     *  * **asc**:
     *    Order by input id, ascending.
     *  * **desc**:
     *    Order by input id, descending.
     *
     *  **Warning**: This is an expert option to configure the func solver engine.
     */
    BITWUZLA_OPT_FUN_DUAL_PROP_QSORT(35),

    /** 
     * **Function solver engine:
     *    Eager lemmas.**
     *
     * Configure mode for eager lemma generation.
     *
     * Values:
     *  * **none**:
     *    Do not generate lemmas eagerly (generate one single lemma per
     *    refinement iteration).
     *  * **conf** **(default)**:
     *    Only generate lemmas eagerly until the first conflict dependent on
     *    another conflict is found.
     *  * **all**:
     *    In each refinement iteration, generate lemmas for all conflicts.
     *
     *  **Warning**: This is an expert option to configure the func solver engine.
     */
    BITWUZLA_OPT_FUN_EAGER_LEMMAS(36),

    /** 
     * **Function solver engine:
     *    Lazy synthesis.**
     *
     * Configure lazy synthesis (to bit-level) of bit-vector expressions.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the func solver engine.
     */
    BITWUZLA_OPT_FUN_LAZY_SYNTHESIZE(37),

    /** 
     * **Function solver engine:
     *    Justification optimization.**
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the func solver engine.
     */
    BITWUZLA_OPT_FUN_JUST(38),

    /** 
     * **Function solver engine:
     *    Justification optimization heuristic.**
     *
     * Configure heuristic to determine path selection for justification
     * optimization.
     *
     * Values:
     *  * **applies** **(default)**:
     *    Choose branch with minimum number of applies.
     *  * **depth**:
     *    Choose branch with minimum depth.
     *  * **left**:
     *    Always choose left branch.
     *
     *  **Warning**: This is an expert option to configure the func solver engine.
     */
    BITWUZLA_OPT_FUN_JUST_HEURISTIC(39),

    /** 
     * **Function solver engine:
     *    Propagation-based local search sequential portfolio.**
     *
     * When function solver engine is enabled, configure propagation-based local
     * search solver engine as preprocessing step within sequential portfolio
     * setting.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the func solver engine.
     */
    BITWUZLA_OPT_FUN_PREPROP(40),

    /** 
     * **Function solver engine:
     *    Stochastic local search sequential portfolio.**
     *
     * When function solver engine is enabled, configure stochastic local
     * search solver engine as preprocessing step within sequential portfolio
     * setting.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the func solver engine.
     */
    BITWUZLA_OPT_FUN_PRESLS(41),

    /** 
     * **Function solver engine:
     *    Represent store as lambda.**
     *
     * Represent array stores as lambdas.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the func solver engine.
     */
    BITWUZLA_OPT_FUN_STORE_LAMBDAS(42),

/* --------------------- SLS Engine Options (Expert) --------------------- */

    /** 
     * **Stochastic local search solver engine:
     *    Justification-based path selection.**
     *
     * Configure justification-based path selection for SLS engine.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_JUST(43),

    /** 
     * **Stochastic local search solver engine:
     *    Group-wise moves.**
     *
     * Configure group-wise moves for SLS engine. When enabled, rather than
     * changing the assignment of one single candidate variable, all candidates
     * are set at the same time (using the same strategy).
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_GW(44),

    /** 
     * **Stochastic local search solver engine:
     *    Incremental move test.**
     *
     * Configure that during best move selection, the previous best neighbor
     * for the current candidate is used for neighborhood exploration rather
     * than its current assignment.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_INC_MOVE_TEST(45),

    /** 
     * **Stochastic local search solver engine:
     *    Propagation moves.**
     *
     * Configure propagation moves, chosen with a ratio of number of propagation
     * moves [BITWUZLA_OPT_SLS_MOVE_PROP_NPROPS] to regular SLS moves
     * [BITWUZLA_OPT_SLS_MOVE_PROP_NSLSS].
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_PROP(46),

    /** 
     * **Stochastic local search solver engine:
     *    Force random walks.**
     *
     * Configure that random walks are forcibly chosen as recovery moves in case
     * of conflicts when a propagation move is performed (rather than performing
     * a regular SLS move).
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_PROP_FORCE_RW(47),

    /** 
     * **Stochastic local search solver engine:
     *    Number of propagation moves.**
     *
     * Configure the number of propagation moves to be performed when propagation
     * moves are enabled. Propagation moves are chosen with a ratio of
     * [BITWUZLA_OPT_SLS_MOVE_PROP_NPROPS] to
     * [BITWUZLA_OPT_SLS_MOVE_PROP_NSLSS].
     *
     * Values:
     *  * An unsigned integer value (**default**: 1)
     *
     * @see BITWUZLA_OPT_SLS_MOVE_PROP
     * @see BITWUZLA_OPT_SLS_MOVE_PROP_NSLSS
     *
     *  **Warning**: This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_PROP_NPROPS(48),

    /** 
     * **Stochastic local search solver engine:
     *    Number of regular SLS moves.**
     *
     * Configure the number of regular SLS moves to be performed when propagation
     * moves are enabled. Propagation moves are chosen with a ratio of
     * [BITWUZLA_OPT_SLS_MOVE_PROP_NPROPS] to
     * [BITWUZLA_OPT_SLS_MOVE_PROP_NSLSS].
     *
     * Values:
     *  * An unsigned integer value (**default**: 1)
     *
     *   @see BITWUZLA_OPT_SLS_MOVE_PROP
     *   @see BITWUZLA_OPT_SLS_MOVE_PROP_NPROPS
     *
     *  **Warning**: This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_PROP_NSLSS(49),

    /** 
     * **Stochastic local search solver engine:
     *    Randomize all candidates.**
     *
     * Configure the randomization of all candidate variables (rather than just
     * a single randomly selected one) in case no best move has been found.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_RAND_ALL(50),

    /** 
     * **Stochastic local search solver engine:
     *    Randomize bit ranges.**
     *
     * Configure the randomization of bit ranges (rather than all bits) of
     * candidate variable(s) in case no best move has been found.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_RAND_RANGE(51),

    /** 
     * **Stochastic local search solver engine:
     *    Random walk.**
     *
     * Configure random walk moves, where one out of all possible neighbors is
     * randomly selected (with given probability
     * [BITWUZLA_OPT_SLS_PROB_MOVE_RAND_WALK]) for a randomly selected
     * candidate variable.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     * @see BITWUZLA_OPT_SLS_PROB_MOVE_RAND_WALK
     *
     *  **Warning**: This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_RAND_WALK(52),

    /** 
     * **Stochastic local search solver engine:
     *    Range-wise bit-flip moves.**
     *
     * Configure range-wise bit-flip moves for SLS engine. When enabled, try
     * range-wise bit-flips when selecting moves, where bits within all ranges
     * from 2 to the bit-width (starting from the LSB) are flipped at once.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_RANGE(53),

    /** 
     * **Stochastic local search solver engine:
     *    Segment-wise bit-flip moves.**
     *
     * Configure range-wise bit-flip moves for SLS engine. When enabled, try
     * segment-wise bit-flips when selecting moves, where bits within segments
     * of multiples of 2 are flipped at once.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_MOVE_SEGMENT(54),

    /** 
     * **Stochastic local search solver engine:
     *    Probability for random walks.**
     *
     * Configure the probability with which a random walk is chosen if random
     * walks are enabled.
     *
     * Values:
     *  * An unsigned integer value <= 1000 (= 100%) (**default**: 100)
     *
     * @see BITWUZLA_OPT_SLS_MOVE_RAND_WALK
     *
     *  **Warning**: This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_PROB_MOVE_RAND_WALK(55),

    /** 
     * **Stochastic local search solver engine:
     *    Number of bit flips.**
     *
     * Configure the number of bit flips used as a limit for the SLS engine.
     *
     * Values:
     *  * An unsigned integer value, no limit if 0 (**default**: 0).
     *
     *  **Warning**: This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_NFLIPS(56),

    /** 
     * **Stochastic local search solver engine:
     *    Move strategy.**
     *
     * Configure the move selection strategy for the SLS engine.
     *
     * Values:
     *  * **best** **(default)**:
     *    Choose best score improving move.
     *  * **walk**:
     *    Choose random walk weighted by score.
     *  * **first**:
     *    Choose first best move (no matter if any other move is better).
     *  * **same**:
     *    Determine move as best move even if its score is not better but the
     *    same as the score of the previous best move.
     *  * **prop**:
     *    Choose propagation move (and recover with SLS move in case of conflict).
     *
     *  **Warning**: This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_STRATEGY(57),

    /** 
     * **Stochastic local search solver engine:
     *    Restarts.**
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_USE_RESTARTS(58),

    /** 
     * **Stochastic local search solver engine:
     *    Bandit scheme.**
     *
     * Configure bandit scheme heuristic for selecting root constraints.
     * If disabled, root constraints are selected randomly.
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure the sls solver engine.
     */
    BITWUZLA_OPT_SLS_USE_BANDIT(59),

    /* -------------------- Prop Engine Options (Expert) --------------------- */

    /** 
     * **Propagation-based local search solver engine:
     *    Value computation for xor.**
     *
     * When enabled, detect arithmetic right shift operations (are rewritten on
     * construction) and use value computation for ashr.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_ASHR(60),

    /** 
     * **Propagation-based local search solver engine:
     *    Constant bits.**
     *
     * Configure constant bit propagation (requries bit-blasting to AIG).
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_CONST_BITS(61),

    /** 
     * **Propagation-based local search solver engine:
     *    Domain propagators.**
     *
     * Configure the use of domain propagators for determining constant bits
     * (instead of bit-blastin to AIG).
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_CONST_DOMAINS(62),

    /** 
     * **Propagation-based local search solver engine:
     *    Entailed propagations.**
     *
     * Maintain a work queue with entailed propagations.
     * If enabled, propagations from this queue are propagated before randomly
     * choosing a yet unsatisfied path from the root.
     *
     * Values:
     *
     *  * **off** **(default)**:
     *    Disable strategy.
     *  * **all**:
     *    Propagate all entailed propagations.
     *  * **first**:
     *    Process only the first entailed propagation.
     *  * **last**:
     *    Process only the last entailed propagation.
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_ENTAILED(63),

    /** 
     * **Propagation-based local search solver engine:
     *    Delta for flipping ite conditions with constant branches.**
     *
     * Configure the delta by which [BITWUZLA_OPT_PROP_PROB_FLIP_COND_CONST] is
     * decreased or increased after a limit
     * [BITWUZLA_OPT_PROP_FLIP_COND_CONST_NPATHSEL] is reached.
     *
     * Values:
     *  * A signed integer value (**default**: 100).
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_FLIP_COND_CONST_DELTA(64),

    /** 
     * **Propagation-based local search solver engine:
     *    Limit for flipping ite conditions with constant branches.**
     *
     * Configure the limit for how often the path to the condition for ite
     * operations with constant branches may be selected before
     * [BITWUZLA_OPT_PROP_PROB_FLIP_COND_CONST] is decreased or increased by
     * [BITWUZLA_OPT_PROP_FLIP_COND_CONST_DELTA].
     *
     * Values:
     *  * A signed integer value (**default**: 500).
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_FLIP_COND_CONST_NPATHSEL(65),

    /** 
     * **Propagation-based local search solver engine:
     *    Infer bounds for inequalities for value computation.**
     *
     * When enabled, infer bounds for value computation for inequalities based on
     * satisfied top level inequalities.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_INFER_INEQ_BOUNDS(66),

    /** 
     * **Propagation-based local search solver engine:
     *    No move on conflict.**
     *
     * When enabled, no move is performed when running into a conflict during
     * value computation.
     *
     * **Note**: This is the default behavior for the SLS engine when propagation
     *       moves are enabled, where a conflict triggers a recovery by means
     *       of a regular SLS move.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_NO_MOVE_ON_CONFLICT(67),

    /** 
     * **Propagation-based local search solver engine:
     *    Number of propagations.**
     *
     * Configure the number of propagations used as a limit for the
     * propagation-based local search solver engine. No limit if 0.
     *
     * Values:
     *  * An unsigned integer value (**default**: 0).
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_NPROPS(68),

    /** 
     * **Propagation-based local search solver engine:
     *    Number of updates.**
     *
     * Configure the number of model value updates used as a limit for the
     * propagation-based local search solver engine. No limit if 0.
     *
     * Values:
     *  * An unsigned integer value (**default**: 0).
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_NUPDATES(69),

    /** 
     * **Propagation-based local search solver engine:
     *    Path selection.**
     *
     * Configure mode for path selection.
     *
     * Values:
     *  * **essential** **(default)**:
     *    Select path based on essential inputs.
     *  * **random**:
     *    Select path randomly.
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PATH_SEL(70),

    /** 
     * **Propagation-based local search solver engine:
     *    Probability for producing inverse rather than consistent values.**
     *
     * Values:
     *  * An unsigned integer value <= 1000 (= 100%) (**default**: 0).
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_FALLBACK_RANDOM_VALUE(71),

    /** 
     * **Propagation-based local search solver engine:
     *    Probability for flipping one of the don't care bits for ands.**
     *
     * Configure the probability with which to keep the current assignement of
     * the operand to a bit-vector and with max one bit flipped (rather than
     * fully randomizing the assignment) when selecting an inverse or consistent
     * value.
     *
     * Values:
     *  * An unsigned integer value <= 1000 (= 100%) (**default**: 0).
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_AND_FLIP(72),

    /** 
     * **Propagation-based local search solver engine:
     *    Probability for using the current assignment with one bit flipped for
     *    equalities.**
     *
     * Configure the probability with which the current assignment of an operand
     * to a disequality is kept with just a single bit flipped (rather than fully
     * randomizing the assignment) when selecting an inverse or consistent value.
     *
     * Values:
     *  * An unsigned integer value <= 1000 (= 100%) (**default**: 0).
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_EQ_FLIP(73),

    /** 
     * **Propagation-based local search solver engine:
     *    Probability for flipping ite condition.**
     *
     * Configure the probability with which to select the path to the condition
     * (in case of an ite operation) rather than the enabled branch during down
     * propagation).
     *
     * Values:
     *  * An unsigned integer value <= 1000 (= 100%) (**default**: 100).
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_FLIP_COND(74),

    /** 
     * **Propagation-based local search solver engine:
     *    Probability for flipping ite condition with constant branches.**
     *
     * Configure the probability with which to select the path to the condition
     * (in case of an ite operation) rather than the enabled branch during down
     * propagation) if either the 'then' or 'else' branch is constant.
     *
     * Values:
     *  * An unsigned integer value <= 1000 (= 100%) (**default**: 100).
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_FLIP_COND_CONST(75),

    /** 
     * **Propagation-based local search solver engine:
     *    Probability for selecting random input.**
     *
     * Configure the probability with which to select a random input instead of
     * an essential input when selecting the path.
     *
     * Values:
     *  * An unsigned integer value <= 1000 (= 100%) (**default**: 0).
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_RANDOM_INPUT(76),

    /** 
     * **Propagation-based local search solver engine:
     *    Probability for flipping one of the don't care bits for extracts.**
     *
     * Configure the probability with which to flip one of the don't care bits of
     * the current assignment of the operand to a bit-vector extract (when the
     * asignment is kept) when selecting an inverse or consistent value.
     *
     * Values:
     *  * An unsigned integer value <= 1000 (= 100%) (**default**: 0).
     *
     * @see BITWUZLA_OPT_PROP_PROB_SLICE_KEEP_DC
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_SLICE_FLIP(77),

    /** 
     * **Propagation-based local search solver engine:
     *    Probability for keeping the value of don't care bits for extracts.**
     *
     * Configure the probability with which to keep the current value of don't
     * care bits of an extract operation (rather than fully randomizing them)
     * when selecting an inverse or consistent value.
     *
     * Values:
     *  * An unsigned integer value <= 1000 (= 100%) (**default**: 500).
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_SLICE_KEEP_DC(78),

    /** 
     * **Propagation-based local search solver engine:
     *    Probability for inverse values.**
     *
     * Configure the probability with which to choose an inverse value over a
     * consistent value when aninverse value exists.
     *
     * Values:
     *  * An unsigned integer value <= 1000 (= 100%) (**default**: 990).
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_PROB_USE_INV_VALUE(79),

    /** 
     * **Propagation-based local search solver engine:
     *    Bandit scheme.**
     *
     * Configure bandit scheme heuristic for selecting root constraints.
     * If enabled, root constraint selection via bandit scheme is based on a
     * scoring scheme similar to the one used in the SLS engine.
     * If disabled, root constraints are selected randomly.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_USE_BANDIT(80),

    /** 
     * **Propagation-based local search solver engine:
     *    Inverse value computation for inequalities over concats.**
     *
     * When enabled, use special inverse value computation for inequality over
     * concats.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_USE_INV_LT_CONCAT(81),

    /** 
     * **Propagation-based local search solver engine:
     *    Restarts.**
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_USE_RESTARTS(82),

    /** 
     * **Propagation-based local search solver engine:
     *    Value computation for sign extension.**
     *
     * When enabled, detect sign extension operations (are rewritten on
     * construction) and use value computation for sign extension.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_SEXT(83),

    /** 
     * **Propagation-based local search solver engine:
     *    Skip if no progress.**
     *
     * When enabled, moves that make no progress, that is, that produce a target
     * value that is the seame as the current assignment of a variable, are
     * skipped.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_SKIP_NO_PROGRESS(84),

    /** 
     * **Propagation-based local search solver engine:
     *    Value computation for xor.**
     *
     * When enabled, detect xor operations (are rewritten on construction) and
     * use value computation for xor.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the prop solver engine.
     */
    BITWUZLA_OPT_PROP_XOR(85),

    /* ------------------- AigProp Engine Options (Expert) ------------------- */

    /** 
     * **AIG-level propagation-based local search solver engine:
     *    Number of propagations.**
     *
     * Configure the number of propagations used as a limit for the
     * propagation-based local search solver engine. No limit if 0.
     *
     * Values:
     *  * An unsigned integer value (**default**: 0).
     *
     *  **Warning**: This is an expert option to configure the aigprop solver engine.
     */
    BITWUZLA_OPT_AIGPROP_NPROPS(86),

    /** 
     * **AIG-level propagation-based local search solver engine:
     *    Bandit scheme.**
     *
     * Configure bandit scheme heuristic for selecting root constraints.
     * If enabled, root constraint selection via bandit scheme is based on a
     * scoring scheme similar to the one used in the SLS engine.
     * If disabled, root constraints are selected randomly.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the aigprop solver engine.
     */
    BITWUZLA_OPT_AIGPROP_USE_BANDIT(87),

    /** 
     * **AIG-level propagation-based local search solver engine:
     *    Restarts.**
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option to configure the aigprop solver engine.
     */
    BITWUZLA_OPT_AIGPROP_USE_RESTARTS(88),

    /* ----------------- Quantifier Eninge Options (Expert) ------------------ */

    /** 
     * **Quantifier solver engine:
     *    Constructive Equality Resolution.**
     *
     * Configure the use of Constructive Equality Resolution simplification in
     * the quantifier solver engine.
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure the quantifier solver
     *  engine.
     */
    BITWUZLA_OPT_QUANT_CER(89),

    /** 
     * **Quantifier solver engine:
     *    Destructive Equality Resolution.**
     *
     * Configure the use of Destructive Equality Resolution simplification in
     * the quantifier solver engine.
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure the quantifier solver
     *  engine.
     */
    BITWUZLA_OPT_QUANT_DER(90),

    /** 
     * **Quantifier solver engine:
     *    Dual solver.**
     *
     * Configure the use of the dual (negated) version of the quantified formula.
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure the quantifier solver
     *  engine.
     */
    BITWUZLA_OPT_QUANT_DUAL_SOLVER(91),

    /** 
     * **Quantifier solver engine:
     *    Miniscoping.**
     *
     * Configure the use of miniscoping in the quantifier solver engine.
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure the quantifier solver
     *  engine.
     */
    BITWUZLA_OPT_QUANT_MINISCOPE(92),

    /** 
     * **Quantifier solver engine:
     *    Synthesis mode.**
     *
     * Configure mode for synthesizing Skolem functions.
     *
     * Values:
     * * **none**:
     *   Do not synthesize skolem functions (use model values for instantiation).
     * * **el**:
     *   Use enumerative learning to synthesize skolem functions.
     * * **elmc**:
     *   Use enumerative learning modulo the predicates in the cone of influence
     *   of the existential variables to synthesize skolem functions.
     * * **elelmc**:
     *   Chain `el` and `elmc` approaches to synthesize skolem functions.
     * * **elmr** **(default)**:
     *   Use enumerative learning modulo the given root constraints to synthesize
     *   skolem functions.
     *
     *  **Warning**: This is an expert option to configure the quantifier solver
     *  engine.
     */
    BITWUZLA_OPT_QUANT_SYNTH(93),

    /** 
     * **Quantifier solver engine:
     *    Update model with respect to synthesized skolem.**
     *
     * Configure to update the current model with respect to the synthesized
     * skolem function if enabled.
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure the quantifier solver
     *  engine.
     */
    BITWUZLA_OPT_QUANT_FIXSYNTH(94),

    /** 
     * **Quantifier solver engine:
     *    Base case for ITE model.**
     *
     * Configure the base case of a concrete model for ITEs. Constant if enabled,
     * else undefined.
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure the quantifier solver
     *  engine.
     */
    BITWUZLA_OPT_QUANT_SYNTH_ITE_COMPLETE(95),

    /** 
     * **Quantifier solver engine:
     *    Limit for synthesis.**
     *
     * Configure the limit of enumerated expressions for the enumerative learning
     * synthesis algorithm implemented in the quantified solver engine.
     *
     * Values:
     *  * An unsigned integer value (**default**: 10000).
     *
     *  **Warning**: This is an expert option to configure the quantifier solver
     *  engine.
     */
    BITWUZLA_OPT_QUANT_SYNTH_LIMIT(96),

    /** 
     * **Quantifier solver engine:
     *    Quantifier instantiation.**
     *
     * Configure the generalization of quantifier instantiations via enumerative
     * learning.
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option to configure the quantifier solver
     *  engine.
     */
    BITWUZLA_OPT_QUANT_SYNTH_QI(97),

    /* ------------------------ Other Expert Options ------------------------- */

    /** 
     * **Check model (debug only).**
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option.
     */
    BITWUZLA_OPT_CHECK_MODEL(98),

    /** 
     * **Check result when unconstrained optimization is enabled (debug only).**
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option.
     */
    BITWUZLA_OPT_CHECK_UNCONSTRAINED(99),

    /** 
     * **Check unsat assumptions (debug only).**
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option.
     */
    BITWUZLA_OPT_CHECK_UNSAT_ASSUMPTIONS(100),

    /** 
     * **Interpret sorts introduced with declare-sort as bit-vectors of given
     *    width.**
     *
     * Disabled if zero.
     *
     * Values:
     *  * An unsigned integer value (**default**: 0).
     *
     *  **Warning**: This is an expert option.
     */
    BITWUZLA_OPT_DECLSORT_BV_WIDTH(101),

    /** 
     * **Share partial models determined via local search with bit-blasting
     *    engine.**
     *
     * This option is only effective when local search engines are combined with
     * the bit-blasting engine in a sequential portfolio.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option.
     */
    BITWUZLA_OPT_LS_SHARE_SAT(102),

    /** 
     * **Interactive parsing mode.**
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option.
     */
    BITWUZLA_OPT_PARSE_INTERACTIVE(103),

    /** 
     * **Use CaDiCaL's freeze/melt.**
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option.
     */
    BITWUZLA_OPT_SAT_ENGINE_CADICAL_FREEZE(104),

    /** 
     * **Lingeling fork mode.**
     *
     * Values:
     *  * **1**: enable **(default)**
     *  * **0**: disable
     *
     *  **Warning**: This is an expert option.
     */
    BITWUZLA_OPT_SAT_ENGINE_LGL_FORK(105),

    /** 
     * **Number of threads to use in the SAT solver.**
     *
     * This option is only effective for SAT solvers with support for
     * multi-threading.
     *
     * Values:
     *  * An unsigned integer value > 0 (**default**: 1).
     *
     *  **Warning**: This is an expert option.
     */
    BITWUZLA_OPT_SAT_ENGINE_N_THREADS(106),

    /** 
     * **Enable SMT-COMP mode.**
     *
     * Parser only option. Only effective when an SMT2 input file is parsed.
     *
     * Values:
     *  * **1**: enable
     *  * **0**: disable **(default)**
     *
     *  **Warning**: This is an expert option.
     */
    BITWUZLA_OPT_SMT_COMP_MODE(107),

    BITWUZLA_OPT_NUM_OPTS(108);

    companion object {
        private val valueMapping = BitwuzlaOption.values().associateBy { it.value }
        private val nameMapping = BitwuzlaOption.values().associateBy { it.name }
        fun fromValue(value: Int): BitwuzlaOption = valueMapping.getValue(value)
        fun forName(name: String): BitwuzlaOption? = nameMapping[name]
    }
}
