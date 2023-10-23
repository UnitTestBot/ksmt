@file:Suppress("MagicNumber")

package org.ksmt.solver.bitwuzla.bindings

/**
 * The term kind.
 */
enum class BitwuzlaKind(val value: Int) {
    /** First order constant. */
    BITWUZLA_KIND_CONSTANT(0),

    /** Constant array. */
    BITWUZLA_KIND_CONST_ARRAY(1),

    /** Value. */
    BITWUZLA_KIND_VALUE(2),

    /** Bound variable. */
    BITWUZLA_KIND_VARIABLE(3),

    /**
     * Boolean and.
     *
     * SMT-LIB: `and`
     */
    BITWUZLA_KIND_AND(4),

    /**
     * Disequality.
     *
     * SMT-LIB: `distinct`
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_DISTINCT(5),

    /**
     * Equality.
     *
     * SMT-LIB: `=`
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_EQUAL(6),

    /**
     * Boolean if and only if.
     *
     * SMT-LIB: `=`
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_IFF(7),

    /**
     * Boolean implies.
     *
     * SMT-LIB: `=>`
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_IMPLIES(8),

    /**
     * Boolean not.
     *
     * SMT_LIB: `not`
     *
     * Number of Arguments: 1
     */
    BITWUZLA_KIND_NOT(9),

    /**
     * Boolean or.
     *
     * SMT_LIB: `or`
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_OR(10),

    /**
     * Boolean xor.
     *
     * SMT_LIB: `xor`
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_XOR(11),

    //// Core
    /**
     * If-then-else.
     *
     * SMT_LIB: `ite`
     *
     * Number of Arguments: 3
     */
    BITWUZLA_KIND_ITE(12),

    //// Quantifiers
    /**
     * Existential quantification.
     *
     * SMT_LIB: `exists`
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_EXISTS(13),

    /**
     * Universal quantification.
     *
     * SMT_LIB: `forall`
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_FORALL(14),

    //// Functions
    /**
     * Function application.
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_APPLY(15),

    /**
     * Lambda.
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_LAMBDA(16),

    //// Arrays
    /**
     * Array select.
     *
     * SMT_LIB: `select`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_ARRAY_SELECT(17),

    /**
     * Array store.
     *
     * SMT_LIB: `store`
     *
     * Number of Arguments: 3
     */
    BITWUZLA_KIND_ARRAY_STORE(18),

    //// Bit-vectors
    /**
     * Bit-vector addition.
     *
     * SMT_LIB: `bvadd`
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_BV_ADD(19),

    /**
     * Bit-vector and.
     *
     * SMT_LIB: `bvand`
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_BV_AND(20),

    /**
     * Bit-vector arithmetic right shift.
     *
     * SMT_LIB: `bvashr`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_ASHR(21),

    /**
     * Bit-vector comparison.
     *
     * SMT_LIB: `bvcomp`
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_BV_COMP(22),

    /**
     * Bit-vector concat.
     *
     * SMT_LIB: `concat`
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_BV_CONCAT(23),

    /**
     * Bit-vector decrement.
     *
     * Decrement by one.
     *
     * Number of arguments: 1
     */
    BITWUZLA_KIND_BV_DEC(24),

    /**
     * Bit-vector increment.
     *
     * Increment by one.
     *
     * Number of Arguments: 1
     */
    BITWUZLA_KIND_BV_INC(25),

    /**
     * Bit-vector multiplication.
     *
     * SMT_LIB: `bvmul`
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_BV_MUL(26),

    /**
     * Bit-vector nand.
     *
     * SMT_LIB: `bvnand`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_NAND(27),

    /**
     * Bit-vector negation (two's complement).
     *
     * SMT_LIB: `bvneg`
     *
     * Number of Arguments: 1
     */
    BITWUZLA_KIND_BV_NEG(28),

    /**
     * Bit-vector nor.
     *
     * SMT_LIB: `bvnor`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_NOR(29),

    /**
     * Bit-vector not (one's complement).
     *
     * SMT_LIB: `bvnot`
     *
     * Number of Arguments: 1
     */
    BITWUZLA_KIND_BV_NOT(30),

    /**
     * Bit-vector or.
     *
     * SMT_LIB: `bvor`
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_BV_OR(31),

    /**
     * Bit-vector and reduction.
     *
     * Bit-wise `and` (reduction), all bits are `and`'ed together into a single
     * bit. This corresponds to bit-wise `and` reduction as known from Verilog.
     *
     * Number of Arguments: 1
     */
    BITWUZLA_KIND_BV_REDAND(32),

    /**
     * Bit-vector reduce or.
     *
     * Bit-wise `or` (reduction), all bits are `or`'ed together into a single
     * bit. This corresponds to bit-wise `or` reduction as known from Verilog.
     *
     * Number of Arguments: 1
     */
    BITWUZLA_KIND_BV_REDOR(33),

    /**
     * Bit-vector reduce xor.
     *
     * Bit-wise `xor` (reduction), all bits are `xor`'ed together into a single
     * bit. This corresponds to bit-wise `xor` reduction as known from Verilog.
     *
     * Number of Arguments: 1
     */
    BITWUZLA_KIND_BV_REDXOR(34),

    /**
     * Bit-vector rotate left (not indexed).
     *
     * This is a non-indexed variant of SMT-LIB `rotate_left`.
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_ROL(35),

    /**
     * Bit-vector rotate right.
     *
     * This is a non-indexed variant of SMT-LIB `rotate_right`.
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_ROR(36),

    /**
     * Bit-vector signed addition overflow test.
     *
     * Predicate indicating if signed addition produces an overflow.
     *
     * SMT_LIB: `bvsaddo`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_SADD_OVERFLOW(37),

    /**
     * Bit-vector signed division overflow test.
     *
     * Predicate indicating if signed division produces an overflow.
     *
     * SMT_LIB: `bvsdivo`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_SDIV_OVERFLOW(38),

    /**
     * Bit-vector signed division.
     *
     * SMT_LIB: `bvsdiv`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_SDIV(39),

    /**
     * Bit-vector signed greater than or equal.
     *
     * SMT_LIB: `bvsle`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_SGE(40),

    /**
     * Bit-vector signed greater than.
     *
     * SMT_LIB: `bvslt`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_SGT(41),

    /**
     * Bit-vector logical left shift.
     *
     * SMT_LIB: `bvshl`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_SHL(42),

    /**
     * Bit-vector logical right shift.
     *
     * SMT_LIB: `bvshr`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_SHR(43),

    /**
     * Bit-vector signed less than or equal.
     *
     * SMT_LIB: `bvsle`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_SLE(44),

    /**
     * Bit-vector signed less than.
     *
     * SMT_LIB: `bvslt`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_SLT(45),

    /**
     * Bit-vector signed modulo.
     *
     * SMT_LIB: `bvsmod`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_SMOD(46),

    /**
     * Bit-vector signed multiplication overflow test.
     *
     * Predicate indicating if signed multiplication produces an overflow.
     *
     * SMT_LIB: `bvsmulo`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_SMUL_OVERFLOW(47),

    /**
     * Bit-vector signed remainder.
     *
     * SMT_LIB: `bvsrem`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_SREM(48),

    /**
     * Bit-vector signed subtraction overflow test.
     *
     * Predicate indicating if signed subtraction produces an overflow.
     *
     * SMT_LIB: `bvssubo`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_SSUB_OVERFLOW(49),

    /**
     * Bit-vector subtraction.
     *
     * SMT_LIB: `bvsub`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_SUB(50),

    /**
     * Bit-vector unsigned addition overflow test.
     *
     * Predicate indicating if unsigned addition produces an overflow.
     *
     * SMT_LIB: `bvuaddo`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_UADD_OVERFLOW(51),

    /**
     * Bit-vector unsigned division.
     *
     * SMT_LIB: `bvudiv`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_UDIV(52),

    /**
     * Bit-vector unsigned greater than or equal.
     *
     * SMT_LIB: `bvuge`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_UGE(53),

    /**
     * Bit-vector unsigned greater than.
     *
     * SMT_LIB: `bvugt`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_UGT(54),

    /**
     * Bit-vector unsigned less than or equal.
     *
     * SMT_LIB: `bvule`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_ULE(55),

    /**
     * Bit-vector unsigned less than.
     *
     * SMT_LIB: `bvult`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_ULT(56),

    /**
     * Bit-vector unsigned multiplication overflow test.
     *
     * Predicate indicating if unsigned multiplication produces an overflow.
     *
     * SMT_LIB: `bvumulo`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_UMUL_OVERFLOW(57),

    /**
     * Bit-vector unsigned remainder.
     *
     * SMT_LIB: `bvurem`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_UREM(58),

    /**
     * Bit-vector unsigned subtraction overflow test.
     *
     * Predicate indicating if unsigned subtraction produces an overflow.
     *
     * SMT_LIB: `bvusubo`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_USUB_OVERFLOW(59),

    /**
     * Bit-vector xnor.
     *
     * SMT_LIB: `bvxnor`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_BV_XNOR(60),

    /**
     * Bit-vector xor.
     *
     * SMT_LIB: `bvxor`
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_BV_XOR(61),

    /**
     * Bit-vector extract.
     *
     * SMT_LIB: `extract` (indexed)
     *
     * Number of Arguments: 1
     *
     * Number of Indices: 2 (`u`, `l` with `u >= l`)
     */
    BITWUZLA_KIND_BV_EXTRACT(62),

    /**
     * Bit-vector repeat.
     *
     * SMT_LIB: `repeat` (indexed)
     *
     * Number of Arguments: 1
     *
     * Number of Indices: 1 (i s.t. `i * n` fits into [Long])
     */
    BITWUZLA_KIND_BV_REPEAT(63),

    /**
     * Bit-vector rotate left by integer.
     *
     * SMT_LIB: `rotate_left` (indexed)
     *
     * Number of Arguments: 1
     *
     * Number of Indices: 1
     */
    BITWUZLA_KIND_BV_ROLI(64),

    /**
     * Bit-vector rotate right by integer.
     *
     * SMT_LIB: `rotate_right` (indexed)
     *
     * Number of Arguments: 1
     *
     * Number of Indices: 1
     */
    BITWUZLA_KIND_BV_RORI(65),

    /**
     * Bit-vector sign extend.
     *
     * SMT_LIB: `sign_extend` (indexed)
     *
     * Number of Arguments: 1
     *
     * Number of Indices: 1 (`i` s.t. `i + n` fits into [Long])
     */
    BITWUZLA_KIND_BV_SIGN_EXTEND(66),

    /**
     * Bit-vector zero extend.
     *
     * SMT_LIB: `zero_extend` (indexed)
     *
     * Number of Arguments: 1
     *
     * Number of Indices: 1 (`i` s.t. `i + n` fits into [Long])
     */
    BITWUZLA_KIND_BV_ZERO_EXTEND(67),

    //// Floating-point arithmetic
    /**
     * Floating-point absolute value.
     *
     * SMT_LIB: `fp.abs`
     *
     * Number of Arguments: 1
     */
    BITWUZLA_KIND_FP_ABS(68),

    /**
     * Floating-point addition.
     *
     * SMT_LIB: `fp.add`
     *
     * Number of Arguments: 3
     */
    BITWUZLA_KIND_FP_ADD(69),

    /**
     * Floating-point division.
     *
     * SMT_LIB: `fp.div`
     *
     * Number of Arguments: 3
     */
    BITWUZLA_KIND_FP_DIV(70),

    /**
     * Floating-point equality.
     *
     * SMT_LIB: `fp.eq`
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_FP_EQUAL(71),

    /**
     * Floating-point fused multiplcation and addition.
     *
     * SMT_LIB: `fp.fma`
     *
     * Number of Arguments: 4
     */
    BITWUZLA_KIND_FP_FMA(72),

    /**
     * Floating-point IEEE 754 value.
     *
     * SMT_LIB: `fp`
     *
     * Number of Arguments: 3
     */
    BITWUZLA_KIND_FP_FP(73),

    /**
     * Floating-point greater than or equal.
     *
     * SMT_LIB: `fp.geq`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_FP_GEQ(74),

    /**
     * Floating-point greater than.
     *
     * SMT_LIB: `fp.gt`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_FP_GT(75),

    /**
     * Floating-point is infinity tester.
     *
     * SMT_LIB: `fp.isInfinite`
     *
     * Number of Arguments: 1
     */
    BITWUZLA_KIND_FP_IS_INF(76),

    /**
     * Floating-point is Nan tester.
     *
     * SMT_LIB: `fp.isNaN`
     *
     * Number of Arguments: 1
     */
    BITWUZLA_KIND_FP_IS_NAN(77),

    /**
     * Floating-point is negative tester.
     *
     * SMT_LIB: `fp.isNegative`
     *
     * Number of Arguments: 1
     */
    BITWUZLA_KIND_FP_IS_NEG(78),

    /**
     * Floating-point is normal tester.
     *
     * SMT_LIB: `fp.isNormal`
     *
     * Number of Arguments: 1
     */
    BITWUZLA_KIND_FP_IS_NORMAL(79),

    /**
     * Floating-point is positive tester.
     *
     * SMT_LIB: `fp.isPositive`
     *
     * Number of Arguments: 1
     */
    BITWUZLA_KIND_FP_IS_POS(80),

    /**
     * Floating-point is subnormal tester.
     *
     * SMT_LIB: `fp.isSubnormal`
     *
     * Number of Arguments: 1
     */
    BITWUZLA_KIND_FP_IS_SUBNORMAL(81),

    /**
     * Floating-point is zero tester.
     *
     * SMT_LIB: `fp.isZero`
     *
     * Number of Arguments: 1
     */
    BITWUZLA_KIND_FP_IS_ZERO(82),

    /**
     * Floating-point less than or equal.
     *
     * SMT_LIB: `fp.leq`
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_FP_LEQ(83),

    /**
     * Floating-point less than.
     *
     * SMT_LIB: `fp.lt`
     *
     * Number of Arguments: >= 2
     */
    BITWUZLA_KIND_FP_LT(84),

    /**
     * Floating-point max.
     *
     * SMT_LIB: `fp.max`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_FP_MAX(85),

    /**
     * Floating-point min.
     *
     * SMT_LIB: `fp.min`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_FP_MIN(86),

    /**
     * Floating-point multiplcation.
     *
     * SMT_LIB: `fp.mul`
     *
     * Number of Arguments: 3
     */
    BITWUZLA_KIND_FP_MUL(87),

    /**
     * Floating-point negation.
     *
     * SMT_LIB: `fp.neg`
     *
     * Number of Arguments: 1
     */
    BITWUZLA_KIND_FP_NEG(88),

    /**
     * Floating-point remainder.
     *
     * SMT_LIB: `fp.rem`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_FP_REM(89),

    /**
     * Floating-point round to integral.
     *
     * SMT_LIB: `fp.roundToIntegral`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_FP_RTI(90),

    /**
     * Floating-point square root.
     *
     * SMT_LIB: `fp.sqrt`
     *
     * Number of Arguments: 2
     */
    BITWUZLA_KIND_FP_SQRT(91),

    /**
     * Floating-point subtraction.
     *
     * SMT_LIB: `fp.sub`
     *
     * Number of Arguments: 3
     */
    BITWUZLA_KIND_FP_SUB(92),

    /**
     * Floating-point to_fp from IEEE 754 bit-vector.
     *
     * SMT_LIB: `to_fp` (indexed)
     *
     * Number of Arguments: 1
     *
     * Number of Indices: 2 (`e`, `s`)
     */
    BITWUZLA_KIND_FP_TO_FP_FROM_BV(93),

    /**
     * Floating-point to_fp from floating-point.
     *
     * SMT_LIB: `to_fp` (indexed)
     *
     * Number of Arguments: 2
     *
     * Number of Indices: 2 (`e`, `s`)
     */
    BITWUZLA_KIND_FP_TO_FP_FROM_FP(94),

    /**
     * Floating-point to_fp from signed bit-vector value.
     *
     * SMT_LIB: `to_fp` (indexed)
     *
     * Number of Arguments: 2
     *
     * Number of Indices: 2 (`e`, `s`)
     */
    BITWUZLA_KIND_FP_TO_FP_FROM_SBV(95),

    /**
     * Floating-point to_fp from unsigned bit-vector value.
     *
     * SMT_LIB: `to_fp_unsigned` (indexed)
     *
     * Number of Arguments: 2
     *
     * Number of Indices: 2 (`e`, `s`)
     */
    BITWUZLA_KIND_FP_TO_FP_FROM_UBV(96),

    /**
     * Floating-point to_sbv.
     *
     * SMT_LIB: `fp.to_sbv` (indexed)
     *
     * Number of Arguments: 2
     *
     * Number of Indices: 1 (`n`)
     */
    BITWUZLA_KIND_FP_TO_SBV(97),

    /**
     * Floating-point to_ubv.
     *
     * SMT_LIB: `fp.to_ubv` (indexed)
     *
     * Number of Arguments: 2
     *
     * Number of Indices: 1 (`n`)
     */
    BITWUZLA_KIND_FP_TO_UBV(98),
    BITWUZLA_KIND_NUM_KINDS(99);

    companion object {
        private val valueMapping = BitwuzlaKind.values().associateBy { it.value }
        fun fromValue(value: Int): BitwuzlaKind = valueMapping.getValue(value)
    }
}
