@file:Suppress("MagicNumber")

package io.ksmt.solver.bitwuzla.bindings

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
     */
    BITWUZLA_KIND_DISTINCT(5),

    /**
     * Equality.
     *
     * SMT-LIB: `=`
     */
    BITWUZLA_KIND_EQUAL(6),

    /**
     * Boolean if and only if.
     *
     * SMT-LIB: `=`
     */
    BITWUZLA_KIND_IFF(7),

    /**
     * Boolean implies.
     *
     * SMT-LIB: `=>`
     */
    BITWUZLA_KIND_IMPLIES(8),

    /**
     * Boolean not.
     *
     * SMT-LIB: `not`
     */
    BITWUZLA_KIND_NOT(9),

    /**
     * Boolean or.
     *
     * SMT-LIB: `or`
     */
    BITWUZLA_KIND_OR(10),

    /**
     * Boolean xor.
     *
     * SMT-LIB: `xor`
     */
    BITWUZLA_KIND_XOR(11),

    /**
     * If-then-else.
     *
     * SMT-LIB: `ite`
     */
    BITWUZLA_KIND_ITE(12),

    /**
     * Existential quantification.
     *
     * SMT-LIB: `exists`
     */
    BITWUZLA_KIND_EXISTS(13),

    /**
     * Universal quantification.
     *
     * SMT-LIB: `forall`
     */
    BITWUZLA_KIND_FORALL(14),

    /** Function application. */
    BITWUZLA_KIND_APPLY(15),

    /** Lambda. */
    BITWUZLA_KIND_LAMBDA(16),

    /**
     * Array select.
     *
     * SMT-LIB: `select`
     */
    BITWUZLA_KIND_SELECT(17),

    /**
     * Array store.
     *
     * SMT-LIB: `store`
     */
    BITWUZLA_KIND_STORE(18),

    /**
     * Bit-vector addition.
     *
     * SMT-LIB: `bvadd`
     */
    BITWUZLA_KIND_BV_ADD(19),

    /**
     * Bit-vector and.
     *
     * SMT-LIB: `bvand`
     */
    BITWUZLA_KIND_BV_AND(20),

    /**
     * Bit-vector arithmetic right shift.
     *
     * SMT-LIB: `bvashr`
     */
    BITWUZLA_KIND_BV_ASHR(21),

    /**
     * Bit-vector comparison.
     *
     * SMT-LIB: `bvcomp`
     */
    BITWUZLA_KIND_BV_COMP(22),

    /**
     * Bit-vector concat.
     *
     * SMT-LIB: `concat`
     */
    BITWUZLA_KIND_BV_CONCAT(23),

    /**
     * Bit-vector decrement.
     *
     * Decrement by one.
     */
    BITWUZLA_KIND_BV_DEC(24),

    /**
     * Bit-vector increment.
     *
     * Increment by one.
     */
    BITWUZLA_KIND_BV_INC(25),

    /**
     * Bit-vector multiplication.
     *
     * SMT-LIB: `bvmul`
     */
    BITWUZLA_KIND_BV_MUL(26),

    /**
     * Bit-vector nand.
     *
     * SMT-LIB: `bvnand`
     */
    BITWUZLA_KIND_BV_NAND(27),

    /**
     * Bit-vector negation (two's complement).
     *
     * SMT-LIB: `bvneg`
     */
    BITWUZLA_KIND_BV_NEG(28),

    /**
     * Bit-vector negation (two's complement) overflow.
     *
     * SMT-LIB: `bvnego`
     */
    BITWUZLA_KIND_BV_NEG_OVERFLOW(29),

    /**
     * Bit-vector nor.
     *
     * SMT-LIB: `bvnor`
     */
    BITWUZLA_KIND_BV_NOR(30),

    /**
     * Bit-vector not (one's complement).
     *
     * SMT-LIB: `bvnot`
     */
    BITWUZLA_KIND_BV_NOT(31),

    /**
     * Bit-vector or.
     *
     * SMT-LIB: `bvor`
     */
    BITWUZLA_KIND_BV_OR(32),

    /**
     * Bit-vector and reduction.
     *
     * Bit-wise *and* reduction, all bits are *and*'ed together into a single bit.
     *
     * This corresponds to bit-wise *and* reduction as known from Verilog.
     */
    BITWUZLA_KIND_BV_REDAND(33),

    /**
     * Bit-vector reduce or.
     *
     * Bit-wise *or* reduction, all bits are *or*'ed together into a single bit.
     *
     * This corresponds to bit-wise *or* reduction as known from Verilog.
     */
    BITWUZLA_KIND_BV_REDOR(34),

    /**
     * Bit-vector reduce xor.
     *
     * Bit-wise *xor* reduction, all bits are *xor*'ed together into a single bit.
     *
     * This corresponds to bit-wise *xor* reduction as known from Verilog.
     */
    BITWUZLA_KIND_BV_REDXOR(35),

    /**
     * Bit-vector rotate left (not indexed).
     *
     * This is a non-indexed variant of SMT-LIB \c rotate_left.
     */
    BITWUZLA_KIND_BV_ROL(36),

    /**
     * Bit-vector rotate right.
     *
     * This is a non-indexed variant of SMT-LIB \c rotate_right.
     */
    BITWUZLA_KIND_BV_ROR(37),

    /**
     * Bit-vector signed addition overflow test.
     *
     * Single bit to indicate if signed addition produces an overflow.
     */
    BITWUZLA_KIND_BV_SADD_OVERFLOW(38),

    /**
     * Bit-vector signed division overflow test.
     *
     * Single bit to indicate if signed division produces an overflow.
     */
    BITWUZLA_KIND_BV_SDIV_OVERFLOW(39),

    /**
     * Bit-vector signed division.
     *
     * SMT-LIB: `bvsdiv`
     */
    BITWUZLA_KIND_BV_SDIV(40),

    /**
     * Bit-vector signed greater than or equal.
     *
     * SMT-LIB: `bvsle`
     */
    BITWUZLA_KIND_BV_SGE(41),

    /**
     * Bit-vector signed greater than.
     *
     * SMT-LIB: `bvslt`
     */
    BITWUZLA_KIND_BV_SGT(42),

    /**
     * Bit-vector logical left shift.
     *
     * SMT-LIB: `bvshl`
     */
    BITWUZLA_KIND_BV_SHL(43),

    /**
     * Bit-vector logical right shift.
     *
     * SMT-LIB: `bvshr`
     */
    BITWUZLA_KIND_BV_SHR(44),

    /**
     * Bit-vector signed less than or equal.
     *
     * SMT-LIB: `bvsle`
     */
    BITWUZLA_KIND_BV_SLE(45),

    /**
     * Bit-vector signed less than.
     *
     * SMT-LIB: `bvslt`
     */
    BITWUZLA_KIND_BV_SLT(46),

    /**
     * Bit-vector signed modulo.
     *
     * SMT-LIB: `bvsmod`
     */
    BITWUZLA_KIND_BV_SMOD(47),

    /**
     * Bit-vector signed multiplication overflow test.
     *
     * SMT-LIB: `bvsmod`
     */
    BITWUZLA_KIND_BV_SMUL_OVERFLOW(48),

    /**
     * Bit-vector signed remainder.
     *
     * SMT-LIB: `bvsrem`
     */
    BITWUZLA_KIND_BV_SREM(49),

    /**
     * Bit-vector signed subtraction overflow test.
     *
     * Single bit to indicate if signed subtraction produces an overflow.
     */
    BITWUZLA_KIND_BV_SSUB_OVERFLOW(50),

    /**
     * Bit-vector subtraction.
     *
     * SMT-LIB: `bvsub`
     */
    BITWUZLA_KIND_BV_SUB(51),

    /**
     * Bit-vector unsigned addition overflow test.
     *
     * Single bit to indicate if unsigned addition produces an overflow.
     */
    BITWUZLA_KIND_BV_UADD_OVERFLOW(52),

    /**
     * Bit-vector unsigned division.
     *
     * SMT-LIB: `bvudiv`
     */
    BITWUZLA_KIND_BV_UDIV(53),

    /**
     * Bit-vector unsigned greater than or equal.
     *
     * SMT-LIB: `bvuge`
     */
    BITWUZLA_KIND_BV_UGE(54),

    /**
     * Bit-vector unsigned greater than.
     *
     * SMT-LIB: `bvugt`
     */
    BITWUZLA_KIND_BV_UGT(55),

    /**
     * Bit-vector unsigned less than or equal.
     *
     * SMT-LIB: `bvule`
     */
    BITWUZLA_KIND_BV_ULE(56),

    /**
     * Bit-vector unsigned less than.
     *
     * SMT-LIB: `bvult`
     */
    BITWUZLA_KIND_BV_ULT(57),

    /**
     * Bit-vector unsigned multiplication overflow test.
     *
     * Single bit to indicate if unsigned multiplication produces an overflow.
     */
    BITWUZLA_KIND_BV_UMUL_OVERFLOW(58),

    /**
     * Bit-vector unsigned remainder.
     *
     * SMT-LIB: `bvurem`
     */
    BITWUZLA_KIND_BV_UREM(59),

    /**
     * Bit-vector unsigned subtraction overflow test.
     *
     * Single bit to indicate if unsigned subtraction produces an overflow.
     */
    BITWUZLA_KIND_BV_USUB_OVERFLOW(60),

    /**
     * Bit-vector xnor.
     *
     * SMT-LIB: `bvxnor`
     */
    BITWUZLA_KIND_BV_XNOR(61),

    /**
     * Bit-vector xor.
     *
     * SMT-LIB: `bvxor`
     */
    BITWUZLA_KIND_BV_XOR(62),

    /**
     * Bit-vector extract.
     *
     * SMT-LIB: `extract` (indexed)
     */
    BITWUZLA_KIND_BV_EXTRACT(63),

    /**
     * Bit-vector repeat.
     *
     * SMT-LIB: `repeat` (indexed)
     */
    BITWUZLA_KIND_BV_REPEAT(64),

    /**
     * Bit-vector rotate left by integer.
     *
     * SMT-LIB: `rotate_left` (indexed)
     */
    BITWUZLA_KIND_BV_ROLI(65),

    /**
     * Bit-vector rotate right by integer.
     *
     * SMT-LIB: `rotate_right` (indexed)
     */
    BITWUZLA_KIND_BV_RORI(66),

    /**
     * Bit-vector sign extend.
     *
     * SMT-LIB: `sign_extend` (indexed)
     */
    BITWUZLA_KIND_BV_SIGN_EXTEND(67),

    /**
     * Bit-vector zero extend.
     *
     * SMT-LIB: `zero_extend` (indexed)
     */
    BITWUZLA_KIND_BV_ZERO_EXTEND(68),

    /**
     * Floating-point absolute value.
     *
     * SMT-LIB: `fp.abs`
     */
    BITWUZLA_KIND_FP_ABS(69),

    /**
     * Floating-point addition.
     *
     * SMT-LIB: `fp.add`
     */
    BITWUZLA_KIND_FP_ADD(70),

    /**
     * Floating-point division.
     *
     * SMT-LIB: `fp.div`
     */
    BITWUZLA_KIND_FP_DIV(71),

    /**
     * Floating-point equality.
     *
     * SMT-LIB: `fp.eq`
     */
    BITWUZLA_KIND_FP_EQUAL(72),

    /**
     * Floating-point fused multiplcation and addition.
     *
     * SMT-LIB: `fp.fma`
     */
    BITWUZLA_KIND_FP_FMA(73),

    /**
     * Floating-point IEEE 754 value.
     *
     * SMT-LIB: `fp`
     */
    BITWUZLA_KIND_FP_FP(74),

    /**
     * Floating-point greater than or equal.
     *
     * SMT-LIB: `fp.geq`
     */
    BITWUZLA_KIND_FP_GEQ(75),

    /**
     * Floating-point greater than.
     *
     * SMT-LIB: `fp.gt`
     */
    BITWUZLA_KIND_FP_GT(76),

    /**
     * Floating-point is infinity tester.
     *
     * SMT-LIB: `fp.isInfinite`
     */
    BITWUZLA_KIND_FP_IS_INF(77),

    /**
     * Floating-point is Nan tester.
     *
     * SMT-LIB: `fp.isNaN`
     */
    BITWUZLA_KIND_FP_IS_NAN(78),

    /**
     * Floating-point is negative tester.
     *
     * SMT-LIB: `fp.isNegative`
     */
    BITWUZLA_KIND_FP_IS_NEG(79),

    /**
     * Floating-point is normal tester.
     *
     * SMT-LIB: `fp.isNormal`
     */
    BITWUZLA_KIND_FP_IS_NORMAL(80),

    /**
     * Floating-point is positive tester.
     *
     * SMT-LIB: `fp.isPositive`
     */
    BITWUZLA_KIND_FP_IS_POS(81),

    /**
     * Floating-point is subnormal tester.
     *
     * SMT-LIB: `fp.isSubnormal`
     */
    BITWUZLA_KIND_FP_IS_SUBNORMAL(82),

    /**
     * Floating-point is zero tester.
     *
     * SMT-LIB: `fp.isZero`
     */
    BITWUZLA_KIND_FP_IS_ZERO(83),

    /**
     * Floating-point less than or equal.
     *
     * SMT-LIB: `fp.leq`
     */
    BITWUZLA_KIND_FP_LEQ(84),

    /**
     * Floating-point less than.
     *
     * SMT-LIB: `fp.lt`
     */
    BITWUZLA_KIND_FP_LT(85),

    /**
     * Floating-point max.
     *
     * SMT-LIB: `fp.max`
     */
    BITWUZLA_KIND_FP_MAX(86),

    /**
     * Floating-point min.
     *
     * SMT-LIB: `fp.min`
     */
    BITWUZLA_KIND_FP_MIN(87),

    /**
     * Floating-point multiplcation.
     *
     * SMT-LIB: `fp.mul`
     */
    BITWUZLA_KIND_FP_MUL(88),

    /**
     * Floating-point negation.
     *
     * SMT-LIB: `fp.neg`
     */
    BITWUZLA_KIND_FP_NEG(89),

    /**
     * Floating-point remainder.
     *
     * SMT-LIB: `fp.rem`
     */
    BITWUZLA_KIND_FP_REM(90),

    /**
     * Floating-point round to integral.
     *
     * SMT-LIB: `fp.roundToIntegral`
     */
    BITWUZLA_KIND_FP_RTI(91),

    /**
     * Floating-point round to square root.
     *
     * SMT-LIB: `fp.sqrt`
     */
    BITWUZLA_KIND_FP_SQRT(92),

    /**
     * Floating-point round to subtraction.
     *
     * SMT-LIB: `fp.sqrt`
     */
    BITWUZLA_KIND_FP_SUB(93),

    /**
     * Floating-point to_fp from IEEE 754 bit-vector.
     *
     * SMT-LIB: `to_fp` (indexed)
     */
    BITWUZLA_KIND_FP_TO_FP_FROM_BV(94),

    /**
     * Floating-point to_fp from floating-point.
     *
     * SMT-LIB: `to_fp` (indexed)
     */
    BITWUZLA_KIND_FP_TO_FP_FROM_FP(95),

    /**
     * Floating-point to_fp from signed bit-vector value.
     *
     * SMT-LIB: `to_fp` (indexed)
     */
    BITWUZLA_KIND_FP_TO_FP_FROM_SBV(96),

    /**
     * Floating-point to_fp from unsigned bit-vector value.
     *
     * SMT-LIB: `to_fp_unsigned` (indexed)
     */
    BITWUZLA_KIND_FP_TO_FP_FROM_UBV(97),

    /**
     * Floating-point to_sbv.
     *
     * SMT-LIB: `fp.to_sbv` (indexed)
     */
    BITWUZLA_KIND_FP_TO_SBV(98),

    /**
     * Floating-point to_ubv.
     *
     * SMT-LIB: `fp.to_ubv` (indexed)
     */
    BITWUZLA_KIND_FP_TO_UBV(99);

    companion object {
        private val valueMapping = BitwuzlaKind.values().associateBy { it.value }
        fun fromValue(value: Int): BitwuzlaKind = valueMapping.getValue(value)
    }
}
