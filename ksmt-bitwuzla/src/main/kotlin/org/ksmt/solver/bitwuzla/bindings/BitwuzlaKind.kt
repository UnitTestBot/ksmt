package org.ksmt.solver.bitwuzla.bindings

/**
 * The term kind.<br></br>
 * enum values<br></br>
 * *native declaration : bitwuzla.h:1595*
 */
enum class BitwuzlaKind(val value: Int) {
    /** First order constant.  */
    BITWUZLA_KIND_CONST(0),

    /** Constant array.  */
    BITWUZLA_KIND_CONST_ARRAY(1),

    /** Value.  */
    BITWUZLA_KIND_VAL(2),

    /** Bound variable.  */
    BITWUZLA_KIND_VAR(3),

    /**
     * Boolean and.<br></br>
     * *  SMT-LIB: \c and
     */
    BITWUZLA_KIND_AND(4),

    /** Function application.  */
    BITWUZLA_KIND_APPLY(5),

    /**
     * Array select.<br></br>
     * *  SMT-LIB: \c select
     */
    BITWUZLA_KIND_ARRAY_SELECT(6),

    /**
     * Array store.<br></br>
     * * SMT-LIB: \c store
     */
    BITWUZLA_KIND_ARRAY_STORE(7),

    /**
     * Bit-vector addition.<br></br>
     * *  SMT-LIB: \c bvadd
     */
    BITWUZLA_KIND_BV_ADD(8),

    /**
     * Bit-vector and.<br></br>
     * * SMT-LIB: \c bvand
     */
    BITWUZLA_KIND_BV_AND(9),

    /**
     * Bit-vector arithmetic right shift.<br></br>
     * * SMT-LIB: \c bvashr
     */
    BITWUZLA_KIND_BV_ASHR(10),

    /**
     * Bit-vector comparison.<br></br>
     * * SMT-LIB: \c bvcomp
     */
    BITWUZLA_KIND_BV_COMP(11),

    /**
     * Bit-vector concat.<br></br>
     * * SMT-LIB: \c concat
     */
    BITWUZLA_KIND_BV_CONCAT(12),

    /**
     * Bit-vector decrement.<br></br>
     * * Decrement by one.
     */
    BITWUZLA_KIND_BV_DEC(13),

    /**
     * Bit-vector increment.<br></br>
     * * Increment by one.
     */
    BITWUZLA_KIND_BV_INC(14),

    /**
     * Bit-vector multiplication.<br></br>
     * * SMT-LIB: \c bvmul
     */
    BITWUZLA_KIND_BV_MUL(15),

    /**
     * Bit-vector nand.<br></br>
     * * SMT-LIB: \c bvnand
     */
    BITWUZLA_KIND_BV_NAND(16),

    /**
     * Bit-vector negation (two's complement).<br></br>
     * * SMT-LIB: \c bvneg
     */
    BITWUZLA_KIND_BV_NEG(17),

    /**
     * Bit-vector nor.<br></br>
     * * SMT-LIB: \c bvnor
     */
    BITWUZLA_KIND_BV_NOR(18),

    /**
     * Bit-vector not (one's complement).<br></br>
     * * SMT-LIB: \c bvnot
     */
    BITWUZLA_KIND_BV_NOT(19),

    /**
     * Bit-vector or.<br></br>
     * * SMT-LIB: \c bvor
     */
    BITWUZLA_KIND_BV_OR(20),

    /**
     * Bit-vector and reduction.<br></br>
     * * Bit-wise *and* reduction, all bits are *and*'ed together into a single bit.<br></br>
     * This corresponds to bit-wise *and* reduction as known from Verilog.
     */
    BITWUZLA_KIND_BV_REDAND(21),

    /**
     * Bit-vector reduce or.<br></br>
     * * Bit-wise *or* reduction, all bits are *or*'ed together into a single bit.<br></br>
     * This corresponds to bit-wise *or* reduction as known from Verilog.
     */
    BITWUZLA_KIND_BV_REDOR(22),

    /**
     * Bit-vector reduce xor.<br></br>
     * * Bit-wise *xor* reduction, all bits are *xor*'ed together into a single bit.<br></br>
     * This corresponds to bit-wise *xor* reduction as known from Verilog.
     */
    BITWUZLA_KIND_BV_REDXOR(23),

    /**
     * Bit-vector rotate left (not indexed).<br></br>
     * * This is a non-indexed variant of SMT-LIB \c rotate_left.
     */
    BITWUZLA_KIND_BV_ROL(24),

    /**
     * Bit-vector rotate right.<br></br>
     * * This is a non-indexed variant of SMT-LIB \c rotate_right.
     */
    BITWUZLA_KIND_BV_ROR(25),

    /**
     * Bit-vector signed addition overflow test.<br></br>
     * * Single bit to indicate if signed addition produces an overflow.
     */
    BITWUZLA_KIND_BV_SADD_OVERFLOW(26),

    /**
     * Bit-vector signed division overflow test.<br></br>
     * * Single bit to indicate if signed division produces an overflow.
     */
    BITWUZLA_KIND_BV_SDIV_OVERFLOW(27),

    /**
     * Bit-vector signed division.<br></br>
     * * SMT-LIB: \c bvsdiv
     */
    BITWUZLA_KIND_BV_SDIV(28),

    /**
     * Bit-vector signed greater than or equal.<br></br>
     * * SMT-LIB: \c bvsle
     */
    BITWUZLA_KIND_BV_SGE(29),

    /**
     * Bit-vector signed greater than.<br></br>
     * * SMT-LIB: \c bvslt
     */
    BITWUZLA_KIND_BV_SGT(30),

    /**
     * Bit-vector logical left shift.<br></br>
     * * SMT-LIB: \c bvshl
     */
    BITWUZLA_KIND_BV_SHL(31),

    /**
     * Bit-vector logical right shift.<br></br>
     * * SMT-LIB: \c bvshr
     */
    BITWUZLA_KIND_BV_SHR(32),

    /**
     * Bit-vector signed less than or equal.<br></br>
     * * SMT-LIB: \c bvsle
     */
    BITWUZLA_KIND_BV_SLE(33),

    /**
     * Bit-vector signed less than.<br></br>
     * * SMT-LIB: \c bvslt
     */
    BITWUZLA_KIND_BV_SLT(34),

    /**
     * Bit-vector signed modulo.<br></br>
     * * SMT-LIB: \c bvsmod
     */
    BITWUZLA_KIND_BV_SMOD(35),

    /**
     * Bit-vector signed multiplication overflow test.<br></br>
     * * SMT-LIB: \c bvsmod
     */
    BITWUZLA_KIND_BV_SMUL_OVERFLOW(36),

    /**
     * Bit-vector signed remainder.<br></br>
     * * SMT-LIB: \c bvsrem
     */
    BITWUZLA_KIND_BV_SREM(37),

    /**
     * Bit-vector signed subtraction overflow test.<br></br>
     * * Single bit to indicate if signed subtraction produces an overflow.
     */
    BITWUZLA_KIND_BV_SSUB_OVERFLOW(38),

    /**
     * Bit-vector subtraction.<br></br>
     * * SMT-LIB: \c bvsub
     */
    BITWUZLA_KIND_BV_SUB(39),

    /**
     * Bit-vector unsigned addition overflow test.<br></br>
     * * Single bit to indicate if unsigned addition produces an overflow.
     */
    BITWUZLA_KIND_BV_UADD_OVERFLOW(40),

    /**
     * Bit-vector unsigned division.<br></br>
     * * SMT-LIB: \c bvudiv
     */
    BITWUZLA_KIND_BV_UDIV(41),

    /**
     * Bit-vector unsigned greater than or equal.<br></br>
     * * SMT-LIB: \c bvuge
     */
    BITWUZLA_KIND_BV_UGE(42),

    /**
     * Bit-vector unsigned greater than.<br></br>
     * * SMT-LIB: \c bvugt
     */
    BITWUZLA_KIND_BV_UGT(43),

    /**
     * Bit-vector unsigned less than or equal.<br></br>
     * * SMT-LIB: \c bvule
     */
    BITWUZLA_KIND_BV_ULE(44),

    /**
     * Bit-vector unsigned less than.<br></br>
     * * SMT-LIB: \c bvult
     */
    BITWUZLA_KIND_BV_ULT(45),

    /**
     * Bit-vector unsigned multiplication overflow test.<br></br>
     * * Single bit to indicate if unsigned multiplication produces an overflow.
     */
    BITWUZLA_KIND_BV_UMUL_OVERFLOW(46),

    /**
     * Bit-vector unsigned remainder.<br></br>
     * * SMT-LIB: \c bvurem
     */
    BITWUZLA_KIND_BV_UREM(47),

    /**
     * Bit-vector unsigned subtraction overflow test.<br></br>
     * * Single bit to indicate if unsigned subtraction produces an overflow.
     */
    BITWUZLA_KIND_BV_USUB_OVERFLOW(48),

    /**
     * Bit-vector xnor.<br></br>
     * * SMT-LIB: \c bvxnor
     */
    BITWUZLA_KIND_BV_XNOR(49),

    /**
     * Bit-vector xor.<br></br>
     * * SMT-LIB: \c bvxor
     */
    BITWUZLA_KIND_BV_XOR(50),

    /**
     * Disequality.<br></br>
     * * SMT-LIB: \c distinct
     */
    BITWUZLA_KIND_DISTINCT(51),

    /**
     * Equality.<br></br>
     * * SMT-LIB: \c =
     */
    BITWUZLA_KIND_EQUAL(52),

    /**
     * Existential quantification.<br></br>
     * * SMT-LIB: \c exists
     */
    BITWUZLA_KIND_EXISTS(53),

    /**
     * Universal quantification.<br></br>
     * * SMT-LIB: \c forall
     */
    BITWUZLA_KIND_FORALL(54),

    /**
     * Floating-point absolute value.<br></br>
     * * SMT-LIB: \c fp.abs
     */
    BITWUZLA_KIND_FP_ABS(55),

    /**
     * Floating-point addition.<br></br>
     * * SMT-LIB: \c fp.add
     */
    BITWUZLA_KIND_FP_ADD(56),

    /**
     * Floating-point division.<br></br>
     * * SMT-LIB: \c fp.div
     */
    BITWUZLA_KIND_FP_DIV(57),

    /**
     * Floating-point equality.<br></br>
     * * SMT-LIB: \c fp.eq
     */
    BITWUZLA_KIND_FP_EQ(58),

    /**
     * Floating-point fused multiplcation and addition.<br></br>
     * * SMT-LIB: \c fp.fma
     */
    BITWUZLA_KIND_FP_FMA(59),

    /**
     * Floating-point IEEE 754 value.<br></br>
     * * SMT-LIB: \c fp
     */
    BITWUZLA_KIND_FP_FP(60),

    /**
     * Floating-point greater than or equal.<br></br>
     * * SMT-LIB: \c fp.geq
     */
    BITWUZLA_KIND_FP_GEQ(61),

    /**
     * Floating-point greater than.<br></br>
     * * SMT-LIB: \c fp.gt
     */
    BITWUZLA_KIND_FP_GT(62),

    /**
     * Floating-point is infinity tester.<br></br>
     * * SMT-LIB: \c fp.isInfinite
     */
    BITWUZLA_KIND_FP_IS_INF(63),

    /**
     * Floating-point is Nan tester.<br></br>
     * * SMT-LIB: \c fp.isNaN
     */
    BITWUZLA_KIND_FP_IS_NAN(64),

    /**
     * Floating-point is negative tester.<br></br>
     * * SMT-LIB: \c fp.isNegative
     */
    BITWUZLA_KIND_FP_IS_NEG(65),

    /**
     * Floating-point is normal tester.<br></br>
     * * SMT-LIB: \c fp.isNormal
     */
    BITWUZLA_KIND_FP_IS_NORMAL(66),

    /**
     * Floating-point is positive tester.<br></br>
     * * SMT-LIB: \c fp.isPositive
     */
    BITWUZLA_KIND_FP_IS_POS(67),

    /**
     * Floating-point is subnormal tester.<br></br>
     * * SMT-LIB: \c fp.isSubnormal
     */
    BITWUZLA_KIND_FP_IS_SUBNORMAL(68),

    /**
     * Floating-point is zero tester.<br></br>
     * * SMT-LIB: \c fp.isZero
     */
    BITWUZLA_KIND_FP_IS_ZERO(69),

    /**
     * Floating-point less than or equal.<br></br>
     * * SMT-LIB: \c fp.leq
     */
    BITWUZLA_KIND_FP_LEQ(70),

    /**
     * Floating-point less than.<br></br>
     * * SMT-LIB: \c fp.lt
     */
    BITWUZLA_KIND_FP_LT(71),

    /**
     * Floating-point max.<br></br>
     * * SMT-LIB: \c fp.max
     */
    BITWUZLA_KIND_FP_MAX(72),

    /**
     * Floating-point min.<br></br>
     * * SMT-LIB: \c fp.min
     */
    BITWUZLA_KIND_FP_MIN(73),

    /**
     * Floating-point multiplcation.<br></br>
     * * SMT-LIB: \c fp.mul
     */
    BITWUZLA_KIND_FP_MUL(74),

    /**
     * Floating-point negation.<br></br>
     * * SMT-LIB: \c fp.neg
     */
    BITWUZLA_KIND_FP_NEG(75),

    /**
     * Floating-point remainder.<br></br>
     * * SMT-LIB: \c fp.rem
     */
    BITWUZLA_KIND_FP_REM(76),

    /**
     * Floating-point round to integral.<br></br>
     * * SMT-LIB: \c fp.roundToIntegral
     */
    BITWUZLA_KIND_FP_RTI(77),

    /**
     * Floating-point round to square root.<br></br>
     * * SMT-LIB: \c fp.sqrt
     */
    BITWUZLA_KIND_FP_SQRT(78),

    /**
     * Floating-point round to subtraction.<br></br>
     * * SMT-LIB: \c fp.sqrt
     */
    BITWUZLA_KIND_FP_SUB(79),

    /**
     * Boolean if and only if.<br></br>
     * * SMT-LIB: \c =
     */
    BITWUZLA_KIND_IFF(80),

    /**
     * Boolean implies.<br></br>
     * * SMT-LIB: \c =>
     */
    BITWUZLA_KIND_IMPLIES(81),

    /**
     * If-then-else.<br></br>
     * * SMT-LIB: \c ite
     */
    BITWUZLA_KIND_ITE(82),

    /** Lambda.  */
    BITWUZLA_KIND_LAMBDA(83),

    /**
     * Boolean not.<br></br>
     * * SMT-LIB: \c not
     */
    BITWUZLA_KIND_NOT(84),

    /**
     * Boolean or.<br></br>
     * * SMT-LIB: \c or
     */
    BITWUZLA_KIND_OR(85),

    /**
     * Boolean xor.<br></br>
     * * SMT-LIB: \c xor
     */
    BITWUZLA_KIND_XOR(86),

    /**
     * Bit-vector extract.<br></br>
     * * SMT-LIB: \c extract (indexed)
     */
    BITWUZLA_KIND_BV_EXTRACT(87),

    /**
     * Bit-vector repeat.<br></br>
     * * SMT-LIB: \c repeat (indexed)
     */
    BITWUZLA_KIND_BV_REPEAT(88),

    /**
     * Bit-vector rotate left by integer.<br></br>
     * * SMT-LIB: \c rotate_left (indexed)
     */
    BITWUZLA_KIND_BV_ROLI(89),

    /**
     * Bit-vector rotate right by integer.<br></br>
     * * SMT-LIB: \c rotate_right (indexed)
     */
    BITWUZLA_KIND_BV_RORI(90),

    /**
     * Bit-vector sign extend.<br></br>
     * * SMT-LIB: \c sign_extend (indexed)
     */
    BITWUZLA_KIND_BV_SIGN_EXTEND(91),

    /**
     * Bit-vector zero extend.<br></br>
     * * SMT-LIB: \c zero_extend (indexed)
     */
    BITWUZLA_KIND_BV_ZERO_EXTEND(92),

    /**
     * Floating-point to_fp from IEEE 754 bit-vector.<br></br>
     * * SMT-LIB: \c to_fp (indexed)
     */
    BITWUZLA_KIND_FP_TO_FP_FROM_BV(93),

    /**
     * Floating-point to_fp from floating-point.<br></br>
     * * SMT-LIB: \c to_fp (indexed)
     */
    BITWUZLA_KIND_FP_TO_FP_FROM_FP(94),

    /**
     * Floating-point to_fp from signed bit-vector value.<br></br>
     * * SMT-LIB: \c to_fp (indexed)
     */
    BITWUZLA_KIND_FP_TO_FP_FROM_SBV(95),

    /**
     * Floating-point to_fp from unsigned bit-vector value.<br></br>
     * * SMT-LIB: \c to_fp_unsigned (indexed)
     */
    BITWUZLA_KIND_FP_TO_FP_FROM_UBV(96),

    /**
     * Floating-point to_sbv.<br></br>
     * * SMT-LIB: \c fp.to_sbv (indexed)
     */
    BITWUZLA_KIND_FP_TO_SBV(97),

    /**
     * Floating-point to_ubv.<br></br>
     * * SMT-LIB: \c fp.to_ubv (indexed)
     */
    BITWUZLA_KIND_FP_TO_UBV(98),
    BITWUZLA_NUM_KINDS(99);

    companion object {
        private val valueMapping = BitwuzlaKind.values().associateBy { it.value }
        fun fromValue(value: Int): BitwuzlaKind = valueMapping.getValue(value)
    }
}
