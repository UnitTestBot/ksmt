package io.ksmt.solver.bitwuzla.bindings

/**
 * The term kind.
 */
enum class BitwuzlaKind {
    /** First order constant. */
    BITWUZLA_KIND_CONSTANT,

    /** Constant array. */
    BITWUZLA_KIND_CONST_ARRAY,

    /** Value. */
    BITWUZLA_KIND_VALUE,

    /** Bound variable. */
    BITWUZLA_KIND_VARIABLE,

    /**
     * Boolean and.
     *
     * SMT-LIB: `and`
     */
    BITWUZLA_KIND_AND,

    /**
     * Disequality.
     *
     * SMT-LIB: `distinct`
     */
    BITWUZLA_KIND_DISTINCT,

    /**
     * Equality.
     *
     * SMT-LIB: `=`
     */
    BITWUZLA_KIND_EQUAL,

    /**
     * Boolean if and only if.
     *
     * SMT-LIB: `=`
     */
    BITWUZLA_KIND_IFF,

    /**
     * Boolean implies.
     *
     * SMT-LIB: `=>`
     */
    BITWUZLA_KIND_IMPLIES,

    /**
     * Boolean not.
     *
     * SMT-LIB: `not`
     */
    BITWUZLA_KIND_NOT,

    /**
     * Boolean or.
     *
     * SMT-LIB: `or`
     */
    BITWUZLA_KIND_OR,

    /**
     * Boolean xor.
     *
     * SMT-LIB: `xor`
     */
    BITWUZLA_KIND_XOR,

    /**
     * If-then-else.
     *
     * SMT-LIB: `ite`
     */
    BITWUZLA_KIND_ITE,

    /**
     * Existential quantification.
     *
     * SMT-LIB: `exists`
     */
    BITWUZLA_KIND_EXISTS,

    /**
     * Universal quantification.
     *
     * SMT-LIB: `forall`
     */
    BITWUZLA_KIND_FORALL,

    /** Function application. */
    BITWUZLA_KIND_APPLY,

    /** Lambda. */
    BITWUZLA_KIND_LAMBDA,

    /**
     * Array select.
     *
     * SMT-LIB: `select`
     */
    BITWUZLA_KIND_SELECT,

    /**
     * Array store.
     *
     * SMT-LIB: `store`
     */
    BITWUZLA_KIND_STORE,

    /**
     * Bit-vector addition.
     *
     * SMT-LIB: `bvadd`
     */
    BITWUZLA_KIND_BV_ADD,

    /**
     * Bit-vector and.
     *
     * SMT-LIB: `bvand`
     */
    BITWUZLA_KIND_BV_AND,

    /**
     * Bit-vector arithmetic right shift.
     *
     * SMT-LIB: `bvashr`
     */
    BITWUZLA_KIND_BV_ASHR,

    /**
     * Bit-vector comparison.
     *
     * SMT-LIB: `bvcomp`
     */
    BITWUZLA_KIND_BV_COMP,

    /**
     * Bit-vector concat.
     *
     * SMT-LIB: `concat`
     */
    BITWUZLA_KIND_BV_CONCAT,

    /**
     * Bit-vector decrement.
     *
     * Decrement by one.
     */
    BITWUZLA_KIND_BV_DEC,

    /**
     * Bit-vector increment.
     *
     * Increment by one.
     */
    BITWUZLA_KIND_BV_INC,

    /**
     * Bit-vector multiplication.
     *
     * SMT-LIB: `bvmul`
     */
    BITWUZLA_KIND_BV_MUL,

    /**
     * Bit-vector nand.
     *
     * SMT-LIB: `bvnand`
     */
    BITWUZLA_KIND_BV_NAND,

    /**
     * Bit-vector negation (two's complement).
     *
     * SMT-LIB: `bvneg`
     */
    BITWUZLA_KIND_BV_NEG,

    /**
     * Bit-vector negation (two's complement) overflow.
     *
     * SMT-LIB: `bvnego`
     */
    BITWUZLA_KIND_BV_NEG_OVERFLOW,

    /**
     * Bit-vector nor.
     *
     * SMT-LIB: `bvnor`
     */
    BITWUZLA_KIND_BV_NOR,

    /**
     * Bit-vector not (one's complement).
     *
     * SMT-LIB: `bvnot`
     */
    BITWUZLA_KIND_BV_NOT,

    /**
     * Bit-vector or.
     *
     * SMT-LIB: `bvor`
     */
    BITWUZLA_KIND_BV_OR,

    /**
     * Bit-vector and reduction.
     *
     * Bit-wise *and* reduction, all bits are *and*'ed together into a single bit.
     *
     * This corresponds to bit-wise *and* reduction as known from Verilog.
     */
    BITWUZLA_KIND_BV_REDAND,

    /**
     * Bit-vector reduce or.
     *
     * Bit-wise *or* reduction, all bits are *or*'ed together into a single bit.
     *
     * This corresponds to bit-wise *or* reduction as known from Verilog.
     */
    BITWUZLA_KIND_BV_REDOR,

    /**
     * Bit-vector reduce xor.
     *
     * Bit-wise *xor* reduction, all bits are *xor*'ed together into a single bit.
     *
     * This corresponds to bit-wise *xor* reduction as known from Verilog.
     */
    BITWUZLA_KIND_BV_REDXOR,

    /**
     * Bit-vector rotate left (not indexed).
     *
     * This is a non-indexed variant of SMT-LIB \c rotate_left.
     */
    BITWUZLA_KIND_BV_ROL,

    /**
     * Bit-vector rotate right.
     *
     * This is a non-indexed variant of SMT-LIB \c rotate_right.
     */
    BITWUZLA_KIND_BV_ROR,

    /**
     * Bit-vector signed addition overflow test.
     *
     * Single bit to indicate if signed addition produces an overflow.
     */
    BITWUZLA_KIND_BV_SADD_OVERFLOW,

    /**
     * Bit-vector signed division overflow test.
     *
     * Single bit to indicate if signed division produces an overflow.
     */
    BITWUZLA_KIND_BV_SDIV_OVERFLOW,

    /**
     * Bit-vector signed division.
     *
     * SMT-LIB: `bvsdiv`
     */
    BITWUZLA_KIND_BV_SDIV,

    /**
     * Bit-vector signed greater than or equal.
     *
     * SMT-LIB: `bvsle`
     */
    BITWUZLA_KIND_BV_SGE,

    /**
     * Bit-vector signed greater than.
     *
     * SMT-LIB: `bvslt`
     */
    BITWUZLA_KIND_BV_SGT,

    /**
     * Bit-vector logical left shift.
     *
     * SMT-LIB: `bvshl`
     */
    BITWUZLA_KIND_BV_SHL,

    /**
     * Bit-vector logical right shift.
     *
     * SMT-LIB: `bvshr`
     */
    BITWUZLA_KIND_BV_SHR,

    /**
     * Bit-vector signed less than or equal.
     *
     * SMT-LIB: `bvsle`
     */
    BITWUZLA_KIND_BV_SLE,

    /**
     * Bit-vector signed less than.
     *
     * SMT-LIB: `bvslt`
     */
    BITWUZLA_KIND_BV_SLT,

    /**
     * Bit-vector signed modulo.
     *
     * SMT-LIB: `bvsmod`
     */
    BITWUZLA_KIND_BV_SMOD,

    /**
     * Bit-vector signed multiplication overflow test.
     *
     * SMT-LIB: `bvsmod`
     */
    BITWUZLA_KIND_BV_SMUL_OVERFLOW,

    /**
     * Bit-vector signed remainder.
     *
     * SMT-LIB: `bvsrem`
     */
    BITWUZLA_KIND_BV_SREM,

    /**
     * Bit-vector signed subtraction overflow test.
     *
     * Single bit to indicate if signed subtraction produces an overflow.
     */
    BITWUZLA_KIND_BV_SSUB_OVERFLOW,

    /**
     * Bit-vector subtraction.
     *
     * SMT-LIB: `bvsub`
     */
    BITWUZLA_KIND_BV_SUB,

    /**
     * Bit-vector unsigned addition overflow test.
     *
     * Single bit to indicate if unsigned addition produces an overflow.
     */
    BITWUZLA_KIND_BV_UADD_OVERFLOW,

    /**
     * Bit-vector unsigned division.
     *
     * SMT-LIB: `bvudiv`
     */
    BITWUZLA_KIND_BV_UDIV,

    /**
     * Bit-vector unsigned greater than or equal.
     *
     * SMT-LIB: `bvuge`
     */
    BITWUZLA_KIND_BV_UGE,

    /**
     * Bit-vector unsigned greater than.
     *
     * SMT-LIB: `bvugt`
     */
    BITWUZLA_KIND_BV_UGT,

    /**
     * Bit-vector unsigned less than or equal.
     *
     * SMT-LIB: `bvule`
     */
    BITWUZLA_KIND_BV_ULE,

    /**
     * Bit-vector unsigned less than.
     *
     * SMT-LIB: `bvult`
     */
    BITWUZLA_KIND_BV_ULT,

    /**
     * Bit-vector unsigned multiplication overflow test.
     *
     * Single bit to indicate if unsigned multiplication produces an overflow.
     */
    BITWUZLA_KIND_BV_UMUL_OVERFLOW,

    /**
     * Bit-vector unsigned remainder.
     *
     * SMT-LIB: `bvurem`
     */
    BITWUZLA_KIND_BV_UREM,

    /**
     * Bit-vector unsigned subtraction overflow test.
     *
     * Single bit to indicate if unsigned subtraction produces an overflow.
     */
    BITWUZLA_KIND_BV_USUB_OVERFLOW,

    /**
     * Bit-vector xnor.
     *
     * SMT-LIB: `bvxnor`
     */
    BITWUZLA_KIND_BV_XNOR,

    /**
     * Bit-vector xor.
     *
     * SMT-LIB: `bvxor`
     */
    BITWUZLA_KIND_BV_XOR,

    /**
     * Bit-vector extract.
     *
     * SMT-LIB: `extract` (indexed)
     */
    BITWUZLA_KIND_BV_EXTRACT,

    /**
     * Bit-vector repeat.
     *
     * SMT-LIB: `repeat` (indexed)
     */
    BITWUZLA_KIND_BV_REPEAT,

    /**
     * Bit-vector rotate left by integer.
     *
     * SMT-LIB: `rotate_left` (indexed)
     */
    BITWUZLA_KIND_BV_ROLI,

    /**
     * Bit-vector rotate right by integer.
     *
     * SMT-LIB: `rotate_right` (indexed)
     */
    BITWUZLA_KIND_BV_RORI,

    /**
     * Bit-vector sign extend.
     *
     * SMT-LIB: `sign_extend` (indexed)
     */
    BITWUZLA_KIND_BV_SIGN_EXTEND,

    /**
     * Bit-vector zero extend.
     *
     * SMT-LIB: `zero_extend` (indexed)
     */
    BITWUZLA_KIND_BV_ZERO_EXTEND,

    /**
     * Floating-point absolute value.
     *
     * SMT-LIB: `fp.abs`
     */
    BITWUZLA_KIND_FP_ABS,

    /**
     * Floating-point addition.
     *
     * SMT-LIB: `fp.add`
     */
    BITWUZLA_KIND_FP_ADD,

    /**
     * Floating-point division.
     *
     * SMT-LIB: `fp.div`
     */
    BITWUZLA_KIND_FP_DIV,

    /**
     * Floating-point equality.
     *
     * SMT-LIB: `fp.eq`
     */
    BITWUZLA_KIND_FP_EQUAL,

    /**
     * Floating-point fused multiplcation and addition.
     *
     * SMT-LIB: `fp.fma`
     */
    BITWUZLA_KIND_FP_FMA,

    /**
     * Floating-point IEEE 754 value.
     *
     * SMT-LIB: `fp`
     */
    BITWUZLA_KIND_FP_FP,

    /**
     * Floating-point greater than or equal.
     *
     * SMT-LIB: `fp.geq`
     */
    BITWUZLA_KIND_FP_GEQ,

    /**
     * Floating-point greater than.
     *
     * SMT-LIB: `fp.gt`
     */
    BITWUZLA_KIND_FP_GT,

    /**
     * Floating-point is infinity tester.
     *
     * SMT-LIB: `fp.isInfinite`
     */
    BITWUZLA_KIND_FP_IS_INF,

    /**
     * Floating-point is Nan tester.
     *
     * SMT-LIB: `fp.isNaN`
     */
    BITWUZLA_KIND_FP_IS_NAN,

    /**
     * Floating-point is negative tester.
     *
     * SMT-LIB: `fp.isNegative`
     */
    BITWUZLA_KIND_FP_IS_NEG,

    /**
     * Floating-point is normal tester.
     *
     * SMT-LIB: `fp.isNormal`
     */
    BITWUZLA_KIND_FP_IS_NORMAL,

    /**
     * Floating-point is positive tester.
     *
     * SMT-LIB: `fp.isPositive`
     */
    BITWUZLA_KIND_FP_IS_POS,

    /**
     * Floating-point is subnormal tester.
     *
     * SMT-LIB: `fp.isSubnormal`
     */
    BITWUZLA_KIND_FP_IS_SUBNORMAL,

    /**
     * Floating-point is zero tester.
     *
     * SMT-LIB: `fp.isZero`
     */
    BITWUZLA_KIND_FP_IS_ZERO,

    /**
     * Floating-point less than or equal.
     *
     * SMT-LIB: `fp.leq`
     */
    BITWUZLA_KIND_FP_LEQ,

    /**
     * Floating-point less than.
     *
     * SMT-LIB: `fp.lt`
     */
    BITWUZLA_KIND_FP_LT,

    /**
     * Floating-point max.
     *
     * SMT-LIB: `fp.max`
     */
    BITWUZLA_KIND_FP_MAX,

    /**
     * Floating-point min.
     *
     * SMT-LIB: `fp.min`
     */
    BITWUZLA_KIND_FP_MIN,

    /**
     * Floating-point multiplcation.
     *
     * SMT-LIB: `fp.mul`
     */
    BITWUZLA_KIND_FP_MUL,

    /**
     * Floating-point negation.
     *
     * SMT-LIB: `fp.neg`
     */
    BITWUZLA_KIND_FP_NEG,

    /**
     * Floating-point remainder.
     *
     * SMT-LIB: `fp.rem`
     */
    BITWUZLA_KIND_FP_REM,

    /**
     * Floating-point round to integral.
     *
     * SMT-LIB: `fp.roundToIntegral`
     */
    BITWUZLA_KIND_FP_RTI,

    /**
     * Floating-point round to square root.
     *
     * SMT-LIB: `fp.sqrt`
     */
    BITWUZLA_KIND_FP_SQRT,

    /**
     * Floating-point round to subtraction.
     *
     * SMT-LIB: `fp.sqrt`
     */
    BITWUZLA_KIND_FP_SUB,

    /**
     * Floating-point to_fp from IEEE 754 bit-vector.
     *
     * SMT-LIB: `to_fp` (indexed)
     */
    BITWUZLA_KIND_FP_TO_FP_FROM_BV,

    /**
     * Floating-point to_fp from floating-point.
     *
     * SMT-LIB: `to_fp` (indexed)
     */
    BITWUZLA_KIND_FP_TO_FP_FROM_FP,

    /**
     * Floating-point to_fp from signed bit-vector value.
     *
     * SMT-LIB: `to_fp` (indexed)
     */
    BITWUZLA_KIND_FP_TO_FP_FROM_SBV,

    /**
     * Floating-point to_fp from unsigned bit-vector value.
     *
     * SMT-LIB: `to_fp_unsigned` (indexed)
     */
    BITWUZLA_KIND_FP_TO_FP_FROM_UBV,

    /**
     * Floating-point to_sbv.
     *
     * SMT-LIB: `fp.to_sbv` (indexed)
     */
    BITWUZLA_KIND_FP_TO_SBV,

    /**
     * Floating-point to_ubv.
     *
     * SMT-LIB: `fp.to_ubv` (indexed)
     */
    BITWUZLA_KIND_FP_TO_UBV;

    companion object {
        private val valueMapping = BitwuzlaKind.values().associateBy { it.ordinal }
        fun fromValue(value: Int): BitwuzlaKind = valueMapping.getValue(value)
    }
}
