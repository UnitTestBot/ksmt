package io.github.cvc5

fun Solver.mkQuantifier(
    isUniversal: Boolean,
    boundVars: Array<Term>,
    body: Term
): Term {
    val kind = if (isUniversal) Kind.FORALL else Kind.EXISTS

    val quantifiedVars = mkTerm(Kind.VARIABLE_LIST, boundVars)

    return mkTerm(kind, quantifiedVars, body)
}

fun Solver.mkLambda(
    boundVars: Array<Term>,
    body: Term
): Term {
    val lambdaVars = mkTerm(Kind.VARIABLE_LIST, boundVars)
    return mkTerm(Kind.LAMBDA, lambdaVars, body)
}

val Term.bvZeroExtensionSize: Int
    get() {
        require(kind == Kind.BITVECTOR_ZERO_EXTEND) { "Required op is ${Kind.BITVECTOR_ZERO_EXTEND}, but was $kind" }
        return op[0].integerValue.toInt()
    }

val Term.bvSignExtensionSize: Int
    get() {
        require(kind == Kind.BITVECTOR_SIGN_EXTEND) { "Required op is ${Kind.BITVECTOR_SIGN_EXTEND}, but was $kind" }
        return op[0].integerValue.toInt()
    }

val Term.bvUpperExtractionBitIndex: Int
    get() {
        require(kind == Kind.BITVECTOR_EXTRACT) { "Required op is ${Kind.BITVECTOR_EXTRACT}, but was $kind" }
        return op[0].integerValue.toInt()
    }

val Term.bvLowerExtractionBitIndex: Int
    get() {
        require(kind == Kind.BITVECTOR_EXTRACT) { "Required op is ${Kind.BITVECTOR_EXTRACT}, but was $kind" }
        return op[1].integerValue.toInt()
    }

val Term.bvRotateBitsCountTerm: Term
    get() {
        require(kind == Kind.BITVECTOR_ROTATE_LEFT || kind == Kind.BITVECTOR_ROTATE_RIGHT) {
            "Required op is ${Kind.BITVECTOR_ROTATE_LEFT} or ${Kind.BITVECTOR_ROTATE_RIGHT}, but was $kind"
        }
        return op[0]
    }

val Term.bvRotateBitsCount: Int
    get() = bvRotateBitsCountTerm.integerValue.toInt()

val Term.bvRepeatTimes: Int
    get() {
        require(kind == Kind.BITVECTOR_REPEAT) { "Required op is ${Kind.BITVECTOR_REPEAT}, but was $kind" }
        return op[0].integerValue.toInt()
    }

private val toBvAllowedOps = listOf(
    Kind.INT_TO_BITVECTOR,
    Kind.FLOATINGPOINT_TO_SBV,
    Kind.FLOATINGPOINT_TO_UBV
)
val Term.bvSizeToConvertTo: Int
    get() {
        require(kind in toBvAllowedOps) { "Required op are $toBvAllowedOps, but was $kind" }
        return op[0].integerValue.toInt()
    }

private val toFpAllowedOps = listOf(
    Kind.FLOATINGPOINT_TO_FP_FROM_IEEE_BV,
    Kind.FLOATINGPOINT_TO_FP_FROM_REAL,
    Kind.FLOATINGPOINT_TO_FP_FROM_FP,
    Kind.FLOATINGPOINT_TO_FP_FROM_SBV,
    Kind.FLOATINGPOINT_TO_FP_FROM_UBV

)

val Term.toFpExponentSize: Int
    get() {
        require(kind in toFpAllowedOps) { "Required ops are $toFpAllowedOps, but was $kind" }
        return op[0].integerValue.toInt()
    }

val Term.toFpSignificandSize: Int
    get() {
        require(kind in toFpAllowedOps) { "Required ops are $toFpAllowedOps, but was $kind" }
        return op[1].integerValue.toInt()
    }

val Term.intDivisibleArg: Int
    get() {
        require(kind == Kind.DIVISIBLE) { "Required op is ${Kind.DIVISIBLE}, but was $kind" }
        return op[0].integerValue.toInt()
    }
