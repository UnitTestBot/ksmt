package io.github.cvc5

@Suppress("unused")
fun Solver.mkQuantifier(
    isUniversal: Boolean,
    boundVars: Array<Term>,
    body: Term,
    patterns: Array<Term>
): Term {
    val kind = if (isUniversal) Kind.FORALL else Kind.EXISTS

    val quantifiedVars = mkTerm(Kind.VARIABLE_LIST, boundVars)
    val pattern = mkTerm(Kind.INST_PATTERN, patterns)

    return mkTerm(kind, quantifiedVars, body, pattern)
}

fun Solver.mkQuantifier(
    isUniversal: Boolean,
    boundVars: Array<Term>,
    body: Term
): Term {
    val kind = if (isUniversal) Kind.FORALL else Kind.EXISTS

    val quantifiedVars = mkTerm(Kind.VARIABLE_LIST, boundVars)

    return mkTerm(kind, quantifiedVars, body)
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

val Term.bvSizeToConvertTo: Int
    get() {
        require(kind == Kind.INT_TO_BITVECTOR) { "Required op is ${Kind.INT_TO_BITVECTOR}, but was $kind" }
        return op[0].integerValue.toInt()
    }

val Term.toFpExponentSize: Int
    get() {
        require(kind == Kind.FLOATINGPOINT_TO_FP_FROM_IEEE_BV) { "Required op is ${Kind.FLOATINGPOINT_TO_FP_FROM_IEEE_BV}, but was $kind" }
        return op[0].integerValue.toInt()
    }

val Term.toFpSignificandSize: Int
    get() {
        require(kind == Kind.FLOATINGPOINT_TO_FP_FROM_IEEE_BV) { "Required op is ${Kind.FLOATINGPOINT_TO_FP_FROM_IEEE_BV}, but was $kind" }
        return op[1].integerValue.toInt()
    }

val Term.intDivisibleArg: Int
    get() {
        require(kind == Kind.DIVISIBLE) { "Required op is ${Kind.DIVISIBLE}, but was $kind" }
        return op[0].integerValue.toInt()
    }

val Term.integerAndBitWidth: Int
    get() {
        require(kind == Kind.IAND) { "Required op is ${Kind.IAND}, but was $kind" }
        return op[0].integerValue.toInt()
    }