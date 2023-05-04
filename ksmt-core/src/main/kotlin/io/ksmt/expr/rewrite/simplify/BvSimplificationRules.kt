package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KBitVecValue
import io.ksmt.expr.KBvAddExpr
import io.ksmt.expr.KBvAndExpr
import io.ksmt.expr.KBvConcatExpr
import io.ksmt.expr.KBvExtractExpr
import io.ksmt.expr.KBvMulExpr
import io.ksmt.expr.KBvNegationExpr
import io.ksmt.expr.KBvNotExpr
import io.ksmt.expr.KBvOrExpr
import io.ksmt.expr.KBvShiftLeftExpr
import io.ksmt.expr.KBvXorExpr
import io.ksmt.expr.KExpr
import io.ksmt.expr.KIteExpr
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KIntSort
import io.ksmt.utils.BvUtils
import io.ksmt.utils.BvUtils.bigIntValue
import io.ksmt.utils.BvUtils.bitwiseAnd
import io.ksmt.utils.BvUtils.bitwiseNot
import io.ksmt.utils.BvUtils.bitwiseOr
import io.ksmt.utils.BvUtils.bitwiseXor
import io.ksmt.utils.BvUtils.bvMaxValueSigned
import io.ksmt.utils.BvUtils.bvMaxValueUnsigned
import io.ksmt.utils.BvUtils.bvMinValueSigned
import io.ksmt.utils.BvUtils.bvOne
import io.ksmt.utils.BvUtils.bvValue
import io.ksmt.utils.BvUtils.bvValueIs
import io.ksmt.utils.BvUtils.bvZero
import io.ksmt.utils.BvUtils.extractBv
import io.ksmt.utils.BvUtils.isBvMaxValueSigned
import io.ksmt.utils.BvUtils.isBvMaxValueUnsigned
import io.ksmt.utils.BvUtils.isBvMinValueSigned
import io.ksmt.utils.BvUtils.isBvOne
import io.ksmt.utils.BvUtils.isBvZero
import io.ksmt.utils.BvUtils.minus
import io.ksmt.utils.BvUtils.plus
import io.ksmt.utils.BvUtils.powerOfTwoOrNull
import io.ksmt.utils.BvUtils.shiftLeft
import io.ksmt.utils.BvUtils.shiftRightArith
import io.ksmt.utils.BvUtils.shiftRightLogical
import io.ksmt.utils.BvUtils.signBit
import io.ksmt.utils.BvUtils.signExtension
import io.ksmt.utils.BvUtils.signedDivide
import io.ksmt.utils.BvUtils.signedGreaterOrEqual
import io.ksmt.utils.BvUtils.signedLessOrEqual
import io.ksmt.utils.BvUtils.signedMod
import io.ksmt.utils.BvUtils.signedRem
import io.ksmt.utils.BvUtils.times
import io.ksmt.utils.BvUtils.toBigIntegerSigned
import io.ksmt.utils.BvUtils.toBigIntegerUnsigned
import io.ksmt.utils.BvUtils.unaryMinus
import io.ksmt.utils.BvUtils.unsignedDivide
import io.ksmt.utils.BvUtils.unsignedLessOrEqual
import io.ksmt.utils.BvUtils.unsignedRem
import io.ksmt.utils.BvUtils.zeroExtension
import io.ksmt.utils.cast
import io.ksmt.utils.toBigInteger
import io.ksmt.utils.uncheckedCast
import java.math.BigInteger

inline fun <T : KBvSort> KContext.simplifyBvNotExprLight(
    arg: KExpr<T>,
    cont: (KExpr<T>) -> KExpr<T>
): KExpr<T> = when (arg) {
    is KBitVecValue<T> -> arg.bitwiseNot().uncheckedCast()
    // (bvnot (bvnot a)) ==> a
    is KBvNotExpr<T> -> arg.value
    else -> cont(arg)
}

/** (bvnot (concat a b)) ==> (concat (bvnot a) (bvnot b)) */
inline fun <T : KBvSort> KContext.simplifyBvNotExprConcat(
    arg: KExpr<T>,
    rewriteBvNotExpr: KContext.(KExpr<KBvSort>) -> KExpr<KBvSort>,
    rewriteBvConcatExpr: KContext.(List<KExpr<KBvSort>>) -> KExpr<KBvSort>,
    cont: (KExpr<T>) -> KExpr<T>
): KExpr<T> =
    if (arg is KBvConcatExpr) {
        val concatParts = flatConcatArgs(arg)
        val negatedParts = concatParts.map { rewriteBvNotExpr(it) }
        rewriteBvConcatExpr(negatedParts).uncheckedCast()
    } else {
        cont(arg)
    }

/** (bvnot (ite c a b)) ==> (ite c (bvnot a) (bvnot b)) */
inline fun <T : KBvSort> KContext.simplifyBvNotExprIte(
    arg: KExpr<T>,
    rewriteBvNotExpr: KContext.(KExpr<T>) -> KExpr<T>,
    rewriteBvIte: KContext.(KExpr<KBoolSort>, KExpr<T>, KExpr<T>) -> KExpr<T>,
    cont: (KExpr<T>) -> KExpr<T>
): KExpr<T> =
    if (arg is KIteExpr<T> && (arg.trueBranch is KBitVecValue<T> || arg.falseBranch is KBitVecValue<T>)) {
        val trueBranch = rewriteBvNotExpr(arg.trueBranch)
        val falseBranch = rewriteBvNotExpr(arg.falseBranch)

        rewriteBvIte(arg.condition, trueBranch, falseBranch)
    } else {
        cont(arg)
    }

inline fun <T : KBvSort> KContext.simplifyBvOrExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> = evalBvOperation(lhs, rhs, { a, b -> a.bitwiseOr(b) }) {
    executeIfValue(lhs, rhs) { value, other ->
        if (value.isBvMaxValueUnsigned()) return value
        if (value.isBvZero()) return other
    }

    return cont(lhs, rhs)
}.uncheckedCast()

/** (bvor const1 (bvor const2 x)) ==> (bvor (bvor const1 const2) x) */
inline fun <T : KBvSort> KContext.simplifyBvOrExprNestedOr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvOrExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    executeIfValue(lhs, rhs) { value, other ->
        if (other is KBvOrExpr<T>) {
            executeIfValue(other.arg0, other.arg1) { otherValue, otherOther ->
                return rewriteBvOrExpr(value.bitwiseOr(otherValue).uncheckedCast(), otherOther)
            }
        }
    }

    return cont(lhs, rhs)
}

inline fun <T : KBvSort> KContext.simplifyFlatBvOrExpr(
    args: List<KExpr<T>>,
    cont: (List<KExpr<T>>) -> KExpr<T>
): KExpr<T> {
    require(args.isNotEmpty()) {
        "Bitvector flat or requires at least a single argument"
    }

    return simplifyBvAndOr(
        args = args,
        neutralElement = bvZero(args.first().sort.sizeBits).uncheckedCast(),
        isZeroElement = { it.isBvMaxValueUnsigned() },
        buildZeroElement = { bvMaxValueUnsigned(args.first().sort.sizeBits).uncheckedCast() },
        operation = { a, b -> a.bitwiseOr(b).uncheckedCast() },
        cont = cont
    )
}

/**
 * (bvor (concat a b) c) ==>
 *  (concat
 *      (bvor (extract (0, <a_size>) c))
 *      (bvor b (extract (<a_size>, <a_size> + <b_size>) c))
 *  )
 * */
inline fun <T : KBvSort> KContext.simplifyFlatBvOrExprDistributeOverConcat(
    args: List<KExpr<T>>,
    rewriteBvExtractExpr: KContext.(Int, Int, KExpr<KBvSort>) -> KExpr<KBvSort>,
    rewriteFlatBvOrExpr: KContext.(List<KExpr<KBvSort>>) -> KExpr<KBvSort>,
    rewriteBvConcatExpr: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBvSort>,
    cont: (List<KExpr<T>>) -> KExpr<T>
): KExpr<T> =
    if (args.any { it is KBvConcatExpr }) {
        distributeOperationOverConcat(
            args = args,
            rewriteBvExtractExpr = rewriteBvExtractExpr,
            rewriteFlatBvOpExpr = rewriteFlatBvOrExpr,
            rewriteBvConcatExpr = rewriteBvConcatExpr
        )
    } else {
        cont(args)
    }


inline fun <T : KBvSort> KContext.simplifyBvAndExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> = evalBvOperation(lhs, rhs, { a, b -> a.bitwiseAnd(b) }) {
    executeIfValue(lhs, rhs) { value, other ->
        if (value.isBvMaxValueUnsigned()) return other
        if (value.isBvZero()) return value
    }

    return cont(lhs, rhs)
}.uncheckedCast()

/** (bvand const1 (bvand const2 x)) ==> (bvand (bvand const1 const2) x) */
inline fun <T : KBvSort> KContext.simplifyBvAndExprNestedAnd(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvAndExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    executeIfValue(lhs, rhs) { value, other ->
        if (other is KBvAndExpr<T>) {
            executeIfValue(other.arg0, other.arg1) { otherValue, otherOther ->
                return rewriteBvAndExpr(value.bitwiseAnd(otherValue).uncheckedCast(), otherOther)
            }
        }
    }

    return cont(lhs, rhs)
}

inline fun <T : KBvSort> KContext.simplifyFlatBvAndExpr(
    args: List<KExpr<T>>,
    cont: (List<KExpr<T>>) -> KExpr<T>
): KExpr<T> {
    require(args.isNotEmpty()) {
        "Bitvector flat and requires at least a single argument"
    }

    return simplifyBvAndOr(
        args = args,
        neutralElement = bvMaxValueUnsigned(args.first().sort.sizeBits).uncheckedCast(),
        isZeroElement = { it.isBvZero() },
        buildZeroElement = { bvZero(args.first().sort.sizeBits).uncheckedCast() },
        operation = { a, b -> a.bitwiseAnd(b).uncheckedCast() },
        cont = cont
    )
}

/**
 * (bvand (concat a b) c) ==>
 *  (concat
 *      (bvand (extract (0, <a_size>) c))
 *      (bvand b (extract (<a_size>, <a_size> + <b_size>) c))
 *  )
 * */
inline fun <T : KBvSort> KContext.simplifyFlatBvAndExprDistributeOverConcat(
    args: List<KExpr<T>>,
    rewriteBvExtractExpr: KContext.(Int, Int, KExpr<KBvSort>) -> KExpr<KBvSort>,
    rewriteFlatBvAndExpr: KContext.(List<KExpr<KBvSort>>) -> KExpr<KBvSort>,
    rewriteBvConcatExpr: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBvSort>,
    cont: (List<KExpr<T>>) -> KExpr<T>
): KExpr<T> =
    if (args.any { it is KBvConcatExpr }) {
        distributeOperationOverConcat(
            args = args,
            rewriteBvExtractExpr = rewriteBvExtractExpr,
            rewriteFlatBvOpExpr = rewriteFlatBvAndExpr,
            rewriteBvConcatExpr = rewriteBvConcatExpr
        )
    } else {
        cont(args)
    }

inline fun <T : KBvSort> KContext.simplifyBvXorExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> = evalBvOperation(lhs, rhs, { a, b -> a.bitwiseXor(b) }) {
    executeIfValue(lhs, rhs) { value, other ->
        if (value.isBvZero()) return other
    }

    return cont(lhs, rhs)
}.uncheckedCast()

/** (bvxor const1 (bvxor const2 x)) ==> (bvxor (bvxor const1 const2) x) */
inline fun <T : KBvSort> KContext.simplifyBvXorExprNestedXor(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvXorExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    executeIfValue(lhs, rhs) { value, other ->
        if (other is KBvXorExpr<T>) {
            executeIfValue(other.arg0, other.arg1) { otherValue, otherOther ->
                return rewriteBvXorExpr(value.bitwiseXor(otherValue).uncheckedCast(), otherOther)
            }
        }
    }

    return cont(lhs, rhs)
}

/** (bvxor 0xFFFF... a) ==> (bvnot a) */
inline fun <T : KBvSort> KContext.simplifyBvXorExprMaxConst(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvNotExpr: KContext.(KExpr<T>) -> KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    executeIfValue(lhs, rhs) { value, other ->
        if (value.isBvMaxValueUnsigned()) return rewriteBvNotExpr(other)
    }

    return cont(lhs, rhs)
}

@Suppress("LongMethod", "ComplexMethod")
inline fun <T : KBvSort> KContext.simplifyFlatBvXorExpr(
    args: List<KExpr<T>>,
    cont: (Boolean, List<KExpr<T>>) -> KExpr<T>
): KExpr<T> {
    require(args.isNotEmpty()) {
        "Bitvector flat xor requires at least a single argument"
    }

    val zero = bvZero(args.first().sort.sizeBits)
    val maxValue = bvMaxValueUnsigned(args.first().sort.sizeBits)
    var constantValue = zero

    val positiveParts = mutableSetOf<KExpr<T>>()
    val negativeParts = mutableSetOf<KExpr<T>>()

    for (arg in args) {
        if (arg is KBitVecValue<T>) {
            constantValue = constantValue.bitwiseXor(arg)
            continue
        }

        if (arg is KBvNotExpr<T>) {
            when (val term = arg.value) {
                in negativeParts -> {
                    // (bxor (bvnot a) b (bvnot a)) ==> (bvxor 0 b)
                    negativeParts.remove(term)
                }

                in positiveParts -> {
                    // (bvxor a b (bvnot a)) ==> (bvxor b 0xFFFF...)
                    positiveParts.remove(term)
                    constantValue = constantValue.bitwiseXor(maxValue)
                }

                else -> {
                    negativeParts.add(term)
                }
            }
        } else {
            when (arg) {
                in positiveParts -> {
                    // (bvxor a b a) ==> (bvxor 0 b)
                    positiveParts.remove(arg)
                }

                in negativeParts -> {
                    // (bvxor (bvnot a) b a) ==> (bvxor b 0xFFFF...)
                    negativeParts.remove(arg)
                    constantValue = constantValue.bitwiseXor(maxValue)
                }

                else -> {
                    positiveParts.add(arg)
                }
            }
        }
    }

    val resultParts = arrayListOf<KExpr<T>>().apply {
        addAll(positiveParts)
        addAll(negativeParts.map { mkBvNotExprNoSimplify(it) })
    }

    if (resultParts.isEmpty()) {
        return constantValue.uncheckedCast()
    }

    var negateResult = false
    when (constantValue) {
        zero -> {
            // (bvxor 0 a) ==> a
        }
        maxValue -> {
            // (bvxor 0xFFFF... a) ==> (bvnot a)
            negateResult = true
        }
        else -> {
            resultParts.add(constantValue.uncheckedCast())
        }
    }

    return cont(negateResult, resultParts)
}

/**
 * (bvxor (concat a b) c) ==>
 *  (concat
 *      (bvxor (extract (0, <a_size>) c))
 *      (bvxor b (extract (<a_size>, <a_size> + <b_size>) c))
 *  )
 * */
inline fun <T : KBvSort> KContext.simplifyFlatBvXorExprDistributeOverConcat(
    args: List<KExpr<T>>,
    rewriteBvExtractExpr: KContext.(Int, Int, KExpr<KBvSort>) -> KExpr<KBvSort>,
    rewriteFlatBvXorExpr: KContext.(List<KExpr<KBvSort>>) -> KExpr<KBvSort>,
    rewriteBvConcatExpr: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBvSort>,
    cont: (List<KExpr<T>>) -> KExpr<T>
): KExpr<T> =
    if (args.any { it is KBvConcatExpr }) {
        distributeOperationOverConcat(
            args = args,
            rewriteBvExtractExpr = rewriteBvExtractExpr,
            rewriteFlatBvOpExpr = rewriteFlatBvXorExpr,
            rewriteBvConcatExpr = rewriteBvConcatExpr
        )
    } else {
        cont(args)
    }

/** (bvnor a b) ==> (bvnot (bvor a b)) */
inline fun <T : KBvSort> KContext.rewriteBvNorExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvOrExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<T>,
    rewriteBvNotExpr: KContext.(KExpr<T>) -> KExpr<T>
): KExpr<T> = rewriteBvNotExpr(rewriteBvOrExpr(lhs, rhs))

/** (bvnand a b) ==> (bvor (bvnot a) (bvnot b)) */
inline fun <T : KBvSort> KContext.rewriteBvNAndExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvOrExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<T>,
    rewriteBvNotExpr: KContext.(KExpr<T>) -> KExpr<T>
): KExpr<T> = rewriteBvOrExpr(rewriteBvNotExpr(lhs), rewriteBvNotExpr(rhs))

/** (bvxnor a b) ==> (bvnot (bvxor a b)) */
inline fun <T : KBvSort> KContext.rewriteBvXNorExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvXorExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<T>,
    rewriteBvNotExpr: KContext.(KExpr<T>) -> KExpr<T>
): KExpr<T> = rewriteBvNotExpr(rewriteBvXorExpr(lhs, rhs))

inline fun <T : KBvSort> KContext.simplifyBvNegationExprLight(
    arg: KExpr<T>,
    cont: (KExpr<T>) -> KExpr<T>
): KExpr<T> = when (arg) {
    is KBitVecValue<T> -> (-arg).uncheckedCast()
    is KBvNegationExpr<T> -> arg.value
    else -> cont(arg)
}

/** (bvneg (bvadd a b)) ==> (bvadd (bvneg a) (bvneg b)) */
inline fun <T : KBvSort> KContext.simplifyBvNegationExprAdd(
    arg: KExpr<T>,
    rewriteBvAddExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<T>,
    rewriteBvNegationExpr: KContext.(KExpr<T>) -> KExpr<T>,
    cont: (KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (arg is KBvAddExpr<T>) {
        val lhsIsSuitableFoRewrite = arg.arg0 is KBitVecValue<T> || arg.arg0 is KBvNegationExpr<T>
        val rhsIsSuitableFoRewrite = arg.arg1 is KBitVecValue<T> || arg.arg1 is KBvNegationExpr<T>
        if (lhsIsSuitableFoRewrite || rhsIsSuitableFoRewrite) {
            return rewriteBvAddExpr(rewriteBvNegationExpr(arg.arg0), rewriteBvNegationExpr(arg.arg1))
        }
    }

    return cont(arg)
}

inline fun <T : KBvSort> KContext.simplifyBvAddExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> = evalBvOperation(lhs, rhs, { a, b -> a + b }) {
    executeIfValue(lhs, rhs) { value, other ->
        if (value.isBvZero()) return other
    }

    return cont(lhs, rhs)
}.uncheckedCast()

/** (+ const1 (+ const2 x)) ==> (+ (+ const1 const2) x) */
inline fun <T : KBvSort> KContext.simplifyBvAddExprNestedAdd(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvAddExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    executeIfValue(lhs, rhs) { value, other ->
        if (other is KBvAddExpr<T>) {
            executeIfValue(other.arg0, other.arg1) { otherValue, otherOther ->
                return rewriteBvAddExpr((value + otherValue).uncheckedCast(), otherOther)
            }
        }
    }

    return cont(lhs, rhs)
}

inline fun <T : KBvSort> KContext.simplifyFlatBvAddExpr(
    args: List<KExpr<T>>,
    cont: (List<KExpr<T>>) -> KExpr<T>
): KExpr<T> {
    require(args.isNotEmpty()) {
        "Bitvector flat add requires at least a single argument"
    }

    val zero = bvZero(args.first().sort.sizeBits)
    var constantValue = zero
    val resultParts = arrayListOf<KExpr<T>>()

    for (arg in args) {
        if (arg is KBitVecValue<T>) {
            constantValue += arg
            continue
        }
        resultParts += arg
    }

    if (resultParts.isEmpty()) {
        return constantValue.uncheckedCast()
    }

    if (constantValue != zero) {
        resultParts.add(constantValue.uncheckedCast())
    }

    return cont(resultParts)
}

inline fun <T : KBvSort> KContext.simplifyBvMulExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> = evalBvOperation(lhs, rhs, { a, b -> a * b }) {
    executeIfValue(lhs, rhs) { value, other ->
        if (value.isBvZero()) return value
        if (value.isBvOne()) return other
    }

    return cont(lhs, rhs)
}.uncheckedCast()

/** (* const1 (* const2 x)) ==> (* (* const1 const2) x) */
inline fun <T : KBvSort> KContext.simplifyBvMulExprNestedMul(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvMulExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    executeIfValue(lhs, rhs) { value, other ->
        if (other is KBvMulExpr<T>) {
            executeIfValue(other.arg0, other.arg1) { otherValue, otherOther ->
                return rewriteBvMulExpr((value * otherValue).uncheckedCast(), otherOther)
            }
        }
    }

    return cont(lhs, rhs)
}

/** (* -1 a) ==> -a */
inline fun <T : KBvSort> KContext.simplifyBvMulExprMinusOneConst(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvNegationExpr: KContext.(KExpr<T>) -> KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    executeIfValue(lhs, rhs) { value, other ->
        if (value.bvValueIs(-1)) return rewriteBvNegationExpr(other)
    }

    return cont(lhs, rhs)
}

inline fun <T : KBvSort> KContext.simplifyFlatBvMulExpr(
    args: List<KExpr<T>>,
    cont: (negateResult: Boolean, List<KExpr<T>>) -> KExpr<T>
): KExpr<T> {
    require(args.isNotEmpty()) {
        "Bitvector flat mul requires at least a single argument"
    }

    val zero = bvZero(args.first().sort.sizeBits)
    val one = bvOne(args.first().sort.sizeBits)

    var constantValue = one
    val resultParts = arrayListOf<KExpr<T>>()

    for (arg in args) {
        if (arg is KBitVecValue<T>) {
            constantValue *= arg
            continue
        }
        resultParts += arg
    }

    // (* 0 a) ==> 0
    if (constantValue.isBvZero()) {
        return zero.uncheckedCast()
    }

    if (resultParts.isEmpty()) {
        return constantValue.uncheckedCast()
    }

    // (* 1 a) ==> a
    if (constantValue.isBvOne()) {
        return cont(false, resultParts)
    }

    // (* -1 a) ==> -a
    val minusOne = zero - one
    if (constantValue == minusOne) {
        return cont(true, resultParts)
    }

    resultParts.add(constantValue.uncheckedCast())
    return cont(false, resultParts)
}

/** (- a b) ==> (+ a -b) */
inline fun <T : KBvSort> KContext.rewriteBvSubExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvAddExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<T>,
    rewriteBvNegationExpr: KContext.(KExpr<T>) -> KExpr<T>
): KExpr<T> = rewriteBvAddExpr(lhs, rewriteBvNegationExpr(rhs))

inline fun <T : KBvSort> KContext.simplifyBvSignedDivExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (rhs is KBitVecValue<T> && !rhs.isBvZero()) {
        if (lhs is KBitVecValue<T>) {
            return lhs.signedDivide(rhs).uncheckedCast()
        }

        if (rhs.isBvOne()) {
            return lhs
        }
    }
    return cont(lhs, rhs)
}

inline fun <T : KBvSort> KContext.simplifyBvSignedModExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (rhs is KBitVecValue<T> && !rhs.isBvZero()) {
        if (lhs is KBitVecValue<T>) {
            return lhs.signedMod(rhs).uncheckedCast()
        }

        if (rhs.isBvOne()) {
            return bvZero(rhs.sort.sizeBits).uncheckedCast()
        }
    }
    return cont(lhs, rhs)
}

inline fun <T : KBvSort> KContext.simplifyBvSignedRemExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (rhs is KBitVecValue<T> && !rhs.isBvZero()) {
        if (lhs is KBitVecValue<T>) {
            return lhs.signedRem(rhs).uncheckedCast()
        }

        if (rhs.isBvOne()) {
            return bvZero(rhs.sort.sizeBits).uncheckedCast()
        }
    }
    return cont(lhs, rhs)
}

inline fun <T : KBvSort> KContext.simplifyBvUnsignedDivExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (rhs is KBitVecValue<T> && !rhs.isBvZero()) {
        if (lhs is KBitVecValue<T>) {
            return lhs.unsignedDivide(rhs).uncheckedCast()
        }

        if (rhs.isBvOne()) {
            return lhs
        }
    }
    return cont(lhs, rhs)
}

/** (udiv a x), x == 2^n ==> (lshr a n) */
inline fun <T : KBvSort> KContext.simplifyBvUnsignedDivExprPowerOfTwoDivisor(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvLogicalShiftRightExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (rhs is KBitVecValue<T> && !rhs.isBvZero()) {
        rhs.powerOfTwoOrNull()?.let { powerOfTwo ->
            return rewriteBvLogicalShiftRightExpr(lhs, mkBv(powerOfTwo, rhs.sort.sizeBits).uncheckedCast())
        }
    }

    return cont(lhs, rhs)
}

inline fun <T : KBvSort> KContext.simplifyBvUnsignedRemExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (rhs is KBitVecValue<T> && !rhs.isBvZero()) {
        if (lhs is KBitVecValue<T>) {
            return lhs.unsignedRem(rhs).uncheckedCast()
        }

        if (rhs.isBvOne()) {
            return bvZero(rhs.sort.sizeBits).uncheckedCast()
        }
    }
    return cont(lhs, rhs)
}

/** (urem a x), x == 2^n ==> (concat 0 (extract [n-1:0] a)) */
inline fun <T : KBvSort> KContext.simplifyBvUnsignedRemExprPowerOfTwoDivisor(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvExtractExpr: KContext.(Int, Int, KExpr<T>) -> KExpr<KBvSort>,
    rewriteBvZeroExtensionExpr: KContext.(Int, KExpr<T>) -> KExpr<KBvSort>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (rhs is KBitVecValue<T> && !rhs.isBvZero()) {
        val powerOfTwo = rhs.powerOfTwoOrNull()
        if (powerOfTwo != null) {
            // take all bits
            if (powerOfTwo >= rhs.sort.sizeBits.toInt()) {
                return lhs
            }

            val remainderBits = rewriteBvExtractExpr(powerOfTwo - 1, 0, lhs)
            val normalizedRemainder = rewriteBvZeroExtensionExpr(
                rhs.sort.sizeBits.toInt() - powerOfTwo, remainderBits.uncheckedCast()
            )
            return normalizedRemainder.uncheckedCast()
        }
    }

    return cont(lhs, rhs)
}

inline fun <T : KBvSort> KContext.simplifyBvReductionAndExprLight(
    arg: KExpr<T>,
    cont: (KExpr<T>) -> KExpr<KBv1Sort>
): KExpr<KBv1Sort> =
    if (arg is KBitVecValue<T>) {
        // 0xFFFFF -> 1 and 0 otherwise
        mkBv(arg.isBvMaxValueUnsigned())
    } else {
        cont(arg)
    }

inline fun <T : KBvSort> KContext.simplifyBvReductionOrExprLight(
    arg: KExpr<T>,
    cont: (KExpr<T>) -> KExpr<KBv1Sort>
): KExpr<KBv1Sort> =
    if (arg is KBitVecValue<T>) {
        // 0x00000 -> 0 and 1 otherwise
        mkBv(!arg.isBvZero())
    } else {
        cont(arg)
    }

inline fun <T : KBvSort> KContext.simplifyEqBvLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> {
    if (lhs == rhs) return trueExpr

    if (lhs is KBitVecValue<T> && rhs is KBitVecValue<T>) {
        return falseExpr
    }

    return cont(lhs, rhs)
}

@Suppress("LongParameterList")
inline fun <T : KBvSort> KContext.simplifyEqBvConcat(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    crossinline rewriteBvExtractExpr: KContext.(Int, Int, KExpr<KBvSort>) -> KExpr<KBvSort>,
    crossinline rewriteBvEq: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBoolSort>,
    crossinline rewriteFlatAnd: KContext.(List<KExpr<KBoolSort>>) -> KExpr<KBoolSort>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> =
    if (lhs is KBvConcatExpr || rhs is KBvConcatExpr) {
        rewriteBvConcatEq(
            lhs,
            rhs,
            { high, low, value -> rewriteBvExtractExpr(high, low, value) },
            { l, r -> rewriteBvEq(l, r) },
            { args -> rewriteFlatAnd(args)}
        )
    } else {
        cont(lhs, rhs)
    }

inline fun <T : KBvSort> KContext.simplifyBvArithShiftRightExprLight(
    lhs: KExpr<T>,
    shift: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (shift is KBitVecValue<T>) {
        // (x >> 0) ==> x
        if (shift.isBvZero()) {
            return lhs
        }

        if (lhs is KBitVecValue<T>) {
            return lhs.shiftRightArith(shift).uncheckedCast()
        }
    }
    return cont(lhs, shift)
}

inline fun <T : KBvSort> KContext.simplifyBvLogicalShiftRightExprLight(
    lhs: KExpr<T>,
    shift: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (shift is KBitVecValue<T>) {
        // (x >>> 0) ==> x
        if (shift.isBvZero()) {
            return lhs
        }

        // (x >>> shift), shift >= size ==> 0
        if (shift.signedGreaterOrEqual(shift.sort.sizeBits.toInt())) {
            return bvZero(shift.sort.sizeBits).uncheckedCast()
        }

        if (lhs is KBitVecValue<T>) {
            return lhs.shiftRightLogical(shift).uncheckedCast()
        }
    }

    // (x >>> x) ==> 0
    if (lhs == shift) {
        return bvZero(shift.sort.sizeBits).uncheckedCast()
    }

    return cont(lhs, shift)
}

/** (bvlshr x shift) ==> (concat 0.[shift].0 (extract [size-1:shift] x)) */
inline fun <T : KBvSort> KContext.simplifyBvLogicalShiftRightExprConstShift(
    arg: KExpr<T>,
    shift: KExpr<T>,
    rewriteBvExtractExpr: KContext.(Int, Int, KExpr<KBvSort>) -> KExpr<KBvSort>,
    rewriteBvConcatExpr: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBvSort>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (shift is KBitVecValue<T>) {
        val intShiftValue = shift.bigIntValue()
        if (intShiftValue >= BigInteger.ZERO && intShiftValue <= Int.MAX_VALUE.toBigInteger()) {
            val lhs = bvZero(intShiftValue.toInt().toUInt())
            val rhs = rewriteBvExtractExpr(arg.sort.sizeBits.toInt() - 1, intShiftValue.toInt(), arg.uncheckedCast())
            return rewriteBvConcatExpr(lhs.uncheckedCast(), rhs).uncheckedCast()
        }
    }

    return cont(arg, shift)
}

inline fun <T : KBvSort> KContext.simplifyBvShiftLeftExprLight(
    lhs: KExpr<T>,
    shift: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (shift is KBitVecValue<T>) {
        if (lhs is KBitVecValue<T>) {
            return lhs.shiftLeft(shift).uncheckedCast()
        }

        // (x << 0) ==> x
        if (shift.isBvZero()) {
            return lhs
        }

        // (x << shift), shift >= size ==> 0
        if (shift.signedGreaterOrEqual(shift.sort.sizeBits.toInt())) {
            return bvZero(shift.sort.sizeBits).uncheckedCast()
        }
    }
    return cont(lhs, shift)
}

/** (bvshl x shift) ==> (concat (extract [size-1-shift:0] x) 0.[shift].0) */
inline fun <T : KBvSort> KContext.simplifyBvShiftLeftExprConstShift(
    lhs: KExpr<T>,
    shift: KExpr<T>,
    rewriteBvExtractExpr: KContext.(Int, Int, KExpr<KBvSort>) -> KExpr<KBvSort>,
    rewriteBvConcatExpr: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBvSort>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    if (shift is KBitVecValue<T>) {
        val intShiftValue = shift.bigIntValue()
        if (intShiftValue >= BigInteger.ZERO && intShiftValue <= Int.MAX_VALUE.toBigInteger()) {
            return rewriteBvConcatExpr(
                rewriteBvExtractExpr(lhs.sort.sizeBits.toInt() - 1 - intShiftValue.toInt(), 0, lhs.uncheckedCast()),
                bvZero(intShiftValue.toInt().toUInt()).uncheckedCast()
            ).uncheckedCast()
        }

    }
    return cont(lhs, shift)
}

/**
 * (bvshl (bvshl x nestedShift) shift) ==>
 *      (ite (bvule nestedShift (+ nestedShift shift)) (bvshl x (+ nestedShift shift)) 0)
 * */
@Suppress("LongParameterList")
inline fun <T : KBvSort> KContext.simplifyBvShiftLeftExprNestedShiftLeft(
    lhs: KExpr<T>,
    shift: KExpr<T>,
    rewriteBvAddExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<T>,
    rewriteBvUnsignedLessOrEqualExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>,
    rewriteBvShiftLeftExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<T>,
    rewriteIte: (KExpr<KBoolSort>, KExpr<T>, KExpr<T>) -> KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> =
    if (lhs is KBvShiftLeftExpr<T>) {
        val nestedArg = lhs.arg
        val nestedShift = lhs.shift
        val sum = rewriteBvAddExpr(nestedShift, shift)
        val cond = rewriteBvUnsignedLessOrEqualExpr(nestedShift, sum)
        rewriteIte(cond, rewriteBvShiftLeftExpr(nestedArg, sum), bvZero(lhs.sort.sizeBits).uncheckedCast())
    } else {
        cont(lhs, shift)
    }

inline fun <T : KBvSort> KContext.simplifyBvRotateLeftExprConstRotation(
    lhs: KExpr<T>,
    rotation: KExpr<T>,
    rewriteBvRotateLeftIndexedExpr: KContext.(Int, KExpr<T>) -> KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> = if (rotation is KBitVecValue<T>) {
    val size = rotation.sort.sizeBits.toInt()
    val intValue = rotation.bigIntValue()
    val rotationValue = intValue.remainder(size.toBigInteger()).toInt()
    rewriteBvRotateLeftIndexedExpr(rotationValue, lhs)
} else {
    cont(lhs, rotation)
}

/** (rotateLeft a x) ==> (concat (extract [size-1-x:0] a) (extract [size-1:size-x] a)) */
inline fun <T : KBvSort> KContext.rewriteBvRotateLeftIndexedExpr(
    rotation: Int,
    value: KExpr<T>,
    crossinline rewriteBvExtractExpr: KContext.(Int, Int, KExpr<T>) -> KExpr<KBvSort>,
    crossinline rewriteBvConcatExpr: KContext.(KExpr<T>, KExpr<KBvSort>) -> KExpr<KBvSort>
): KExpr<T> {
    val size = value.sort.sizeBits.toInt()
    return rotateLeft(
        arg = value,
        size = size,
        rotationNumber = rotation,
        rewriteBvExtractExpr = rewriteBvExtractExpr,
        rewriteBvConcatExpr = rewriteBvConcatExpr
    )
}

/** (rotateRight a x) ==> (rotateLeft a (- size x)) */
inline fun <T : KBvSort> KContext.simplifyBvRotateRightExprConstRotation(
    lhs: KExpr<T>,
    rotation: KExpr<T>,
    rewriteBvRotateRightIndexedExpr: KContext.(Int, KExpr<T>) -> KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> = if (rotation is KBitVecValue<T>) {
    val size = rotation.sort.sizeBits.toInt()
    val intValue = rotation.bigIntValue()
    val rotationValue = intValue.remainder(size.toBigInteger()).toInt()
    rewriteBvRotateRightIndexedExpr(rotationValue, lhs)
} else {
    cont(lhs, rotation)
}

/** (rotateRight a x) ==> (rotateLeft a (- size x)) */
inline fun <T : KBvSort> KContext.rewriteBvRotateRightIndexedExpr(
    rotation: Int,
    value: KExpr<T>,
    crossinline rewriteBvExtractExpr: KContext.(Int, Int, KExpr<T>) -> KExpr<KBvSort>,
    crossinline rewriteBvConcatExpr: KContext.(KExpr<T>, KExpr<KBvSort>) -> KExpr<KBvSort>
): KExpr<T> {
    val size = value.sort.sizeBits.toInt()
    val normalizedRotation = rotation % size
    return rotateLeft(
        arg = value,
        size = size,
        rotationNumber = size - normalizedRotation,
        rewriteBvExtractExpr = rewriteBvExtractExpr,
        rewriteBvConcatExpr = rewriteBvConcatExpr
    )
}

/** (repeat a x) ==> (concat a a ..[x].. a) */
inline fun <T : KBvSort> KContext.simplifyBvRepeatExprLight(
    repeatNumber: Int,
    value: KExpr<T>,
    cont: (Int, KExpr<T>) -> KExpr<KBvSort>
): KExpr<KBvSort> = if (value is KBitVecValue<T> && repeatNumber > 0) {
    var result: KBitVecValue<*> = value
    repeat(repeatNumber - 1) {
        result = BvUtils.concatBv(result, value)
    }
    result.uncheckedCast()
} else {
    cont(repeatNumber, value)
}

/** (repeat a x) ==> (concat a a ..[x].. a) */
inline fun KContext.rewriteBvRepeatExpr(
    repeatNumber: Int,
    value: KExpr<KBvSort>,
    rewriteFlatBvConcatExpr: KContext.(List<KExpr<KBvSort>>) -> KExpr<KBvSort>,
): KExpr<KBvSort> {
    val repeats = arrayListOf<KExpr<KBvSort>>()
    repeat(repeatNumber) {
        repeats += value
    }

    if (repeats.size == 0) {
        return mkBvRepeatExprNoSimplify(repeatNumber, value)
    }

    return rewriteFlatBvConcatExpr(repeats)
}

inline fun <T : KBvSort> KContext.simplifyBvZeroExtensionExprLight(
    extensionSize: Int,
    value: KExpr<T>,
    cont: (Int, KExpr<T>) -> KExpr<KBvSort>
): KExpr<KBvSort> {
    if (extensionSize == 0) {
        return value.uncheckedCast()
    }

    if (value is KBitVecValue<T>) {
        return value.zeroExtension(extensionSize.toUInt()).uncheckedCast()
    }

    return cont(extensionSize, value)
}

/** (zeroext a) ==> (concat 0 a) */
inline fun <T : KBvSort> KContext.rewriteBvZeroExtensionExpr(
    extensionSize: Int,
    value: KExpr<T>,
    rewriteBvConcatExpr: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBvSort>,
): KExpr<KBvSort> = rewriteBvConcatExpr(bvZero(extensionSize.toUInt()).uncheckedCast(), value.uncheckedCast())

inline fun <T : KBvSort> KContext.simplifyBvSignExtensionExprLight(
    extensionSize: Int,
    value: KExpr<T>,
    cont: (Int, KExpr<T>) -> KExpr<KBvSort>
): KExpr<KBvSort> {
    if (extensionSize == 0) {
        return value.uncheckedCast()
    }

    if (value is KBitVecValue<T>) {
        return value.signExtension(extensionSize.toUInt()).uncheckedCast()
    }

    return cont(extensionSize, value)
}

inline fun <T : KBvSort> KContext.simplifyBvExtractExprLight(
    high: Int,
    low: Int,
    value: KExpr<T>,
    cont: (Int, Int, KExpr<T>) -> KExpr<KBvSort>
): KExpr<KBvSort> {
    // (extract [size-1:0] x) ==> x
    if (low == 0 && high == value.sort.sizeBits.toInt() - 1) {
        return value.uncheckedCast()
    }

    if (value is KBitVecValue<T>) {
        return value.extractBv(high, low).uncheckedCast()
    }

    return cont(high, low, value)
}

/** (extract[high:low] (extract[_:nestedLow] x)) ==> (extract[high+nestedLow : low+nestedLow] x) */
inline fun <T : KBvSort> KContext.simplifyBvExtractExprNestedExtract(
    high: Int,
    low: Int,
    value: KExpr<T>,
    rewriteBvExtractExpr: KContext.(Int, Int, KExpr<KBvSort>) -> KExpr<KBvSort>,
    cont: (Int, Int, KExpr<T>) -> KExpr<KBvSort>
): KExpr<KBvSort> = if (value is KBvExtractExpr) {
    val nestedLow = value.low
    rewriteBvExtractExpr(
        high + nestedLow,
        low + nestedLow,
        value.value
    )
} else {
    cont(high, low, value)
}

@Suppress("LongParameterList")
inline fun <T : KBvSort> KContext.simplifyBvExtractExprTryRewrite(
    high: Int,
    low: Int,
    value: KExpr<T>,
    rewriteBvExtractExpr: KContext.(Int, Int, KExpr<KBvSort>) -> KExpr<KBvSort>,
    rewriteFlatBvConcatExpr: KContext.(List<KExpr<KBvSort>>) -> KExpr<KBvSort>,
    rewriteBvNotExpr: KContext.(KExpr<KBvSort>) -> KExpr<KBvSort>,
    rewriteBvOrExpr: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBvSort>,
    rewriteBvAndExpr: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBvSort>,
    rewriteBvXorExpr: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBvSort>,
    rewriteBvAddExpr: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBvSort>,
    rewriteBvMulExpr: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBvSort>,
    cont: (Int, Int, KExpr<T>) -> KExpr<KBvSort>
): KExpr<KBvSort> =
    when {
        // (extract (concat a b)) ==> (concat (extract a) (extract b))
        value is KBvConcatExpr -> {
            distributeExtractOverConcat(
                high = high,
                low = low,
                concatenation = value,
                rewriteBvExtractExpr = { h, l, v -> rewriteBvExtractExpr(h, l, v) },
                rewriteFlatBvConcatExpr = { args -> rewriteFlatBvConcatExpr(args) }
            )
        }
        // (extract [h:l] (bvnot x)) ==> (bvnot (extract [h:l] x))
        value is KBvNotExpr<*> -> {
            rewriteBvNotExpr(rewriteBvExtractExpr(high, low, value.value.uncheckedCast()))
        }
        // (extract [h:l] (bvor a b)) ==> (bvor (extract [h:l] a) (extract [h:l] b))
        value is KBvOrExpr<*> -> {
            val lhs = rewriteBvExtractExpr(high, low, value.arg0.uncheckedCast())
            val rhs = rewriteBvExtractExpr(high, low, value.arg1.uncheckedCast())
            rewriteBvOrExpr(lhs, rhs)
        }
        // (extract [h:l] (bvand a b)) ==> (bvand (extract [h:l] a) (extract [h:l] b))
        value is KBvAndExpr<*> -> {
            val lhs = rewriteBvExtractExpr(high, low, value.arg0.uncheckedCast())
            val rhs = rewriteBvExtractExpr(high, low, value.arg1.uncheckedCast())
            rewriteBvAndExpr(lhs, rhs)
        }
        // (extract [h:l] (bvxor a b)) ==> (bvxor (extract [h:l] a) (extract [h:l] b))
        value is KBvXorExpr<*> -> {
            val lhs = rewriteBvExtractExpr(high, low, value.arg0.uncheckedCast())
            val rhs = rewriteBvExtractExpr(high, low, value.arg1.uncheckedCast())
            rewriteBvXorExpr(lhs, rhs)
        }
        // (extract [h:0] (bvadd a b)) ==> (bvadd (extract [h:0] a) (extract [h:0] b))
        value is KBvAddExpr<*> && low == 0 -> {
            val lhs = rewriteBvExtractExpr(high, 0, value.arg0.uncheckedCast())
            val rhs = rewriteBvExtractExpr(high, 0, value.arg1.uncheckedCast())
            rewriteBvAddExpr(lhs, rhs)
        }
        // (extract [h:0] (bvmul a b)) ==> (bvmul (extract [h:0] a) (extract [h:0] b))
        value is KBvMulExpr<*> && low == 0 -> {
            val lhs = rewriteBvExtractExpr(high, 0, value.arg0.uncheckedCast())
            val rhs = rewriteBvExtractExpr(high, 0, value.arg1.uncheckedCast())
            rewriteBvMulExpr(lhs, rhs)
        }

        else -> cont(high, low, value)
    }

/** eval constants */
inline fun <T : KBvSort, S : KBvSort> KContext.simplifyBvConcatExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<S>,
    cont: (KExpr<T>, KExpr<S>) -> KExpr<KBvSort>
): KExpr<KBvSort> = evalBvOperation(lhs, rhs, { a, b -> BvUtils.concatBv(a, b) }) {
    cont(lhs, rhs)
}

/**
 * (concat const1 (concat const2 a)) ==> (concat (concat const1 const2) a)
 * (concat (concat a const1) const2) ==> (concat a (concat const1 const2))
 * */
inline fun <T : KBvSort, S : KBvSort> KContext.simplifyBvConcatExprNestedConcat(
    lhs: KExpr<T>,
    rhs: KExpr<S>,
    rewriteBvConcatExpr: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBvSort>,
    cont: (KExpr<T>, KExpr<S>) -> KExpr<KBvSort>
): KExpr<KBvSort> {
    // (concat const1 (concat const2 a)) ==> (concat (concat const1 const2) a)
    if (lhs is KBitVecValue<T> && rhs is KBvConcatExpr) {
        val rhsLeft = rhs.arg0
        if (rhsLeft is KBitVecValue<*>) {
            return rewriteBvConcatExpr(BvUtils.concatBv(lhs, rhsLeft).uncheckedCast(), rhs.arg1)
        }
    }
    // (concat (concat a const1) const2) ==> (concat a (concat const1 const2))
    if (rhs is KBitVecValue<S> && lhs is KBvConcatExpr) {
        val lhsRight = lhs.arg1
        if (lhsRight is KBitVecValue<*>) {
            return rewriteBvConcatExpr(lhs.arg0, BvUtils.concatBv(lhsRight, rhs).uncheckedCast())
        }
    }

    return cont(lhs, rhs)
}

@Suppress("LoopWithTooManyJumpStatements")
inline fun KContext.simplifyFlatBvConcatExpr(
    args: List<KExpr<KBvSort>>,
    rewriteBvExtractExpr: KContext.(Int, Int, KExpr<KBvSort>) -> KExpr<KBvSort>,
    rewriteFlatBvConcatExpr: KContext.(List<KExpr<KBvSort>>) -> KExpr<KBvSort>,
    cont: (List<KExpr<KBvSort>>) -> KExpr<KBvSort>
): KExpr<KBvSort> {
    val mergedParts = arrayListOf(args.first())
    var hasSimplifierAuxTerms = false

    for (part in args.drop(1)) {
        val lastPart = mergedParts.last()

        // (concat (concat a const1) (concat const2 b)) ==> (concat a (concat (concat const1 const2) b))
        if (lastPart is KBitVecValue<*> && part is KBitVecValue<*>) {
            mergedParts.removeLast()
            mergedParts.add(BvUtils.concatBv(lastPart, part).cast())
            continue
        }

        // (concat (extract[h1, l1] a) (extract[h2, l2] a)), l1 == h2 + 1 ==> (extract[h1, l2] a)
        if (lastPart is KBvExtractExpr && part is KBvExtractExpr) {
            val possiblyMerged = tryMergeBvConcatExtract(lastPart, part, rewriteBvExtractExpr)
            if (possiblyMerged != null) {
                mergedParts.removeLast()
                mergedParts.add(possiblyMerged)
                hasSimplifierAuxTerms = true
                continue
            }
        }
        mergedParts.add(part)
    }

    return if (hasSimplifierAuxTerms) {
        rewriteFlatBvConcatExpr(mergedParts)
    } else {
        cont(mergedParts)
    }
}

// (concat (extract[h1, l1] a) (extract[h2, l2] a)), l1 == h2 + 1 ==> (extract[h1, l2] a)
inline fun KContext.tryMergeBvConcatExtract(
    lhs: KBvExtractExpr,
    rhs: KBvExtractExpr,
    rewriteBvExtractExpr: KContext.(Int, Int, KExpr<KBvSort>) -> KExpr<KBvSort>
): KExpr<KBvSort>? {
    if (lhs.value != rhs.value || lhs.low != rhs.high + 1) {
        return null
    }
    return rewriteBvExtractExpr(lhs.high, rhs.low, lhs.value)
}


/** (sgt a b) ==> (not (sle a b)) */
inline fun <T : KBvSort> KContext.rewriteBvSignedGreaterExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvSignedLessOrEqualExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>,
    rewriteNot: KContext.(KExpr<KBoolSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = rewriteNot(rewriteBvSignedLessOrEqualExpr(lhs, rhs))

/** (sge a b) ==> (sle b a) */
inline fun <T : KBvSort> KContext.rewriteBvSignedGreaterOrEqualExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvSignedLessOrEqualExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>,
): KExpr<KBoolSort> = rewriteBvSignedLessOrEqualExpr(rhs, lhs)

/** (slt a b) ==> (not (sle b a)) */
inline fun <T : KBvSort> KContext.rewriteBvSignedLessExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvSignedLessOrEqualExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>,
    rewriteNot: KContext.(KExpr<KBoolSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = rewriteNot(rewriteBvSignedLessOrEqualExpr(rhs, lhs))

inline fun <T : KBvSort> KContext.simplifyBvSignedLessOrEqualExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvEq: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = bvLessOrEqual(
    lhs = lhs,
    rhs = rhs,
    signed = true,
    rewriteBvEq = rewriteBvEq,
    cont = cont
)

/** (ugt a b) ==> (not (ule a b)) */
inline fun <T : KBvSort> KContext.rewriteBvUnsignedGreaterExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvUnsignedLessOrEqualExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>,
    rewriteNot: KContext.(KExpr<KBoolSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = rewriteNot(rewriteBvUnsignedLessOrEqualExpr(lhs, rhs))

/** (uge a b) ==> (ule b a) */
inline fun <T : KBvSort> KContext.rewriteBvUnsignedGreaterOrEqualExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvUnsignedLessOrEqualExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>,
): KExpr<KBoolSort> = rewriteBvUnsignedLessOrEqualExpr(rhs, lhs)

/** (ult a b) ==> (not (ule b a)) */
inline fun <T : KBvSort> KContext.rewriteBvUnsignedLessExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvUnsignedLessOrEqualExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>,
    rewriteNot: KContext.(KExpr<KBoolSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = rewriteNot(rewriteBvUnsignedLessOrEqualExpr(rhs, lhs))

inline fun <T : KBvSort> KContext.rewriteBvUnsignedLessOrEqualExprLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvEq: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = bvLessOrEqual(
    lhs = lhs,
    rhs = rhs,
    signed = false,
    rewriteBvEq = rewriteBvEq,
    cont = cont
)

inline fun <T : KBvSort> KContext.simplifyBv2IntExprLight(
    value: KExpr<T>,
    isSigned: Boolean,
    cont: (KExpr<T>, Boolean) -> KExpr<KIntSort>
): KExpr<KIntSort> = if (value is KBitVecValue<T>) {
    val integerValue = if (isSigned) {
        value.toBigIntegerSigned()
    } else {
        value.toBigIntegerUnsigned()
    }
    mkIntNum(integerValue)
} else {
    cont(value, isSigned)
}

@Suppress("LongParameterList")
inline fun <T : KBvSort> KContext.rewriteBvAddNoOverflowExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    isSigned: Boolean,
    rewriteBvSignedLessExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort> = KContext::simplifyBvSignedLessExpr,
    rewriteBvAddExpr: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBvSort> = KContext::simplifyBvAddExpr,
    rewriteBvZeroExtensionExpr: KContext.(Int, KExpr<T>) -> KExpr<KBvSort> = KContext::simplifyBvZeroExtensionExpr,
    rewriteBvExtractExpr: KContext.(Int, Int, KExpr<T>) -> KExpr<KBvSort> = KContext::simplifyBvExtractExpr,
    rewriteBvEq: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBoolSort> = { l, r -> simplifyEq(l, r) },
    rewriteAnd: KContext.(KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort> = { l, r -> simplifyAnd(l, r) },
    rewriteImplies: KContext.(KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort> = KContext::simplifyImplies
): KExpr<KBoolSort> {
    if (isSigned) {
        /**
         * (bvadd no ovf signed a b) ==>
         *    (=> (and (bvslt 0 a) (bvslt 0 b)) (bvslt 0 (bvadd a b)))
         * */

        val zero: KExpr<T> = bvZero(lhs.sort.sizeBits).uncheckedCast()
        val zeroSltA = rewriteBvSignedLessExpr(zero, lhs)
        val zeroSltB = rewriteBvSignedLessExpr(zero, rhs)
        val sum = rewriteBvAddExpr(lhs.uncheckedCast(), rhs.uncheckedCast())
        val zeroSltSum = rewriteBvSignedLessExpr(zero, sum.uncheckedCast())

        return rewriteImplies(rewriteAnd(zeroSltA, zeroSltB), zeroSltSum)
    } else {
        /**
         * (bvadd no ovf unsigned a b) ==>
         *    (= 0 (extract [highestBit] (bvadd (concat 0 a) (concat 0 b))))
         * */

        val extA = rewriteBvZeroExtensionExpr(1, lhs)
        val extB = rewriteBvZeroExtensionExpr(1, rhs)
        val sum = rewriteBvAddExpr(extA, extB)
        val highestBitIdx = sum.sort.sizeBits.toInt() - 1
        val sumFirstBit = rewriteBvExtractExpr(highestBitIdx, highestBitIdx, sum.uncheckedCast())

        return rewriteBvEq(sumFirstBit, mkBv(false).uncheckedCast())
    }
}

@Suppress("LongParameterList")
inline fun <T : KBvSort> KContext.rewriteBvAddNoUnderflowExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvSignedLessExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort> = KContext::simplifyBvSignedLessExpr,
    rewriteBvAddExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<T> = KContext::simplifyBvAddExpr,
    rewriteAnd: KContext.(KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort> = { l, r -> simplifyAnd(l, r) },
    rewriteImplies: KContext.(KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort> = KContext::simplifyImplies
): KExpr<KBoolSort> {
    /**
     * (bvadd no udf a b) ==>
     *    (=> (and (bvslt a 0) (bvslt b 0)) (bvslt (bvadd a b) 0))
     * */
    val zero: KExpr<T> = bvZero(lhs.sort.sizeBits).uncheckedCast()
    val aLtZero = rewriteBvSignedLessExpr(lhs, zero)
    val bLtZero = rewriteBvSignedLessExpr(rhs, zero)
    val sum = rewriteBvAddExpr(lhs, rhs)
    val sumLtZero = rewriteBvSignedLessExpr(sum, zero)

    return rewriteImplies(rewriteAnd(aLtZero, bLtZero), sumLtZero)
}

inline fun <T : KBvSort> KContext.rewriteBvMulNoOverflowExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    isSigned: Boolean,
    cont: (KExpr<T>, KExpr<T>, Boolean) -> KExpr<KBoolSort>
): KExpr<KBoolSort> {
    val simplified = if (isSigned) {
        trySimplifyBvSignedMulNoOverflow(lhs, rhs, isOverflow = true)
    } else {
        trySimplifyBvUnsignedMulNoOverflow(lhs, rhs)
    }
    return simplified ?: cont(lhs, rhs, isSigned)
}

inline fun <T : KBvSort> KContext.rewriteBvMulNoUnderflowExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> =
    trySimplifyBvSignedMulNoOverflow(lhs, rhs, isOverflow = false) ?: cont(lhs, rhs)

inline fun <T : KBvSort> KContext.rewriteBvNegNoOverflowExpr(
    arg: KExpr<T>,
    rewriteBvEq: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBoolSort> = { l, r -> simplifyEq(l, r) },
    rewriteNot: KContext.(KExpr<KBoolSort>) -> KExpr<KBoolSort> = KContext::simplifyNot
): KExpr<KBoolSort> {
    /**
     * (bvneg no ovf a) ==> (not (= a MIN_VALUE))
     * */
    val minValue = bvMinValueSigned(arg.sort.sizeBits)
    return rewriteNot(rewriteBvEq(arg.uncheckedCast(), minValue.uncheckedCast()))
}

inline fun <T : KBvSort> KContext.rewriteBvDivNoOverflowExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvEq: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBoolSort> = { l, r -> simplifyEq(l, r) },
    rewriteAnd: KContext.(KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort> = { l, r -> simplifyAnd(l, r) },
    rewriteNot: KContext.(KExpr<KBoolSort>) -> KExpr<KBoolSort> = KContext::simplifyNot
): KExpr<KBoolSort> {
    /**
     * (bvsdiv no ovf a b) ==>
     *     (not (and (= a MSB) (= b -1)))
     * */
    val size = lhs.sort.sizeBits
    val mostSignificantBit = bvMinValueSigned(size)
    val minusOne = bvValue(size, -1)

    val aIsMsb = rewriteBvEq(lhs.uncheckedCast(), mostSignificantBit.uncheckedCast())
    val bIsMinusOne = rewriteBvEq(rhs.uncheckedCast(), minusOne.uncheckedCast())
    return rewriteNot(rewriteAnd(aIsMsb, bIsMinusOne))
}

@Suppress("LongParameterList")
inline fun <T : KBvSort> KContext.rewriteBvSubNoOverflowExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteBvSignedLessExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort> = KContext::simplifyBvSignedLessExpr,
    rewriteBvAddNoOverflowExpr: (KExpr<T>, KExpr<T>, Boolean) -> KExpr<KBoolSort> =
        { l, r, isSigned -> simplifyBvAddNoOverflowExpr(l, r, isSigned) },
    rewriteBvEq: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBoolSort> = { l, r -> simplifyEq(l, r) },
    rewriteBvNegationExpr: KContext.(KExpr<T>) -> KExpr<T> = KContext::simplifyBvNegationExpr,
    rewriteIte: (KExpr<KBoolSort>, KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort> =
        { c, t, f -> simplifyIte(c, t, f) }
): KExpr<KBoolSort> {
    /**
     * (bvsub no ovf a b) ==>
     *     (ite (= b MIN_VALUE) (bvslt a 0) (bvadd no ovf signed a (bvneg b)))
     * */

    val zero: KExpr<T> = bvZero(lhs.sort.sizeBits).uncheckedCast()
    val minValue: KExpr<T> = bvMinValueSigned(lhs.sort.sizeBits).uncheckedCast()

    val minusB = rewriteBvNegationExpr(rhs)
    val bIsMin = rewriteBvEq(rhs.uncheckedCast(), minValue.uncheckedCast())
    val aLtZero = rewriteBvSignedLessExpr(lhs, zero)
    val noOverflow = rewriteBvAddNoOverflowExpr(lhs, minusB, true)

    return rewriteIte(bIsMin, aLtZero, noOverflow)
}

@Suppress("LongParameterList")
inline fun <T : KBvSort> KContext.rewriteBvSubNoUnderflowExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    isSigned: Boolean,
    rewriteBvSignedLessExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort> = KContext::simplifyBvSignedLessExpr,
    rewriteBvUnsignedLessOrEqualExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort> =
        KContext::simplifyBvUnsignedLessOrEqualExpr,
    createBvAddNoUnderflowExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort> =
        KContext::simplifyBvAddNoUnderflowExpr,
    rewriteBvNegationExpr: KContext.(KExpr<T>) -> KExpr<T> = KContext::simplifyBvNegationExpr,
    rewriteImplies: KContext.(KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort> = KContext::simplifyImplies
): KExpr<KBoolSort> {
    if (isSigned) {
        /**
         * (bvsub no udf signed a b) ==>
         *    (=> (bvslt 0 b) (bvadd no udf (bvneg b)))
         * */
        val zero: KExpr<T> = bvZero(lhs.sort.sizeBits).uncheckedCast()
        val minusB = rewriteBvNegationExpr(rhs)
        val zeroLtB = rewriteBvSignedLessExpr(zero, rhs)
        val noOverflow = createBvAddNoUnderflowExpr(lhs, minusB)

        return rewriteImplies(zeroLtB, noOverflow)
    } else {
        /**
         * (bvsub no udf unsigned a b) ==>
         *    (bvule b a)
         * */
        return rewriteBvUnsignedLessOrEqualExpr(rhs, lhs)
    }
}

inline fun <T : KBvSort> KContext.distributeOperationOverConcat(
    args: List<KExpr<T>>,
    rewriteBvExtractExpr: KContext.(Int, Int, KExpr<KBvSort>) -> KExpr<KBvSort>,
    rewriteFlatBvOpExpr: KContext.(List<KExpr<KBvSort>>) -> KExpr<KBvSort>,
    rewriteBvConcatExpr: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBvSort>
): KExpr<T> {
    val firstConcat = args.first { it is KBvConcatExpr } as KBvConcatExpr
    val size = firstConcat.sort.sizeBits.toInt()
    val partSize = firstConcat.arg0.sort.sizeBits.toInt()

    val args1 = arrayListOf<KExpr<KBvSort>>()
    val args2 = arrayListOf<KExpr<KBvSort>>()

    for (expr in args) {
        args1 += rewriteBvExtractExpr(size - 1, size - partSize, expr.uncheckedCast())

        args2 += rewriteBvExtractExpr(size - partSize - 1, 0, expr.uncheckedCast())
    }

    val mergedArgs1 = rewriteFlatBvOpExpr(args1)
    val mergedArgs2 = rewriteFlatBvOpExpr(args2)

    return rewriteBvConcatExpr(mergedArgs1, mergedArgs2).uncheckedCast()
}

@Suppress("LongParameterList", "LoopWithTooManyJumpStatements")
inline fun <T : KBvSort> KContext.simplifyBvAndOr(
    args: List<KExpr<T>>,
    neutralElement: KBitVecValue<T>,
    isZeroElement: KContext.(KBitVecValue<T>) -> Boolean,
    buildZeroElement: KContext.() -> KBitVecValue<T>,
    operation: (KBitVecValue<T>, KBitVecValue<T>) -> KBitVecValue<T>,
    cont: (List<KExpr<T>>) -> KExpr<T>
): KExpr<T> {
    var constantValue = neutralElement
    val resultParts = arrayListOf<KExpr<T>>()
    val positiveTerms = hashSetOf<KExpr<T>>()
    val negativeTerms = hashSetOf<KExpr<T>>()

    for (arg in args) {
        if (arg is KBitVecValue<T>) {
            constantValue = operation(constantValue, arg)
            continue
        }

        if (arg is KBvNotExpr<T>) {
            val term = arg.value
            // (bvop (bvnot a) b (bvnot a)) ==> (bvop (bvnot a) b)
            if (!negativeTerms.add(term)) {
                continue
            }

            // (bvop a (bvnot a)) ==> zero
            if (term in positiveTerms) {
                return buildZeroElement()
            }
        } else {
            // (bvop a b a) ==> (bvop a b)
            if (!positiveTerms.add(arg)) {
                continue
            }

            // (bvop a (bvnot a)) ==> zero
            if (arg in negativeTerms) {
                return buildZeroElement()
            }
        }

        resultParts += arg
    }

    // (bvop zero a) ==> zero
    if (isZeroElement(constantValue)) {
        return constantValue
    }

    if (resultParts.isEmpty()) {
        return constantValue.uncheckedCast()
    }

    // (bvop neutral a) ==> a
    if (constantValue != neutralElement) {
        resultParts.add(constantValue.uncheckedCast())
    }

    return cont(resultParts)
}

/**
 * (= (concat a b) c) ==>
 *  (and
 *      (= a (extract (0, <a_size>) c))
 *      (= b (extract (<a_size>, <a_size> + <b_size>) c))
 *  )
 * */
inline fun <T : KBvSort> KContext.rewriteBvConcatEq(
    l: KExpr<T>,
    r: KExpr<T>,
    rewriteBvExtractExpr: KContext.(Int, Int, KExpr<KBvSort>) -> KExpr<KBvSort>,
    rewriteBvEq: KContext.(KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBoolSort>,
    rewriteFlatAnd: KContext.(List<KExpr<KBoolSort>>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> {
    val lArgs = if (l is KBvConcatExpr) flatConcatArgs(l) else listOf(l)
    val rArgs = if (r is KBvConcatExpr) flatConcatArgs(r) else listOf(r)
    val result = arrayListOf<KExpr<KBoolSort>>()
    var lowL = 0
    var lowR = 0
    var lIdx = lArgs.size
    var rIdx = rArgs.size
    while (lIdx > 0 && rIdx > 0) {
        val lArg = lArgs[lIdx - 1]
        val rArg = rArgs[rIdx - 1]
        val lSize = lArg.sort.sizeBits.toInt()
        val rSize = rArg.sort.sizeBits.toInt()
        val remainSizeL = lSize - lowL
        val remainSizeR = rSize - lowR
        when {
            remainSizeL == remainSizeR -> {
                val newL = rewriteBvExtractExpr(lSize - 1, lowL, lArg.uncheckedCast())
                val newR = rewriteBvExtractExpr(rSize - 1, lowR, rArg.uncheckedCast())

                result += rewriteBvEq(newL, newR)
                lowL = 0
                lowR = 0
                lIdx--
                rIdx--
            }

            remainSizeL < remainSizeR -> {
                val newL = rewriteBvExtractExpr(lSize - 1, lowL, lArg.uncheckedCast())
                val newR = rewriteBvExtractExpr(remainSizeL + lowR - 1, lowR, rArg.uncheckedCast())
                result += rewriteBvEq(newL, newR)
                lowL = 0
                lowR += remainSizeL
                lIdx--
            }

            else -> {
                val newL = rewriteBvExtractExpr(remainSizeR + lowL - 1, lowL, lArg.uncheckedCast())
                val newR = rewriteBvExtractExpr(rSize - 1, lowR, rArg.uncheckedCast())
                result += rewriteBvEq(newL, newR)
                lowL += remainSizeR
                lowR = 0
                rIdx--
            }
        }
    }

    // restore concat order
    result.reverse()

    return rewriteFlatAnd(result)
}

fun flatConcatArgs(expr: KBvConcatExpr): List<KExpr<KBvSort>> {
    return flatBinaryBvExpr<KBvConcatExpr>(
        expr as KExpr<KBvSort>,
        getLhs = { it.arg0 },
        getRhs = { it.arg1 }
    )
}

inline fun <reified T> flatBinaryBvExpr(
    initial: KExpr<KBvSort>,
    getLhs: (T) -> KExpr<KBvSort>,
    getRhs: (T) -> KExpr<KBvSort>
): List<KExpr<KBvSort>> {
    val flatten = arrayListOf<KExpr<KBvSort>>()
    val unprocessed = arrayListOf<KExpr<KBvSort>>()
    unprocessed += initial
    while (unprocessed.isNotEmpty()) {
        val e = unprocessed.removeLast()
        if (e !is T) {
            flatten += e
            continue
        }
        unprocessed += getRhs(e)
        unprocessed += getLhs(e)
    }
    return flatten
}

/**
 * (extract (concat a b)) ==> (concat (extract a) (extract b))
 * */
@Suppress("LoopWithTooManyJumpStatements", "NestedBlockDepth", "ComplexMethod")
inline fun KContext.distributeExtractOverConcat(
    high: Int,
    low: Int,
    concatenation: KBvConcatExpr,
    rewriteBvExtractExpr: KContext.(Int, Int, KExpr<KBvSort>) -> KExpr<KBvSort>,
    rewriteFlatBvConcatExpr: KContext.(List<KExpr<KBvSort>>) -> KExpr<KBvSort>
): KExpr<KBvSort> {
    val parts = flatConcatArgs(concatenation)

    var idx = concatenation.sort.sizeBits.toInt()
    var firstPartIdx = 0

    // find first part to extract from
    do {
        val firstPart = parts[firstPartIdx]
        val firstPartSize = firstPart.sort.sizeBits.toInt()
        idx -= firstPartSize

        // before first part
        if (idx > high) {
            firstPartIdx++
            continue
        }

        // extract from a single part
        if (idx <= low) {
            return if (idx == low && high - idx == firstPartSize) {
                firstPart
            } else {
                rewriteBvExtractExpr(high - idx, low - idx, firstPart)
            }
        }

        /**
         * idx <= high && idx > low
         * extract from multiple parts starting from firstPartIdx
         * */
        break

    } while (firstPartIdx < parts.size)


    // extract from multiple parts
    val partsToExtractFrom = arrayListOf<KExpr<KBvSort>>()
    val firstPart = parts[firstPartIdx]
    val firstPartSize = firstPart.sort.sizeBits.toInt()

    if (high - idx == firstPartSize - 1) {
        partsToExtractFrom += firstPart
    } else {
        partsToExtractFrom += rewriteBvExtractExpr(high - idx, 0, firstPart)
    }

    for (partIdx in firstPartIdx + 1 until parts.size) {
        val part = parts[partIdx]
        val partSize = part.sort.sizeBits.toInt()
        idx -= partSize

        when {
            idx > low -> {
                // not a last part
                partsToExtractFrom += part
                continue
            }

            idx == low -> {
                partsToExtractFrom += part
                break
            }

            else -> {
                partsToExtractFrom += rewriteBvExtractExpr(partSize - 1, low - idx, part)
                break
            }
        }
    }

    return rewriteFlatBvConcatExpr(partsToExtractFrom)
}

inline fun <T : KBvSort, S : KBvSort> evalBvOperation(
    lhs: KExpr<T>,
    rhs: KExpr<S>,
    operation: (KBitVecValue<T>, KBitVecValue<S>) -> KBitVecValue<*>,
    cont: () -> KExpr<KBvSort>
): KExpr<KBvSort> = if (lhs is KBitVecValue<T> && rhs is KBitVecValue<S>) {
    operation(lhs, rhs).uncheckedCast()
} else {
    cont()
}

inline fun <T : KBvSort> executeIfValue(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    body: (KBitVecValue<T>, KExpr<T>) -> Unit
) {
    if (lhs is KBitVecValue<T>) {
        body(lhs, rhs)
    } else if (rhs is KBitVecValue<T>) {
        body(rhs, lhs)
    }
}

inline fun <T : KBvSort> KContext.rotateLeft(
    arg: KExpr<T>,
    size: Int,
    rotationNumber: Int,
    rewriteBvExtractExpr: KContext.(Int, Int, KExpr<T>) -> KExpr<KBvSort>,
    rewriteBvConcatExpr: KContext.(KExpr<T>, KExpr<KBvSort>) -> KExpr<KBvSort>
): KExpr<T> {
    val rotation = rotationNumber.mod(size)

    if (rotation == 0 || size == 1) {
        return arg
    }

    val lhs = rewriteBvExtractExpr(size - rotation - 1, 0, arg)
    val rhs = rewriteBvExtractExpr(size - 1, size - rotation, arg)
    return rewriteBvConcatExpr(lhs.uncheckedCast(), rhs.uncheckedCast()).uncheckedCast()
}

fun <T : KBvSort> KContext.trySimplifyBvSignedMulNoOverflow(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    isOverflow: Boolean
): KExpr<KBoolSort>? = when {
    lhs is KBitVecValue<T> && lhs.isBvZero() -> trueExpr
    lhs is KBitVecValue<T> && lhs.sort.sizeBits != 1u && lhs.isBvOne() -> trueExpr
    rhs is KBitVecValue<T> && rhs.isBvZero() -> trueExpr
    rhs is KBitVecValue<T> && rhs.sort.sizeBits != 1u && rhs.isBvOne() -> trueExpr
    lhs is KBitVecValue<T> && rhs is KBitVecValue<T> -> evalBvSignedMulNoOverflow(lhs, rhs, isOverflow)
    else -> null
}
fun <T : KBvSort> KContext.evalBvSignedMulNoOverflow(
    lhsValue: KBitVecValue<T>,
    rhsValue: KBitVecValue<T>,
    isOverflow: Boolean
): KExpr<KBoolSort> {
    val size = lhsValue.sort.sizeBits
    val one = bvOne(size)

    val lhsSign = lhsValue.signBit()
    val rhsSign = rhsValue.signBit()

    val operationOverflow = when {
        // lhs < 0 && rhs < 0
        lhsSign && rhsSign -> {
            // no underflow possible
            if (!isOverflow) return trueExpr
            // overflow if rhs <= (MAX_VALUE / lhs - 1)
            val maxValue = bvMaxValueSigned(size)
            val limit = maxValue.signedDivide(lhsValue)
            rhsValue.signedLessOrEqual(limit - one)
        }
        // lhs > 0 && rhs > 0
        !lhsSign && !rhsSign -> {
            // no underflow possible
            if (!isOverflow) return trueExpr
            // overflow if MAX_VALUE / rhs <= lhs - 1
            val maxValue = bvMaxValueSigned(size)
            val limit = maxValue.signedDivide(rhsValue)
            limit.signedLessOrEqual(lhsValue - one)
        }
        // lhs < 0 && rhs > 0
        lhsSign && !rhsSign -> {
            // no overflow possible
            if (isOverflow) return trueExpr
            // underflow if lhs <= MIN_VALUE / rhs - 1
            val minValue = bvMinValueSigned(size)
            val limit = minValue.signedDivide(rhsValue)
            lhsValue.signedLessOrEqual(limit - one)
        }
        // lhs > 0 && rhs < 0
        else -> {
            // no overflow possible
            if (isOverflow) return trueExpr
            // underflow if rhs <= MIN_VALUE / lhs - 1
            val minValue = bvMinValueSigned(size)
            val limit = minValue.signedDivide(lhsValue)
            rhsValue.signedLessOrEqual(limit - one)
        }
    }

    return (!operationOverflow).expr
}

fun <T : KBvSort> KContext.trySimplifyBvUnsignedMulNoOverflow(
    lhs: KExpr<T>,
    rhs: KExpr<T>
): KExpr<KBoolSort>? {
    val size = lhs.sort.sizeBits
    val lhsValue = lhs as? KBitVecValue<T>
    val rhsValue = rhs as? KBitVecValue<T>

    if (lhsValue != null && (lhsValue.isBvZero() || (lhsValue.isBvOne()))) {
        return trueExpr
    }

    if (rhsValue != null && (rhsValue.isBvZero() || (rhsValue.isBvOne()))) {
        return trueExpr
    }

    if (lhsValue != null && rhsValue != null) {
        val longLhs = lhsValue.zeroExtension(size)
        val longRhs = rhsValue.zeroExtension(size)
        val longMaxValue = bvMaxValueUnsigned(size).zeroExtension(size)

        val product = longLhs * longRhs
        return product.unsignedLessOrEqual(longMaxValue).expr
    }

    return null
}

inline fun <T : KBvSort> KContext.bvLessOrEqual(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    signed: Boolean,
    rewriteBvEq: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> {
    if (lhs == rhs) return trueExpr

    val lhsValue = lhs as? KBitVecValue<T>
    val rhsValue = rhs as? KBitVecValue<T>

    if (lhsValue != null && rhsValue != null) {
        val result = if (signed) {
            lhsValue.signedLessOrEqual(rhsValue)
        } else {
            lhsValue.unsignedLessOrEqual(rhsValue)
        }
        return result.expr
    }

    if (lhsValue != null || rhsValue != null) {

        if (rhsValue != null) {
            // a <= b, b == MIN_VALUE ==> a == b
            if (rhsValue.isMinValue(signed)) {
                return rewriteBvEq(lhs, rhs)
            }
            // a <= b, b == MAX_VALUE ==> true
            if (rhsValue.isMaxValue(signed)) {
                return trueExpr
            }
        }

        if (lhsValue != null) {
            // a <= b, a == MIN_VALUE ==> true
            if (lhsValue.isMinValue(signed)) {
                return trueExpr
            }
            // a <= b, a == MAX_VALUE ==> a == b
            if (lhsValue.isMaxValue(signed)) {
                return rewriteBvEq(lhs, rhs)
            }
        }
    }

    return cont(lhs, rhs)
}

fun KBitVecValue<*>.isMinValue(signed: Boolean) =
    if (signed) isBvMinValueSigned() else isBvZero()

fun KBitVecValue<*>.isMaxValue(signed: Boolean) =
    if (signed) isBvMaxValueSigned() else isBvMaxValueUnsigned()
