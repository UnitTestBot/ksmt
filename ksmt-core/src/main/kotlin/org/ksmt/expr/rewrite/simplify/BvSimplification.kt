package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KBvAddExpr
import org.ksmt.expr.KBvMulExpr
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KIntSort
import org.ksmt.utils.BvUtils.bvValueIs
import org.ksmt.utils.BvUtils.bvZero
import org.ksmt.utils.BvUtils.isBvMaxValueSigned
import org.ksmt.utils.BvUtils.isBvMaxValueUnsigned
import org.ksmt.utils.BvUtils.isBvMinValueSigned
import org.ksmt.utils.BvUtils.isBvOne
import org.ksmt.utils.BvUtils.isBvZero
import org.ksmt.utils.BvUtils.plus
import org.ksmt.utils.BvUtils.powerOfTwoOrNull
import org.ksmt.utils.BvUtils.signedDivide
import org.ksmt.utils.BvUtils.signedLessOrEqual
import org.ksmt.utils.BvUtils.signedMod
import org.ksmt.utils.BvUtils.signedRem
import org.ksmt.utils.BvUtils.times
import org.ksmt.utils.BvUtils.unaryMinus
import org.ksmt.utils.BvUtils.unsignedDivide
import org.ksmt.utils.BvUtils.unsignedLessOrEqual
import org.ksmt.utils.BvUtils.unsignedRem
import org.ksmt.utils.uncheckedCast


fun <T : KBvSort> KContext.simplifyBvNotExpr(arg: KExpr<T>): KExpr<T> = mkBvNotExprNoSimplify(arg)

fun <T : KBvSort> KContext.simplifyBvOrExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> = mkBvOrExprNoSimplify(lhs, rhs)

fun <T : KBvSort> KContext.simplifyBvAndExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> = mkBvAndExprNoSimplify(lhs, rhs)

fun <T : KBvSort> KContext.simplifyBvNorExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> = mkBvNorExprNoSimplify(lhs, rhs)

fun <T : KBvSort> KContext.simplifyBvNAndExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> = mkBvNAndExprNoSimplify(lhs, rhs)

fun <T : KBvSort> KContext.simplifyBvXNorExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> = mkBvXNorExprNoSimplify(lhs, rhs)

fun <T : KBvSort> KContext.simplifyBvXorExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> = mkBvXorExprNoSimplify(lhs, rhs)


fun <T : KBvSort> KContext.simplifyBvNegationExpr(arg: KExpr<T>): KExpr<T> =
    if (arg is KBitVecValue<T>) (-arg).uncheckedCast() else mkBvNegationExprNoSimplify(arg)

fun <T : KBvSort> KContext.simplifyBvAddExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    evalBvOperationOr(lhs, rhs, { a, b -> a + b }) {
        // (bvadd x const) ==> (bvadd const x)
        preferLeftValue(lhs, rhs) { left, right ->
            if (left is KBitVecValue<T> && left.isBvZero()) {
                return right
            }

            // flat one level
            if (left is KBitVecValue<T> && right is KBvAddExpr<T>) {
                val rightLeft = right.arg0
                val rightRight = right.arg1

                if (rightLeft is KBitVecValue<T>) {
                    return simplifyBvAddExpr((left + rightLeft).uncheckedCast(), rightRight)
                }

                if (rightRight is KBitVecValue<T>) {
                    return simplifyBvAddExpr((left + rightRight).uncheckedCast(), rightLeft)
                }
            }

            mkBvAddExprNoSimplify(left, right)
        }
    }

fun <T : KBvSort> KContext.simplifyBvMulExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    evalBvOperationOr(lhs, rhs, { a, b -> a * b }) {
        // (bvmul x const) ==> (bvmul const x)
        preferLeftValue(lhs, rhs) { left, right ->
            if (left is KBitVecValue<T>) {
                // (* 0 a) ==> 0
                if (left.isBvZero()) {
                    return left
                }

                // (* 1 a) ==> a
                if (left.isBvOne()) {
                    return right
                }

                // (* -1 a) ==> -a
                if (left.bvValueIs(-1)) {
                    return simplifyBvNegationExpr(right)
                }

                // flat one level
                if (right is KBvMulExpr<T>) {
                    val rightLeft = right.arg0
                    val rightRight = right.arg1

                    if (rightLeft is KBitVecValue<T>) {
                        return simplifyBvMulExpr((left * rightLeft).uncheckedCast(), rightRight)
                    }

                    if (rightRight is KBitVecValue<T>) {
                        return simplifyBvMulExpr((left * rightRight).uncheckedCast(), rightLeft)
                    }
                }
            }

            mkBvMulExprNoSimplify(left, right)
        }
    }

fun <T : KBvSort> KContext.simplifyBvSubExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyBvAddExpr(lhs, simplifyBvNegationExpr(rhs))

fun <T : KBvSort> KContext.simplifyBvSignedDivExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> {
    if (rhs is KBitVecValue<T> && !rhs.isBvZero()) {
        if (lhs is KBitVecValue<T>) {
            return lhs.signedDivide(rhs).uncheckedCast()
        }

        if (rhs.isBvOne()) {
            return lhs
        }
    }
    return mkBvSignedDivExprNoSimplify(lhs, rhs)
}

fun <T : KBvSort> KContext.simplifyBvSignedModExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> {
    if (rhs is KBitVecValue<T> && !rhs.isBvZero()) {
        if (lhs is KBitVecValue<T>) {
            return lhs.signedMod(rhs).uncheckedCast()
        }

        if (rhs.isBvOne()) {
            return bvZero(rhs.sort.sizeBits).uncheckedCast()
        }
    }
    return mkBvSignedModExprNoSimplify(lhs, rhs)
}

fun <T : KBvSort> KContext.simplifyBvSignedRemExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> {
    if (rhs is KBitVecValue<T> && !rhs.isBvZero()) {
        if (lhs is KBitVecValue<T>) {
            return lhs.signedRem(rhs).uncheckedCast()
        }

        if (rhs.isBvOne()) {
            return bvZero(rhs.sort.sizeBits).uncheckedCast()
        }
    }
    return mkBvSignedRemExprNoSimplify(lhs, rhs)
}

fun <T : KBvSort> KContext.simplifyBvUnsignedDivExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> {
    if (rhs is KBitVecValue<T> && !rhs.isBvZero()) {
        if (lhs is KBitVecValue<T>) {
            return lhs.unsignedDivide(rhs).uncheckedCast()
        }

        if (rhs.isBvOne()) {
            return lhs
        }

        rhs.powerOfTwoOrNull()?.let { powerOfTwo ->
            return simplifyBvLogicalShiftRightExpr(lhs, mkBv(powerOfTwo, rhs.sort.sizeBits).uncheckedCast())
        }
    }
    return mkBvUnsignedDivExprNoSimplify(lhs, rhs)
}

fun <T : KBvSort> KContext.simplifyBvUnsignedRemExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> {
    if (rhs is KBitVecValue<T> && !rhs.isBvZero()) {
        if (lhs is KBitVecValue<T>) {
            return lhs.unsignedRem(rhs).uncheckedCast()
        }

        if (rhs.isBvOne()) {
            return bvZero(rhs.sort.sizeBits).uncheckedCast()
        }

        val powerOfTwo = rhs.powerOfTwoOrNull()
        if (powerOfTwo != null) {
            // take all bits
            if (powerOfTwo >= rhs.sort.sizeBits.toInt()) {
                return lhs
            }

            val remainderBits = simplifyBvExtractExpr(high = powerOfTwo - 1, low = 0, lhs)
            val normalizedRemainder = simplifyBvZeroExtensionExpr(
                extensionSize = rhs.sort.sizeBits.toInt() - powerOfTwo, remainderBits
            )
            return normalizedRemainder.uncheckedCast()
        }
    }
    return mkBvUnsignedRemExprNoSimplify(lhs, rhs)
}


fun <T : KBvSort> KContext.simplifyBvReductionAndExpr(arg: KExpr<T>): KExpr<KBv1Sort> =
    mkBvReductionAndExprNoSimplify(arg)

fun <T : KBvSort> KContext.simplifyBvReductionOrExpr(arg: KExpr<T>): KExpr<KBv1Sort> =
    mkBvReductionOrExprNoSimplify(arg)

fun <T : KBvSort> KContext.simplifyBvArithShiftRightExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    mkBvArithShiftRightExprNoSimplify(lhs, rhs)

fun <T : KBvSort> KContext.simplifyBvLogicalShiftRightExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    mkBvLogicalShiftRightExprNoSimplify(lhs, rhs)

fun <T : KBvSort> KContext.simplifyBvShiftLeftExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    mkBvShiftLeftExprNoSimplify(lhs, rhs)

fun <T : KBvSort> KContext.simplifyBvRotateLeftExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    mkBvRotateLeftExprNoSimplify(lhs, rhs)

fun <T : KBvSort> KContext.simplifyBvRotateLeftIndexedExpr(rotation: Int, value: KExpr<T>): KExpr<T> =
    mkBvRotateLeftIndexedExprNoSimplify(rotation, value)

fun <T : KBvSort> KContext.simplifyBvRotateRightExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    mkBvRotateRightExprNoSimplify(lhs, rhs)

fun <T : KBvSort> KContext.simplifyBvRotateRightIndexedExpr(rotation: Int, value: KExpr<T>): KExpr<T> =
    mkBvRotateRightIndexedExprNoSimplify(rotation, value)


fun <T : KBvSort> KContext.simplifyBvRepeatExpr(repeatNumber: Int, value: KExpr<T>): KExpr<KBvSort> =
    mkBvRepeatExprNoSimplify(repeatNumber, value)

fun <T : KBvSort> KContext.simplifyBvZeroExtensionExpr(extensionSize: Int, value: KExpr<T>): KExpr<KBvSort> =
    mkBvZeroExtensionExprNoSimplify(extensionSize, value)

fun <T : KBvSort> KContext.simplifyBvSignExtensionExpr(extensionSize: Int, value: KExpr<T>): KExpr<KBvSort> =
    mkBvSignExtensionExprNoSimplify(extensionSize, value)

fun <T : KBvSort> KContext.simplifyBvExtractExpr(high: Int, low: Int, value: KExpr<T>): KExpr<KBvSort> =
    mkBvExtractExprNoSimplify(high, low, value)

fun <T : KBvSort, S : KBvSort> KContext.simplifyBvConcatExpr(lhs: KExpr<T>, rhs: KExpr<S>): KExpr<KBvSort> =
    mkBvConcatExprNoSimplify(lhs, rhs)


// (sgt a b) ==> (not (sle a b))
fun <T : KBvSort> KContext.simplifyBvSignedGreaterExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    simplifyNot(simplifyBvSignedLessOrEqualExpr(lhs, rhs))

// (sge a b) ==> (sle b a)
fun <T : KBvSort> KContext.simplifyBvSignedGreaterOrEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    simplifyBvSignedLessOrEqualExpr(rhs, lhs)

// (slt a b) ==> (not (sle b a))
fun <T : KBvSort> KContext.simplifyBvSignedLessExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    simplifyNot(simplifyBvSignedLessOrEqualExpr(rhs, lhs))

fun <T : KBvSort> KContext.simplifyBvSignedLessOrEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    bvLessOrEqual(lhs, rhs, signed = true)

// (ugt a b) ==> (not (ule a b))
fun <T : KBvSort> KContext.simplifyBvUnsignedGreaterExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    simplifyNot(simplifyBvUnsignedLessOrEqualExpr(lhs, rhs))

// (uge a b) ==> (ule b a)
fun <T : KBvSort> KContext.simplifyBvUnsignedGreaterOrEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    simplifyBvUnsignedLessOrEqualExpr(rhs, lhs)

// (ult a b) ==> (not (ule b a))
fun <T : KBvSort> KContext.simplifyBvUnsignedLessExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    simplifyNot(simplifyBvUnsignedLessOrEqualExpr(rhs, lhs))

fun <T : KBvSort> KContext.simplifyBvUnsignedLessOrEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    bvLessOrEqual(lhs, rhs, signed = false)


fun <T : KBvSort> KContext.simplifyBvAddNoOverflowExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    isSigned: Boolean
): KExpr<KBoolSort> = mkBvAddNoOverflowExprNoSimplify(lhs, rhs, isSigned)

fun <T : KBvSort> KContext.simplifyBvAddNoUnderflowExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    mkBvAddNoUnderflowExprNoSimplify(lhs, rhs)

fun <T : KBvSort> KContext.simplifyBvMulNoOverflowExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    isSigned: Boolean
): KExpr<KBoolSort> = mkBvMulNoOverflowExprNoSimplify(lhs, rhs, isSigned)

fun <T : KBvSort> KContext.simplifyBvMulNoUnderflowExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    mkBvMulNoUnderflowExprNoSimplify(lhs, rhs)

fun <T : KBvSort> KContext.simplifyBvNegationNoOverflowExpr(arg: KExpr<T>): KExpr<KBoolSort> =
    mkBvNegationNoOverflowExprNoSimplify(arg)

fun <T : KBvSort> KContext.simplifyBvDivNoOverflowExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    mkBvDivNoOverflowExprNoSimplify(lhs, rhs)

fun <T : KBvSort> KContext.simplifyBvSubNoOverflowExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    mkBvSubNoOverflowExprNoSimplify(lhs, rhs)

fun <T : KBvSort> KContext.simplifyBvSubNoUnderflowExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    isSigned: Boolean
): KExpr<KBoolSort> = mkBvSubNoUnderflowExprNoSimplify(lhs, rhs, isSigned)


fun <T : KBvSort> KContext.simplifyBv2IntExpr(value: KExpr<T>, isSigned: Boolean): KExpr<KIntSort> =
    mkBv2IntExprNoSimplify(value, isSigned)


private fun <T : KBvSort> KContext.bvLessOrEqual(lhs: KExpr<T>, rhs: KExpr<T>, signed: Boolean): KExpr<KBoolSort> {
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
                return simplifyEq(lhs, rhs)
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
                return simplifyEq(lhs, rhs)
            }
        }
    }

    return if (signed) {
        mkBvSignedLessOrEqualExprNoSimplify(lhs, rhs)
    } else {
        mkBvUnsignedLessOrEqualExprNoSimplify(lhs, rhs)
    }
}

private fun KBitVecValue<*>.isMinValue(signed: Boolean) =
    if (signed) isBvMinValueSigned() else isBvZero()

private fun KBitVecValue<*>.isMaxValue(signed: Boolean) =
    if (signed) isBvMaxValueSigned() else isBvMaxValueUnsigned()

private inline fun <T : KBvSort> preferLeftValue(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    body: (KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> = if (rhs is KBitVecValue<T>) {
    body(rhs, lhs)
} else {
    body(lhs, rhs)
}

private inline fun <T : KBvSort> evalBvOperationOr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    operation: (KBitVecValue<T>, KBitVecValue<T>) -> KBitVecValue<*>,
    body: () -> KExpr<T>
): KExpr<T> = if (lhs is KBitVecValue<T> && rhs is KBitVecValue<T>) {
    operation(lhs, rhs).uncheckedCast()
} else {
    body()
}
