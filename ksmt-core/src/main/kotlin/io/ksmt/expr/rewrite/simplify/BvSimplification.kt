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
import io.ksmt.expr.KBvXorExpr
import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KIntSort
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
import io.ksmt.utils.BvUtils.concatBv
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
import io.ksmt.utils.toBigInteger
import io.ksmt.utils.uncheckedCast


fun <T : KBvSort> KContext.simplifyBvNotExpr(arg: KExpr<T>): KExpr<T> = when (arg) {
    is KBitVecValue<T> -> arg.bitwiseNot().uncheckedCast()
    // (bvnot (bvnot a)) ==> a
    is KBvNotExpr<T> -> arg.value
    else -> mkBvNotExprNoSimplify(arg)
}

fun <T : KBvSort> KContext.simplifyBvOrExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    evalBvOperationOr(lhs, rhs, { a, b -> a.bitwiseOr(b) }) {
        // (bvor x const) ==> (bvor const x)
        preferLeftValue(lhs, rhs) { left, right ->
            if (left is KBitVecValue<T>) {
                // (bvor 0xFFFF... a) ==> 0xFFFF...
                if (left.isBvMaxValueUnsigned()) {
                    return left
                }

                // (bvor 0x0000... a) ==> a
                if (left.isBvZero()) {
                    return right
                }

                if (right is KBvOrExpr<T>) {
                    tryFlatOneLevel(right, { arg0 }, { arg1 }) { rightValue, rightOther ->
                        return simplifyBvOrExpr(left.bitwiseOr(rightValue).uncheckedCast(), rightOther)
                    }
                }
            }

            mkBvOrExprNoSimplify(left, right)
        }
    }

fun <T : KBvSort> KContext.simplifyBvAndExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    evalBvOperationOr(lhs, rhs, { a, b -> a.bitwiseAnd(b) }) {
        // (bvand x const) ==> (bvand const x)
        preferLeftValue(lhs, rhs) { left, right ->
            if (left is KBitVecValue<T>) {
                // (bvand 0xFFFF... a) ==> a
                if (left.isBvMaxValueUnsigned()) {
                    return right
                }

                // (bvand 0x0000... a) ==> 0x0000...
                if (left.isBvZero()) {
                    return left
                }

                if (right is KBvAndExpr<T>) {
                    tryFlatOneLevel(right, { arg0 }, { arg1 }) { rightValue, rightOther ->
                        return simplifyBvAndExpr(left.bitwiseAnd(rightValue).uncheckedCast(), rightOther)
                    }
                }
            }

            mkBvAndExprNoSimplify(left, right)
        }
    }

fun <T : KBvSort> KContext.simplifyBvXorExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    evalBvOperationOr(lhs, rhs, { a, b -> a.bitwiseXor(b) }) {
        // (bvxor x const) ==> (bvxor const x)
        preferLeftValue(lhs, rhs) { left, right ->
            if (left is KBitVecValue<T>) {
                // (bvxor 0xFFFF... a) ==> (bvnot a)
                if (left.isBvMaxValueUnsigned()) {
                    return simplifyBvNotExpr(right)
                }

                // (bvxor 0x0000... a) ==> a
                if (left.isBvZero()) {
                    return right
                }

                if (right is KBvXorExpr<T>) {
                    tryFlatOneLevel(right, { arg0 }, { arg1 }) { rightValue, rightOther ->
                        return simplifyBvXorExpr(left.bitwiseXor(rightValue).uncheckedCast(), rightOther)
                    }
                }
            }

            mkBvXorExprNoSimplify(left, right)
        }
    }

fun <T : KBvSort> KContext.simplifyBvNorExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyBvNotExpr(simplifyBvOrExpr(lhs, rhs))

fun <T : KBvSort> KContext.simplifyBvNAndExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyBvNotExpr(simplifyBvAndExpr(lhs, rhs))

fun <T : KBvSort> KContext.simplifyBvXNorExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyBvNotExpr(simplifyBvXorExpr(lhs, rhs))


fun <T : KBvSort> KContext.simplifyBvNegationExpr(arg: KExpr<T>): KExpr<T> {
    if (arg is KBitVecValue<T>) return (-arg).uncheckedCast()

    if (arg is KBvNegationExpr<T>) return arg.value

    if (arg is KBvAddExpr<T>) {
        // (bvneg (bvadd a b)) ==> (bvadd (bvneg a) (bvneg b))
        val lhsIsSuitableFoRewrite = arg.arg0 is KBitVecValue<T> || arg.arg0 is KBvNegationExpr<T>
        val rhsIsSuitableFoRewrite = arg.arg1 is KBitVecValue<T> || arg.arg1 is KBvNegationExpr<T>
        if (lhsIsSuitableFoRewrite || rhsIsSuitableFoRewrite) {
            return simplifyBvAddExpr(simplifyBvNegationExpr(arg.arg0), simplifyBvNegationExpr(arg.arg1))
        }
    }

    return mkBvNegationExprNoSimplify(arg)
}

fun <T : KBvSort> KContext.simplifyBvAddExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    evalBvOperationOr(lhs, rhs, { a, b -> a + b }) {
        // (bvadd x const) ==> (bvadd const x)
        preferLeftValue(lhs, rhs) { left, right ->
            if (left is KBitVecValue<T> && left.isBvZero()) {
                return right
            }

            // flat one level
            if (left is KBitVecValue<T> && right is KBvAddExpr<T>) {
                tryFlatOneLevel(right, { arg0 }, { arg1 }) { rightValue, rightOther ->
                    return simplifyBvAddExpr((left + rightValue).uncheckedCast(), rightOther)
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
                    tryFlatOneLevel(right, { arg0 }, { arg1 }) { rightValue, rightOther ->
                        return simplifyBvMulExpr((left * rightValue).uncheckedCast(), rightOther)
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
    if (arg is KBitVecValue<T>) {
        // 0xFFFFF -> 1 and 0 otherwise
        mkBv(arg.isBvMaxValueUnsigned())
    } else {
        mkBvReductionAndExprNoSimplify(arg)
    }

fun <T : KBvSort> KContext.simplifyBvReductionOrExpr(arg: KExpr<T>): KExpr<KBv1Sort> =
    if (arg is KBitVecValue<T>) {
        // 0x00000 -> 0 and 1 otherwise
        mkBv(!arg.isBvZero())
    } else {
        mkBvReductionOrExprNoSimplify(arg)
    }

fun <T : KBvSort> KContext.simplifyBvArithShiftRightExpr(lhs: KExpr<T>, shift: KExpr<T>): KExpr<T> {
    if (shift is KBitVecValue<T>) {
        if (lhs is KBitVecValue<T>) {
            return lhs.shiftRightArith(shift).uncheckedCast()
        }

        // (x >> 0) ==> x
        if (shift.isBvZero()) {
            return lhs
        }
    }
    return mkBvArithShiftRightExprNoSimplify(lhs, shift)
}

fun <T : KBvSort> KContext.simplifyBvLogicalShiftRightExpr(lhs: KExpr<T>, shift: KExpr<T>): KExpr<T> {
    if (shift is KBitVecValue<T>) {
        if (lhs is KBitVecValue<T>) {
            return lhs.shiftRightLogical(shift).uncheckedCast()
        }

        // (x >>> 0) ==> x
        if (shift.isBvZero()) {
            return lhs
        }

        // (x >>> shift), shift >= size ==> 0
        if (shift.signedGreaterOrEqual(shift.sort.sizeBits.toInt())) {
            return bvZero(shift.sort.sizeBits).uncheckedCast()
        }
    }

    // (x >>> x) ==> 0
    if (lhs == shift) {
        return bvZero(shift.sort.sizeBits).uncheckedCast()
    }

    return mkBvLogicalShiftRightExprNoSimplify(lhs, shift)
}

fun <T : KBvSort> KContext.simplifyBvShiftLeftExpr(lhs: KExpr<T>, shift: KExpr<T>): KExpr<T> {
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
    return mkBvShiftLeftExprNoSimplify(lhs, shift)
}

// (rotateLeft a x) ==> (concat (extract [size-1-x:0] a) (extract [size-1:size-x] a))
fun <T : KBvSort> KContext.simplifyBvRotateLeftExpr(lhs: KExpr<T>, rotation: KExpr<T>): KExpr<T> {
    if (rotation is KBitVecValue<T>) {
        val size = rotation.sort.sizeBits.toInt()
        val intValue = rotation.bigIntValue()
        val rotationValue = intValue.remainder(size.toBigInteger()).toInt()
        return simplifyBvRotateLeftIndexedExpr(rotationValue, lhs)
    }
    return mkBvRotateLeftExprNoSimplify(lhs, rotation)
}

fun <T : KBvSort> KContext.simplifyBvRotateLeftIndexedExpr(rotation: Int, value: KExpr<T>): KExpr<T> {
    val size = value.sort.sizeBits.toInt()
    return rotateLeft(value, size, rotation)
}

// (rotateRight a x) ==> (rotateLeft a (- size x))
fun <T : KBvSort> KContext.simplifyBvRotateRightExpr(lhs: KExpr<T>, rotation: KExpr<T>): KExpr<T> {
    if (rotation is KBitVecValue<T>) {
        val size = rotation.sort.sizeBits.toInt()
        val intValue = rotation.bigIntValue()
        val rotationValue = intValue.remainder(size.toBigInteger()).toInt()
        return simplifyBvRotateRightIndexedExpr(rotationValue, lhs)
    }
    return mkBvRotateRightExprNoSimplify(lhs, rotation)
}

fun <T : KBvSort> KContext.simplifyBvRotateRightIndexedExpr(rotation: Int, value: KExpr<T>): KExpr<T> {
    val size = value.sort.sizeBits.toInt()
    val normalizedRotation = rotation % size
    return rotateLeft(value, size, rotationNumber = size - normalizedRotation)
}

// (repeat a x) ==> (concat a a ..[x].. a)
fun <T : KBvSort> KContext.simplifyBvRepeatExpr(repeatNumber: Int, value: KExpr<T>): KExpr<KBvSort> {
    if (value is KBitVecValue<T> && repeatNumber > 0) {
        var result: KBitVecValue<*> = value
        repeat(repeatNumber - 1) {
            result = concatBv(result, value)
        }
        return result.uncheckedCast()
    }
    return mkBvRepeatExprNoSimplify(repeatNumber, value)
}

fun <T : KBvSort> KContext.simplifyBvZeroExtensionExpr(extensionSize: Int, value: KExpr<T>): KExpr<KBvSort> {
    if (extensionSize == 0) {
        return value.uncheckedCast()
    }

    if (value is KBitVecValue<T>) {
        return value.zeroExtension(extensionSize.toUInt()).uncheckedCast()
    }

    return mkBvZeroExtensionExprNoSimplify(extensionSize, value)
}

fun <T : KBvSort> KContext.simplifyBvSignExtensionExpr(extensionSize: Int, value: KExpr<T>): KExpr<KBvSort> {
    if (extensionSize == 0) {
        return value.uncheckedCast()
    }

    if (value is KBitVecValue<T>) {
        return value.signExtension(extensionSize.toUInt()).uncheckedCast()
    }

    return mkBvSignExtensionExprNoSimplify(extensionSize, value)
}

fun <T : KBvSort> KContext.simplifyBvExtractExpr(high: Int, low: Int, value: KExpr<T>): KExpr<KBvSort> {
    // (extract [size-1:0] x) ==> x
    if (low == 0 && high == value.sort.sizeBits.toInt() - 1) {
        return value.uncheckedCast()
    }

    if (value is KBitVecValue<T>) {
        return value.extractBv(high, low).uncheckedCast()
    }

    // (extract[high:low] (extract[_:nestedLow] x)) ==> (extract[high+nestedLow : low+nestedLow] x)
    if (value is KBvExtractExpr) {
        val nestedLow = value.low
        return simplifyBvExtractExpr(
            high = high + nestedLow,
            low = low + nestedLow,
            value = value.value
        )
    }

    return mkBvExtractExprNoSimplify(high, low, value)
}

fun <T : KBvSort, S : KBvSort> KContext.simplifyBvConcatExpr(lhs: KExpr<T>, rhs: KExpr<S>): KExpr<KBvSort> {
    if (lhs is KBitVecValue<T> && rhs is KBitVecValue<S>) {
        return concatBv(lhs, rhs).uncheckedCast()
    }

    if (lhs is KBitVecValue<T> && rhs is KBvConcatExpr) {
        val rhsLeft = rhs.arg0
        if (rhsLeft is KBitVecValue<*>) {
            return simplifyBvConcatExpr(concatBv(lhs, rhsLeft), rhs.arg1)
        }
    }

    if (rhs is KBitVecValue<S> && lhs is KBvConcatExpr) {
        val lhsRight = lhs.arg1
        if (lhsRight is KBitVecValue<*>) {
            return simplifyBvConcatExpr(lhs.arg0, concatBv(lhsRight, rhs))
        }
    }

    return mkBvConcatExprNoSimplify(lhs, rhs)
}


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


fun <T : KBvSort> KContext.simplifyBv2IntExpr(value: KExpr<T>, isSigned: Boolean): KExpr<KIntSort> {
    if (value is KBitVecValue<T>) {
        val integerValue = if (isSigned) {
            value.toBigIntegerSigned()
        } else {
            value.toBigIntegerUnsigned()
        }
        return mkIntNum(integerValue)
    }
    return mkBv2IntExprNoSimplify(value, isSigned)
}


fun <T : KBvSort> KContext.simplifyBvAddNoOverflowExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    isSigned: Boolean
): KExpr<KBoolSort> {
    if (lhs is KBitVecValue<T> && rhs is KBitVecValue<T>) {
        return rewriteBvAddNoOverflowExpr(lhs, rhs, isSigned)
    }
    return mkBvAddNoOverflowExprNoSimplify(lhs, rhs, isSigned)
}

fun <T : KBvSort> KContext.simplifyBvAddNoUnderflowExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> {
    if (lhs is KBitVecValue<T> && rhs is KBitVecValue<T>) {
        return rewriteBvAddNoUnderflowExpr(lhs, rhs)
    }
    return mkBvAddNoUnderflowExprNoSimplify(lhs, rhs)
}

fun <T : KBvSort> KContext.simplifyBvMulNoOverflowExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    isSigned: Boolean
): KExpr<KBoolSort> {
    if (lhs is KBitVecValue<T> && rhs is KBitVecValue<T>) {
        return rewriteBvMulNoOverflowExpr(lhs, rhs, isSigned)
    }
    return mkBvMulNoOverflowExprNoSimplify(lhs, rhs, isSigned)
}

fun <T : KBvSort> KContext.simplifyBvMulNoUnderflowExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> {
    if (lhs is KBitVecValue<T> && rhs is KBitVecValue<T>) {
        return rewriteBvMulNoUnderflowExpr(lhs, rhs)
    }
    return mkBvMulNoUnderflowExprNoSimplify(lhs, rhs)
}

fun <T : KBvSort> KContext.simplifyBvNegationNoOverflowExpr(arg: KExpr<T>): KExpr<KBoolSort> {
    if (arg is KBitVecValue<T>) {
        return rewriteBvNegNoOverflowExpr(arg)
    }
    return mkBvNegationNoOverflowExprNoSimplify(arg)
}

fun <T : KBvSort> KContext.simplifyBvDivNoOverflowExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> {
    if (lhs is KBitVecValue<T> && rhs is KBitVecValue<T>) {
        return rewriteBvDivNoOverflowExpr(lhs, rhs)
    }
    return mkBvDivNoOverflowExprNoSimplify(lhs, rhs)
}

fun <T : KBvSort> KContext.simplifyBvSubNoOverflowExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> {
    if (lhs is KBitVecValue<T> && rhs is KBitVecValue<T>) {
        return rewriteBvSubNoOverflowExpr(lhs, rhs)
    }
    return mkBvSubNoOverflowExprNoSimplify(lhs, rhs)
}

fun <T : KBvSort> KContext.simplifyBvSubNoUnderflowExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    isSigned: Boolean
): KExpr<KBoolSort> {
    if (lhs is KBitVecValue<T> && rhs is KBitVecValue<T>) {
        return rewriteBvSubNoUnderflowExpr(lhs, rhs, isSigned)
    }
    return mkBvSubNoUnderflowExprNoSimplify(lhs, rhs, isSigned)
}


fun <T : KBvSort> KContext.rewriteBvAddNoOverflowExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    isSigned: Boolean
): KExpr<KBoolSort> {
    if (isSigned) {
        /**
         * (bvadd no ovf signed a b) ==>
         *    (=> (and (bvslt 0 a) (bvslt 0 b)) (bvslt 0 (bvadd a b)))
         * */

        val zero: KExpr<T> = bvZero(lhs.sort.sizeBits).uncheckedCast()
        val zeroSltA = simplifyBvSignedLessExpr(zero, lhs)
        val zeroSltB = simplifyBvSignedLessExpr(zero, rhs)
        val sum = simplifyBvAddExpr(lhs, rhs)
        val zeroSltSum = simplifyBvSignedLessExpr(zero, sum)

        return simplifyImplies(simplifyAnd(zeroSltA, zeroSltB), zeroSltSum)
    } else {
        /**
         * (bvadd no ovf unsigned a b) ==>
         *    (= 0 (extract [highestBit] (bvadd (concat 0 a) (concat 0 b))))
         * */

        val extA = simplifyBvZeroExtensionExpr(1, lhs)
        val extB = simplifyBvZeroExtensionExpr(1, rhs)
        val sum = simplifyBvAddExpr(extA, extB)
        val highestBitIdx = sum.sort.sizeBits.toInt() - 1
        val sumFirstBit = simplifyBvExtractExpr(highestBitIdx, highestBitIdx, sum)

        return simplifyEq(sumFirstBit, mkBv(false).uncheckedCast())
    }
}

fun <T : KBvSort> KContext.rewriteBvAddNoUnderflowExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> {
    /**
     * (bvadd no udf a b) ==>
     *    (=> (and (bvslt a 0) (bvslt b 0)) (bvslt (bvadd a b) 0))
     * */
    val zero: KExpr<T> = bvZero(lhs.sort.sizeBits).uncheckedCast()
    val aLtZero = simplifyBvSignedLessExpr(lhs, zero)
    val bLtZero = simplifyBvSignedLessExpr(rhs, zero)
    val sum = simplifyBvAddExpr(lhs, rhs)
    val sumLtZero = simplifyBvSignedLessExpr(sum, zero)

    return simplifyImplies(simplifyAnd(aLtZero, bLtZero), sumLtZero)
}

fun <T : KBvSort> KContext.rewriteBvSubNoOverflowExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> {
    /**
     * (bvsub no ovf a b) ==>
     *     (ite (= b MIN_VALUE) (bvslt a 0) (bvadd no ovf signed a (bvneg b)))
     * */

    val zero: KExpr<T> = bvZero(lhs.sort.sizeBits).uncheckedCast()
    val minValue: KExpr<T> = bvMinValueSigned(lhs.sort.sizeBits).uncheckedCast()

    val minusB = simplifyBvNegationExpr(rhs)
    val bIsMin = simplifyEq(rhs, minValue)
    val aLtZero = simplifyBvSignedLessExpr(lhs, zero)
    val noOverflow = simplifyBvAddNoOverflowExpr(lhs, minusB, isSigned = true)

    return simplifyIte(bIsMin, aLtZero, noOverflow)
}

fun <T : KBvSort> KContext.rewriteBvSubNoUnderflowExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    isSigned: Boolean
): KExpr<KBoolSort> {
    if (isSigned) {
        /**
         * (bvsub no udf signed a b) ==>
         *    (=> (bvslt 0 b) (bvadd no udf (bvneg b)))
         * */
        val zero: KExpr<T> = bvZero(lhs.sort.sizeBits).uncheckedCast()
        val minusB = simplifyBvNegationExpr(rhs)
        val zeroLtB = simplifyBvSignedLessExpr(zero, rhs)
        val noOverflow = simplifyBvAddNoUnderflowExpr(lhs, minusB)

        return simplifyImplies(zeroLtB, noOverflow)
    } else {
        /**
         * (bvsub no udf unsigned a b) ==>
         *    (bvule b a)
         * */
        return simplifyBvUnsignedLessOrEqualExpr(rhs, lhs)
    }
}

fun <T : KBvSort> KContext.rewriteBvNegNoOverflowExpr(arg: KExpr<T>): KExpr<KBoolSort> {
    /**
     * (bvneg no ovf a) ==> (not (= a MIN_VALUE))
     * */
    val minValue = bvMinValueSigned(arg.sort.sizeBits)
    return simplifyNot(simplifyEq(arg, minValue.uncheckedCast()))
}

fun <T : KBvSort> KContext.rewriteBvDivNoOverflowExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> {
    /**
     * (bvsdiv no ovf a b) ==>
     *     (not (and (= a MSB) (= b -1)))
     * */
    val size = lhs.sort.sizeBits
    val mostSignificantBit = bvMinValueSigned(size)
    val minusOne = bvValue(size, -1)

    val aIsMsb = simplifyEq(lhs, mostSignificantBit.uncheckedCast())
    val bIsMinusOne = simplifyEq(rhs, minusOne.uncheckedCast())
    return simplifyNot(simplifyAnd(aIsMsb, bIsMinusOne))
}

fun <T : KBvSort> KContext.rewriteBvMulNoOverflowExpr(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    isSigned: Boolean
): KExpr<KBoolSort> {
    val simplified = if (isSigned) {
        trySimplifyBvSignedMulNoOverflow(lhs, rhs, isOverflow = true)
    } else {
        trySimplifyBvUnsignedMulNoOverflow(lhs, rhs)
    }
    return simplified ?: mkBvMulNoOverflowExprNoSimplify(lhs, rhs, isSigned)
}

fun <T : KBvSort> KContext.rewriteBvMulNoUnderflowExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    trySimplifyBvSignedMulNoOverflow(lhs, rhs, isOverflow = false)
        ?: mkBvMulNoUnderflowExprNoSimplify(lhs, rhs)


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

private inline fun <reified T : KExpr<S>, S : KBvSort> tryFlatOneLevel(
    expr: T,
    lhs: T.() -> KExpr<S>,
    rhs: T.() -> KExpr<S>,
    block: (KBitVecValue<S>, KExpr<S>) -> Unit
) {
    val exprLeft = expr.lhs()
    val exprRight = expr.rhs()

    if (exprLeft is KBitVecValue<S>) {
        return block(exprLeft, exprRight)
    }

    if (exprRight is KBitVecValue<S>) {
        return block(exprRight, exprLeft)
    }
}

private fun <T : KBvSort> KContext.rotateLeft(arg: KExpr<T>, size: Int, rotationNumber: Int): KExpr<T> {
    val rotation = rotationNumber.mod(size)

    if (rotation == 0 || size == 1) {
        return arg
    }

    val lhs = simplifyBvExtractExpr(high = size - rotation - 1, low = 0, arg)
    val rhs = simplifyBvExtractExpr(high = size - 1, low = size - rotation, arg)
    return simplifyBvConcatExpr(lhs, rhs).uncheckedCast()
}

private fun <T : KBvSort> KContext.trySimplifyBvSignedMulNoOverflow(
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

private fun <T : KBvSort> KContext.trySimplifyBvUnsignedMulNoOverflow(
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

private fun <T : KBvSort> KContext.evalBvSignedMulNoOverflow(
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
