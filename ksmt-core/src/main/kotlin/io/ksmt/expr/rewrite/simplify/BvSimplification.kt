package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KBitVecValue
import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KIntSort


fun <T : KBvSort> KContext.simplifyBvNotExpr(arg: KExpr<T>): KExpr<T> =
    simplifyBvNotExprLight(arg, ::mkBvNotExprNoSimplify)

fun <T : KBvSort> KContext.simplifyBvOrExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyBvOrExprLight(lhs, rhs) { lhs2, rhs2 ->
        simplifyBvOrExprNestedOr(lhs2, rhs2, KContext::simplifyBvOrExpr, ::mkBvOrExprNoSimplify)
    }

fun <T : KBvSort> KContext.simplifyBvAndExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyBvAndExprLight(lhs, rhs) { lhs2, rhs2 ->
        simplifyBvAndExprNestedAnd(lhs2, rhs2, KContext::simplifyBvAndExpr, ::mkBvAndExprNoSimplify)
    }

fun <T : KBvSort> KContext.simplifyBvXorExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyBvXorExprLight(lhs, rhs) { lhs2, rhs2 ->
        simplifyBvXorExprMaxConst(lhs2, rhs2, KContext::simplifyBvNotExpr) { lhs3, rhs3 ->
            simplifyBvXorExprNestedXor(lhs3, rhs3, KContext::simplifyBvXorExpr, ::mkBvXorExprNoSimplify)
        }
    }

fun <T : KBvSort> KContext.simplifyBvNorExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    rewriteBvNorExpr(lhs, rhs, KContext::simplifyBvOrExpr, KContext::simplifyBvNotExpr)

fun <T : KBvSort> KContext.simplifyBvNAndExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    rewriteBvNAndExpr(lhs, rhs, KContext::simplifyBvOrExpr, KContext::simplifyBvNotExpr)

fun <T : KBvSort> KContext.simplifyBvXNorExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    rewriteBvXNorExpr(lhs, rhs, KContext::simplifyBvXorExpr, KContext::simplifyBvNotExpr)

fun <T : KBvSort> KContext.simplifyBvNegationExpr(arg: KExpr<T>): KExpr<T> =
    simplifyBvNegationExprLight(arg) { arg2 ->
        simplifyBvNegationExprAdd(
            arg = arg2,
            rewriteBvAddExpr = KContext::simplifyBvAddExpr,
            rewriteBvNegationExpr = KContext::simplifyBvNegationExpr,
            cont = ::mkBvNegationExprNoSimplify
        )
    }

fun <T : KBvSort> KContext.simplifyBvAddExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyBvAddExprLight(lhs, rhs) { lhs2, rhs2 ->
        simplifyBvAddExprNestedAdd(lhs2, rhs2, KContext::simplifyBvAddExpr, ::mkBvAddExprNoSimplify)
    }

fun <T : KBvSort> KContext.simplifyBvMulExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyBvMulExprLight(lhs, rhs) { lhs2, rhs2 ->
        simplifyBvMulExprMinusOneConst(lhs2, rhs2, KContext::simplifyBvNegationExpr) { lhs3, rhs3 ->
            simplifyBvMulExprNestedMul(lhs3, rhs3, KContext::simplifyBvMulExpr, ::mkBvMulExprNoSimplify)
        }
    }

fun <T : KBvSort> KContext.simplifyBvSubExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    rewriteBvSubExpr(lhs, rhs, KContext::simplifyBvAddExpr, KContext::simplifyBvNegationExpr)

fun <T : KBvSort> KContext.simplifyBvSignedDivExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyBvSignedDivExprLight(lhs, rhs, ::mkBvSignedDivExprNoSimplify)

fun <T : KBvSort> KContext.simplifyBvSignedModExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyBvSignedModExprLight(lhs, rhs, ::mkBvSignedModExprNoSimplify)

fun <T : KBvSort> KContext.simplifyBvSignedRemExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyBvSignedRemExprLight(lhs, rhs, ::mkBvSignedRemExprNoSimplify)

fun <T : KBvSort> KContext.simplifyBvUnsignedDivExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyBvUnsignedDivExprLight(lhs, rhs) { lhs2, rhs2 ->
        simplifyBvUnsignedDivExprPowerOfTwoDivisor(
            lhs = lhs2,
            rhs = rhs2,
            rewriteBvLogicalShiftRightExpr = KContext::simplifyBvLogicalShiftRightExpr,
            cont = ::mkBvUnsignedDivExprNoSimplify
        )
    }

fun <T : KBvSort> KContext.simplifyBvUnsignedRemExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyBvUnsignedRemExprLight(lhs, rhs) { lhs2, rhs2 ->
        simplifyBvUnsignedRemExprPowerOfTwoDivisor(
            lhs = lhs2,
            rhs = rhs2,
            rewriteBvExtractExpr = KContext::simplifyBvExtractExpr,
            rewriteBvZeroExtensionExpr = KContext::simplifyBvZeroExtensionExpr,
            cont = ::mkBvUnsignedRemExprNoSimplify
        )
    }

fun <T : KBvSort> KContext.simplifyBvReductionAndExpr(arg: KExpr<T>): KExpr<KBv1Sort> =
    simplifyBvReductionAndExprLight(arg, ::mkBvReductionAndExprNoSimplify)

fun <T : KBvSort> KContext.simplifyBvReductionOrExpr(arg: KExpr<T>): KExpr<KBv1Sort> =
    simplifyBvReductionOrExprLight(arg, ::mkBvReductionOrExprNoSimplify)

fun <T : KBvSort> KContext.simplifyBvArithShiftRightExpr(lhs: KExpr<T>, shift: KExpr<T>): KExpr<T> =
    simplifyBvArithShiftRightExprLight(lhs, shift, ::mkBvArithShiftRightExprNoSimplify)

fun <T : KBvSort> KContext.simplifyBvLogicalShiftRightExpr(lhs: KExpr<T>, shift: KExpr<T>): KExpr<T> =
    simplifyBvLogicalShiftRightExprLight(lhs, shift, ::mkBvLogicalShiftRightExprNoSimplify)

fun <T : KBvSort> KContext.simplifyBvShiftLeftExpr(lhs: KExpr<T>, shift: KExpr<T>): KExpr<T> =
    simplifyBvShiftLeftExprLight(lhs, shift, ::mkBvShiftLeftExprNoSimplify)

fun <T : KBvSort> KContext.simplifyBvRotateLeftExpr(lhs: KExpr<T>, rotation: KExpr<T>): KExpr<T> =
    simplifyBvRotateLeftExprConstRotation(
        lhs = lhs,
        rotation = rotation,
        rewriteBvRotateLeftIndexedExpr = KContext::simplifyBvRotateLeftIndexedExpr,
        cont = ::mkBvRotateLeftExprNoSimplify
    )

fun <T : KBvSort> KContext.simplifyBvRotateLeftIndexedExpr(rotation: Int, value: KExpr<T>): KExpr<T> =
    rewriteBvRotateLeftIndexedExpr(rotation, value, KContext::simplifyBvExtractExpr, KContext::simplifyBvConcatExpr)

fun <T : KBvSort> KContext.simplifyBvRotateRightExpr(lhs: KExpr<T>, rotation: KExpr<T>): KExpr<T> =
    simplifyBvRotateRightExprConstRotation(
        lhs = lhs,
        rotation = rotation,
        rewriteBvRotateRightIndexedExpr = KContext::simplifyBvRotateRightIndexedExpr,
        cont = ::mkBvRotateRightExprNoSimplify
    )

fun <T : KBvSort> KContext.simplifyBvRotateRightIndexedExpr(rotation: Int, value: KExpr<T>): KExpr<T> =
    rewriteBvRotateRightIndexedExpr(rotation, value, KContext::simplifyBvExtractExpr, KContext::simplifyBvConcatExpr)

fun <T : KBvSort> KContext.simplifyBvRepeatExpr(repeatNumber: Int, value: KExpr<T>): KExpr<KBvSort> =
    simplifyBvRepeatExprLight(repeatNumber, value, ::mkBvRepeatExprNoSimplify)

fun <T : KBvSort> KContext.simplifyBvZeroExtensionExpr(extensionSize: Int, value: KExpr<T>): KExpr<KBvSort> =
    simplifyBvZeroExtensionExprLight(extensionSize, value, ::mkBvZeroExtensionExprNoSimplify)

fun <T : KBvSort> KContext.simplifyBvSignExtensionExpr(extensionSize: Int, value: KExpr<T>): KExpr<KBvSort> =
    simplifyBvSignExtensionExprLight(extensionSize, value, ::mkBvSignExtensionExprNoSimplify)

fun <T : KBvSort> KContext.simplifyBvExtractExpr(high: Int, low: Int, value: KExpr<T>): KExpr<KBvSort> =
    simplifyBvExtractExprLight(high, low, value) { high2, low2, value2 ->
        simplifyBvExtractExprNestedExtract(
            high = high2,
            low = low2,
            value = value2,
            rewriteBvExtractExpr = KContext::simplifyBvExtractExpr,
            cont = ::mkBvExtractExprNoSimplify
        )
    }

fun <T : KBvSort, S : KBvSort> KContext.simplifyBvConcatExpr(lhs: KExpr<T>, rhs: KExpr<S>): KExpr<KBvSort> =
    simplifyBvConcatExprLight(lhs, rhs) { lhs2, rhs2 ->
        simplifyBvConcatExprNestedConcat(lhs2, rhs2, KContext::simplifyBvConcatExpr, ::mkBvConcatExprNoSimplify)
    }

fun <T : KBvSort> KContext.simplifyBvSignedGreaterExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    rewriteBvSignedGreaterExpr(lhs, rhs, KContext::simplifyBvSignedLessOrEqualExpr, KContext::simplifyNot)

fun <T : KBvSort> KContext.simplifyBvSignedGreaterOrEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    rewriteBvSignedGreaterOrEqualExpr(lhs, rhs, KContext::simplifyBvSignedLessOrEqualExpr)

fun <T : KBvSort> KContext.simplifyBvSignedLessExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    rewriteBvSignedLessExpr(lhs, rhs, KContext::simplifyBvSignedLessOrEqualExpr, KContext::simplifyNot)

fun <T : KBvSort> KContext.simplifyBvSignedLessOrEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    simplifyBvSignedLessOrEqualExprLight(lhs, rhs, KContext::simplifyEq, ::mkBvSignedLessOrEqualExprNoSimplify)

fun <T : KBvSort> KContext.simplifyBvUnsignedGreaterExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    rewriteBvUnsignedGreaterExpr(lhs, rhs, KContext::simplifyBvUnsignedLessOrEqualExpr, KContext::simplifyNot)

fun <T : KBvSort> KContext.simplifyBvUnsignedGreaterOrEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    rewriteBvUnsignedGreaterOrEqualExpr(lhs, rhs, KContext::simplifyBvUnsignedLessOrEqualExpr)

fun <T : KBvSort> KContext.simplifyBvUnsignedLessExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    rewriteBvUnsignedLessExpr(lhs, rhs, KContext::simplifyBvUnsignedLessOrEqualExpr, KContext::simplifyNot)

fun <T : KBvSort> KContext.simplifyBvUnsignedLessOrEqualExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    rewriteBvUnsignedLessOrEqualExprLight(lhs, rhs, KContext::simplifyEq, ::mkBvUnsignedLessOrEqualExprNoSimplify)

fun <T : KBvSort> KContext.simplifyBv2IntExpr(value: KExpr<T>, isSigned: Boolean): KExpr<KIntSort> =
    simplifyBv2IntExprLight(value, isSigned, ::mkBv2IntExprNoSimplify)


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
