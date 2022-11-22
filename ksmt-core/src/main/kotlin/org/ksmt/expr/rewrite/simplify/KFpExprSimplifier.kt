package org.ksmt.expr.rewrite.simplify

import org.ksmt.expr.KExpr
import org.ksmt.expr.KFp32Value
import org.ksmt.expr.KFp64Value
import org.ksmt.expr.KFpValue
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KFpSort

interface KFpExprSimplifier : KExprSimplifierBase {

    fun <T : KFpSort> simplifyEqFp(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> = with(ctx) {
        if (lhs == rhs) return trueExpr

        if (lhs is KFpValue<*> && rhs is KFpValue<*>) {
            // special cases
            if (lhs.isNan() && rhs.isNan()) return trueExpr
            if (lhs.isZero() && rhs.isZero() && lhs.signBit != rhs.signBit) return falseExpr

            // compare floats
            return (lhs.compareTo(rhs) == 0).expr
        }

        return mkEq(lhs, rhs)
    }

    fun <T : KFpSort> areDefinitelyDistinctFp(lhs: KExpr<T>, rhs: KExpr<T>): Boolean {
        if (lhs is KFpValue<*> && rhs is KFpValue<*>) {
            // special cases
            if (lhs.isNan() != rhs.isNan()) return true
            if (lhs.isZero() != rhs.isZero()) return true
            if (lhs.isZero() && lhs.signBit != rhs.signBit) return true

            // compare floats
            return lhs.compareTo(rhs) != 0
        }
        return false
    }

    private fun KFpValue<*>.isNan(): Boolean = when (this) {
        is KFp32Value -> value.isNaN()
        is KFp64Value -> value.isNaN()
        else -> TODO("Float isNan: $this")
    }

    private fun KFpValue<*>.isZero(): Boolean = when (this) {
        is KFp32Value -> value == 0.0f || value == -0.0f
        is KFp64Value -> value == 0.0 || value == -0.0
        else -> TODO("Float isZero: $this")
    }

    private operator fun KFpValue<*>.compareTo(other: KFpValue<*>): Int = when {
        this is KFp32Value && other is KFp32Value -> value.compareTo(other.value)
        this is KFp32Value && other is KFp64Value -> value.compareTo(other.value)
        this is KFp64Value && other is KFp64Value -> value.compareTo(other.value)
        this is KFp64Value && other is KFp32Value -> value.compareTo(other.value)
        else -> TODO("Compare floats: $this compareTo $other")
    }

}
