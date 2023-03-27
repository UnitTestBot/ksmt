package org.ksmt.expr.rewrite.simplify

import org.ksmt.expr.KConst
import org.ksmt.expr.KExpr
import org.ksmt.expr.KInterpretedValue
import org.ksmt.sort.KSort

object ExpressionOrdering : Comparator<KExpr<*>> {
    override fun compare(left: KExpr<*>, right: KExpr<*>): Int = when (left) {
        is KInterpretedValue<*> -> when (right) {
            is KInterpretedValue<*> -> compareDefault(left, right)
            else -> -1
        }

        is KConst<*> -> when (right) {
            is KInterpretedValue<*> -> 1
            is KConst<*> -> compareDefault(left, right)
            else -> -1
        }

        else -> compareDefault(left, right)
    }

    @JvmStatic
    private fun compareDefault(left: KExpr<*>?, right: KExpr<*>?): Int =
        System.identityHashCode(left).compareTo(System.identityHashCode(right))
}

inline fun <T : KSort, R> withExpressionsOrdered(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    body: (KExpr<T>, KExpr<T>) -> R
): R = if (ExpressionOrdering.compare(lhs, rhs) <= 0) {
    body(lhs, rhs)
} else {
    body(rhs, lhs)
}

fun <T : KSort> MutableList<KExpr<T>>.ensureExpressionsOrder() {
    sortWith(ExpressionOrdering)
}
