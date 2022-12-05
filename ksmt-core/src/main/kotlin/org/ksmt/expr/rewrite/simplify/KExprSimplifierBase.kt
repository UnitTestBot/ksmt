package org.ksmt.expr.rewrite.simplify

import org.ksmt.expr.KExpr
import org.ksmt.expr.transformer.KTransformer
import org.ksmt.sort.KSort

interface KExprSimplifierBase : KTransformer {
    /**
     * Checks if the provided expressions can never be equal.
     * For example, two unequal theory interpreted constants cannot be equal.
     * @return true if expressions are unequal and false if equality cannot be checked.
     * */
    fun <T : KSort> areDefinitelyDistinct(lhs: KExpr<T>, rhs: KExpr<T>): Boolean

    /**
     * Force simplifier to rewrite an expression.
     * Typically used for a new expression created by simplifying another expression.
     * */
    fun <T : KSort> rewrite(expr: KExpr<T>): KExpr<T>
}

@JvmInline
internal value class SimplifierAuxExpression<T : KSort>(val expr: KExpr<T>)

internal inline fun <T : KSort> auxExpr(builder: () -> KExpr<T>): SimplifierAuxExpression<T> =
    SimplifierAuxExpression(builder())

internal fun <T : KSort> KExprSimplifierBase.rewrite(expr: SimplifierAuxExpression<T>): KExpr<T> =
    rewrite(expr.expr)
