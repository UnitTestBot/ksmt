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

/**
 * Aux expressions.
 * During the simplification process, we create many auxiliary expressions
 * that will be immediately rewritten.
 * Since we have no option to cleanup [KContext] it is important to store
 * as fewer expressions as possible.
 * We abuse the fact that expression will be rewritten
 * and skip hash-consing via KContext.
 * */
@JvmInline
internal value class SimplifierAuxExpression<T : KSort>(val expr: KExpr<T>)

internal inline fun <T : KSort> auxExpr(builder: () -> KExpr<T>): SimplifierAuxExpression<T> =
    SimplifierAuxExpression(builder())

internal fun <T : KSort> KExprSimplifierBase.rewrite(expr: SimplifierAuxExpression<T>): KExpr<T> =
    rewrite(expr.expr)
