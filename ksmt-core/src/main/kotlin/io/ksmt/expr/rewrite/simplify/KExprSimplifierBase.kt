package io.ksmt.expr.rewrite.simplify

import io.ksmt.expr.KExpr
import io.ksmt.expr.transformer.KTransformer
import io.ksmt.sort.KSort
import io.ksmt.utils.uncheckedCast

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

    /**
     * Ask simplifier to rewrite an expression with bound check.
     * Typically used for expressions which produce expressions of the same
     * type during simplification process.
     *
     * [allowedDepth] --- maximal allowed nested rewrites.
     * */
    fun <T : KSort> boundedRewrite(allowedDepth: Int, expr: KExpr<T>): KExpr<T>

    /**
     * Returns true if simplifier has enough depth to perform bounded rewrite.
     * */
    fun canPerformBoundedRewrite(): Boolean
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

/**
 * Mark simplifier auxiliary expressions.
 * */
internal interface KSimplifierAuxExpr

internal inline fun <T : KSort> auxExpr(builder: () -> KExpr<T>): SimplifierAuxExpression<T> =
    SimplifierAuxExpression(builder())

internal fun <T : KSort> KExprSimplifierBase.rewrite(expr: SimplifierAuxExpression<T>): KExpr<T> =
    rewrite(expr.expr)


const val SIMPLIFIER_DEFAULT_BOUNDED_REWRITE_DEPTH = 3

internal fun <T : KSort> KExprSimplifierBase.boundedRewrite(
    expr: SimplifierAuxExpression<T>,
    depth: Int = SIMPLIFIER_DEFAULT_BOUNDED_REWRITE_DEPTH
): KExpr<T> = boundedRewrite(depth, expr.expr)

/**
 * [left] and [right] are definitely distinct if there
 * is at least one distinct pair of expressions.
 * */
fun KExprSimplifierBase.areDefinitelyDistinct(left: List<KExpr<*>>, right: List<KExpr<*>>): Boolean {
    check(left.size == right.size) {
        "Pairwise distinct check requires both lists to be the same size"
    }

    for (i in left.indices) {
        val lhs: KExpr<KSort> = left[i].uncheckedCast()
        val rhs: KExpr<KSort> = right[i].uncheckedCast()
        if (areDefinitelyDistinct(lhs, rhs)) return true
    }
    return false
}

internal fun <T : KSort> KExprSimplifierBase.boundedRewrite(
    expr: KExpr<T>,
    depth: Int = SIMPLIFIER_DEFAULT_BOUNDED_REWRITE_DEPTH
): KExpr<T> = boundedRewrite(depth, expr)
