package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KDistinctExpr
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExpr
import org.ksmt.expr.KInterpretedConstant
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.utils.uncheckedCast

open class KExprSimplifier(ctx: KContext) :
    KNonRecursiveTransformer(ctx),
    KExprSimplifierBase,
    KBoolExprSimplifier,
    KArithExprSimplifier,
    KBvExprSimplifier,
    KFpExprSimplifier,
    KArrayExprSimplifier {

    private var needPostRewrite = false
    private val rewrittenExpressions = hashMapOf<KExpr<*>, KExpr<*>>()

    @Suppress("UNCHECKED_CAST")
    override fun <T : KSort> transform(expr: KEqExpr<T>) = simplifyApp(
        expr = expr,
        preprocess = { if (expr.lhs == expr.rhs) trueExpr else expr }
    ) { (lhs, rhs) ->
        if (lhs == rhs) return@simplifyApp trueExpr
        when (lhs.sort) {
            boolSort -> simplifyEqBool(lhs.uncheckedCast(), rhs.uncheckedCast())
            intSort -> simplifyEqInt(lhs.uncheckedCast(), rhs.uncheckedCast())
            realSort -> simplifyEqReal(lhs.uncheckedCast(), rhs .uncheckedCast())
            is KBvSort -> simplifyEqBv(lhs as KExpr<KBvSort>, rhs.uncheckedCast() )
            is KFpSort -> simplifyEqFp(lhs as KExpr<KFpSort>, rhs.uncheckedCast() )
            is KArraySort<*, *> -> simplifyEqArray(lhs as KExpr<KArraySort<KSort, KSort>>, rhs.uncheckedCast())
            is KUninterpretedSort -> simplifyEqUninterpreted(lhs.uncheckedCast(), rhs.uncheckedCast())
            else -> mkEq(lhs, rhs)
        }
    }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>) = simplifyApp(
        expr = expr,
        preprocess = {
            when (expr.args.size) {
                0, 1 -> trueExpr
                2 -> !(expr.args[0] eq expr.args[1])
                else -> expr
            }
        }
    ) { args ->
        val distinct = checkAllExpressionsAreDistinct(args)
        distinct?.expr ?: mkDistinct(args)
    }

    private fun <T : KSort> checkAllExpressionsAreDistinct(expressions: List<KExpr<T>>): Boolean? {
        val visitedExprs = hashSetOf<KExpr<T>>()
        var allDistinct = true
        var allExpressionsAreConstants = true

        for (expr in expressions) {
            // (distinct a a) ==> false
            if (!visitedExprs.add(expr)) {
                return false
            }

            allExpressionsAreConstants = allExpressionsAreConstants && expr is KInterpretedConstant

            /**
             *  Check all previously visited expressions and current expression are distinct.
             *  Don't check if all expressions are constants as they are trivially comparable.
             *  */
            allDistinct = allDistinct
                    && !allExpressionsAreConstants
                    && visitedExprs.all { areDefinitelyDistinct(it, expr) }
        }

        if (allDistinct) return true
        return null
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : KSort> areDefinitelyDistinct(lhs: KExpr<T>, rhs: KExpr<T>): Boolean = with(ctx) {
        if (lhs == rhs) return false
        return when (lhs.sort) {
            boolSort -> areDefinitelyDistinctBool(lhs.uncheckedCast(), rhs.uncheckedCast())
            intSort -> areDefinitelyDistinctInt(lhs.uncheckedCast(), rhs.uncheckedCast())
            realSort -> areDefinitelyDistinctReal(lhs.uncheckedCast(), rhs.uncheckedCast())
            is KBvSort -> areDefinitelyDistinctBv(lhs as KExpr<KBvSort>, rhs.uncheckedCast())
            is KFpSort -> areDefinitelyDistinctFp(lhs as KExpr<KFpSort>, rhs.uncheckedCast())
            is KUninterpretedSort -> areDefinitelyDistinctUninterpreted(lhs.uncheckedCast(), rhs.uncheckedCast())
            else -> false
        }
    }

    open fun simplifyEqUninterpreted(
        lhs: KExpr<KUninterpretedSort>,
        rhs: KExpr<KUninterpretedSort>
    ): KExpr<KBoolSort> = ctx.mkEq(lhs, rhs)

    open fun areDefinitelyDistinctUninterpreted(
        lhs: KExpr<KUninterpretedSort>,
        rhs: KExpr<KUninterpretedSort>
    ): Boolean = false

    fun <T : KSort> rewrittenOrNull(expr: KExpr<T>): KExpr<T>? {
        val rewritten = rewrittenExpressions.remove(expr) ?: return null
        val result = transformedExpr(rewritten)
            ?: error("Nested rewrite failed")
        return result.uncheckedCast()
    }

    fun postRewrite(original: KExpr<*>, rewritten: KExpr<*>) {
        rewrittenExpressions[original] = rewritten
        original.transformAfter(listOf(rewritten))
        markExpressionAsNotTransformed()
    }

    fun disablePostRewrite() {
        needPostRewrite = false
    }

    fun enablePostRewrite() {
        needPostRewrite = true
    }

    fun postRewriteEnabled(): Boolean = needPostRewrite

    override fun <T : KSort> rewrite(expr: KExpr<T>): KExpr<T> {
        needPostRewrite = true
        return expr
    }
}

/**
 * Simplify an expression.
 * 1. Preprocess. Rewrite an expression before simplification of an arguments (top-down).
 * 2. Simplify. Rewrite an expression after arguments simplification (bottom-up).
 * 3. Post rewrite. Perform a simplification of a simplification result.
 * */
inline fun <T : KSort, A : KSort> KExprSimplifierBase.simplifyApp(
    expr: KApp<T, KExpr<A>>,
    preprocess: KContext.() -> KExpr<T> = { expr },
    crossinline simplifier: KContext.(List<KExpr<A>>) -> KExpr<T>
): KExpr<T> {
    this as KExprSimplifier

    val rewritten = rewrittenOrNull(expr)

    if (rewritten != null) {
        /**
         * Expression has already been simplified and replaced with [rewritten].
         * [rewritten] has also been simplified.
         * */
        return rewritten
    }

    enablePostRewrite()

    val preprocessed = ctx.preprocess()
    if (preprocessed != expr) {
        /**
         * Expression has been rewritten to another expression [preprocessed].
         * Simplify [preprocessed] and replace current expression with simplification result.
         * */
        postRewrite(expr, preprocessed)
        return expr
    }

    disablePostRewrite()

    // Simplify
    val transformed = transformAppAfterArgsTransformed(expr) { args -> ctx.simplifier(args) }

    if (transformed != expr && postRewriteEnabled()) {
        /**
         * Expression was simplified to another expression [transformed] and
         * post rewrite was requested for a [transformed].
         * Simplify [transformed] and replace current expression with simplification result.
         * */
        postRewrite(expr, transformed)
        return expr
    }

    return transformed
}
