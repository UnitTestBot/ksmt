package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KDistinctExpr
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExpr
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.utils.asExpr

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

    override fun <T : KSort> transform(expr: KEqExpr<T>) = simplifyApp(
        expr = expr,
        preprocess = { if (expr.lhs == expr.rhs) trueExpr else expr }
    ) { (lhs, rhs) ->
        when (val sort = lhs.sort) {
            boolSort -> simplifyEqBool(lhs.asExpr(boolSort), rhs.asExpr(boolSort))
            intSort -> simplifyEqInt(lhs.asExpr(intSort), rhs.asExpr(intSort))
            realSort -> simplifyEqReal(lhs.asExpr(realSort), rhs.asExpr(realSort))
            is KBvSort -> simplifyEqBv(lhs.asExpr(sort), rhs.asExpr(sort))
            is KFpSort -> simplifyEqFp(lhs.asExpr(sort), rhs.asExpr(sort))
            is KArraySort<*, *> -> simplifyEqArray(lhs.asExpr(sort), rhs.asExpr(sort))
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
        // (distinct a a) ==> false
        if (args.toSet().size != args.size) {
            return@simplifyApp falseExpr
        }
        mkDistinct(args)
    }

    override fun <T : KSort> areDefinitelyDistinct(lhs: KExpr<T>, rhs: KExpr<T>): Boolean = with(ctx) {
        if (lhs == rhs) return false
        return when (val sort = lhs.sort) {
            boolSort -> areDefinitelyDistinctBool(lhs.asExpr(boolSort), rhs.asExpr(boolSort))
            intSort -> areDefinitelyDistinctInt(lhs.asExpr(intSort), rhs.asExpr(intSort))
            realSort -> areDefinitelyDistinctReal(lhs.asExpr(realSort), rhs.asExpr(realSort))
            is KBvSort -> areDefinitelyDistinctBv(lhs.asExpr(sort), rhs.asExpr(sort))
            is KFpSort -> areDefinitelyDistinctFp(lhs.asExpr(sort), rhs.asExpr(sort))
            else -> false
        }
    }

    fun <T : KSort> rewrittenOrNull(expr: KExpr<T>): KExpr<T>? {
        val rewritten = rewrittenExpressions.remove(expr) ?: return null
        val result = transformedExpr(rewritten)
            ?: error("Nested rewrite failed")
        return result.asExpr(expr.sort)
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
        needPostRewrite = false
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
