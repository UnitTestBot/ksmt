package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KApp
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KDistinctExpr
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KInterpretedValue
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.rewrite.KExprUninterpretedDeclCollector
import org.ksmt.expr.transformer.KNonRecursiveTransformerBase
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.utils.uncheckedCast

open class KExprSimplifier(override val ctx: KContext) :
    KNonRecursiveTransformerBase(),
    KExprSimplifierBase,
    KBoolExprSimplifier,
    KArithExprSimplifier,
    KBvExprSimplifier,
    KFpExprSimplifier,
    KArrayExprSimplifier {

    @Suppress("UNCHECKED_CAST")
    override fun <T : KSort> transform(expr: KEqExpr<T>) = simplifyExpr(
        expr, expr.lhs, expr.rhs,
        preprocess = { if (expr.lhs == expr.rhs) trueExpr else expr }
    ) { lhs, rhs ->
        if (lhs == rhs) return@simplifyExpr trueExpr
        when (lhs.sort) {
            boolSort -> simplifyEqBool(lhs.uncheckedCast(), rhs.uncheckedCast())
            intSort -> simplifyEqInt(lhs.uncheckedCast(), rhs.uncheckedCast())
            realSort -> simplifyEqReal(lhs.uncheckedCast(), rhs.uncheckedCast())
            is KBvSort -> simplifyEqBv(lhs as KExpr<KBvSort>, rhs.uncheckedCast())
            is KFpSort -> simplifyEqFp(lhs as KExpr<KFpSort>, rhs.uncheckedCast())
            is KArraySort<*, *> -> simplifyEqArray(lhs as KExpr<KArraySort<KSort, KSort>>, rhs.uncheckedCast())
            is KUninterpretedSort -> simplifyEqUninterpreted(lhs.uncheckedCast(), rhs.uncheckedCast())
            else -> mkEqNoSimplify(lhs, rhs)
        }
    }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>) = simplifyExpr(
        expr, expr.args,
        preprocess = {
            when (expr.args.size) {
                0, 1 -> trueExpr
                2 -> !(expr.args[0] eq expr.args[1])
                else -> expr
            }
        }
    ) { args ->
        val distinct = checkAllExpressionsAreDistinct(args)
        distinct?.expr ?: mkDistinctNoSimplify(args)
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

            allExpressionsAreConstants = allExpressionsAreConstants && expr is KInterpretedValue<*>

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

    // Interpreted values can't be simplified.
    override fun <T : KSort> transformValue(expr: KInterpretedValue<T>): KExpr<T> = expr

    override fun <T : KSort, A : KSort> transformApp(expr: KApp<T, A>): KExpr<T> =
        simplifyExpr(expr, expr.args) { args ->
            expr.decl.apply(args)
        }

    // quantified expressions
    override fun <D : KSort, R : KSort> transform(
        expr: KArrayLambda<D, R>
    ): KExpr<KArraySort<D, R>> = simplifyExpr(expr, expr.body) { body ->
        simplifyArrayLambda(expr.indexVarDecl, body)
    }

    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.body) { body ->
            simplifyExistentialQuantifier(expr.bounds, body)
        }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.body) { body ->
            simplifyUniversalQuantifier(expr.bounds, body)
        }

    fun <D : KSort, R : KSort> KContext.simplifyArrayLambda(
        bound: KDecl<D>,
        simplifiedBody: KExpr<R>
    ): KExpr<KArraySort<D, R>> = simplifyQuantifier(
        bounds = listOf(element = bound),
        simplifiedBody = simplifiedBody,
        eliminateQuantifier = { body ->
            val sort = mkArraySort(bound.sort, body.sort)
            mkArrayConst(sort, body)
        },
        buildQuantifier = { _, body -> mkArrayLambda(bound, body) }
    )

    fun KContext.simplifyExistentialQuantifier(
        bounds: List<KDecl<*>>,
        simplifiedBody: KExpr<KBoolSort>
    ): KExpr<KBoolSort> = simplifyQuantifier(
        bounds = bounds,
        simplifiedBody = simplifiedBody,
        eliminateQuantifier = { body -> body },
        buildQuantifier = { simplifiedBounds, body -> mkExistentialQuantifier(body, simplifiedBounds) }
    )

    fun KContext.simplifyUniversalQuantifier(
        bounds: List<KDecl<*>>,
        simplifiedBody: KExpr<KBoolSort>
    ): KExpr<KBoolSort> = simplifyQuantifier(
        bounds = bounds,
        simplifiedBody = simplifiedBody,
        eliminateQuantifier = { body -> body },
        buildQuantifier = { simplifiedBounds, body -> mkUniversalQuantifier(body, simplifiedBounds) }
    )

    inline fun <B : KSort, Q : KSort> simplifyQuantifier(
        bounds: List<KDecl<*>>,
        simplifiedBody: KExpr<B>,
        eliminateQuantifier: (KExpr<B>) -> KExpr<Q>,
        buildQuantifier: (List<KDecl<*>>, KExpr<B>) -> KExpr<Q>
    ): KExpr<Q> {
        // Value definitely doesn't contains bound vars
        if (simplifiedBody is KInterpretedValue<B>) {
            return eliminateQuantifier(simplifiedBody)
        }

        val usedVars = KExprUninterpretedDeclCollector.collectUninterpretedDeclarations(simplifiedBody)
        val usedBounds = bounds.intersect(usedVars)

        // Body doesn't depends on bound vars
        if (usedBounds.isEmpty()) {
            return eliminateQuantifier(simplifiedBody)
        }

        return buildQuantifier(usedBounds.toList(), simplifiedBody)
    }

    open fun simplifyEqUninterpreted(
        lhs: KExpr<KUninterpretedSort>,
        rhs: KExpr<KUninterpretedSort>
    ): KExpr<KBoolSort> = ctx.mkEqNoSimplify(lhs, rhs)

    open fun areDefinitelyDistinctUninterpreted(
        lhs: KExpr<KUninterpretedSort>,
        rhs: KExpr<KUninterpretedSort>
    ): Boolean = false


    private class RewriteFrame(
        val original: KExpr<*>,
        val rewritten: KExpr<*>,
        val rewriteDepthBound: Int
    )

    private var rewriteDepth = UNBOUND_REWRITE_DEPTH
    private var currentFrameDepth = UNBOUND_REWRITE_DEPTH
    private val rewriteStack = arrayListOf<RewriteFrame>()

    fun <T : KSort> rewrittenOrNull(expr: KExpr<T>): KExpr<T>? {
        if (rewriteStack.lastOrNull()?.original != expr) return null

        val rewriteFrame = rewriteStack.removeLast()
        rewriteDepth = rewriteFrame.rewriteDepthBound

        val result = transformedExpr(rewriteFrame.rewritten)
            ?: error("Nested rewrite failed")

        return result.uncheckedCast()
    }

    fun postRewrite(original: KExpr<*>, rewritten: KExpr<*>) {
        rewriteStack += RewriteFrame(original, rewritten, rewriteDepth)

        if (currentFrameDepth != UNBOUND_REWRITE_DEPTH) {
            rewriteDepth = minOf(rewriteDepth - 1, currentFrameDepth)
        }

        original.transformAfter(listOf(rewritten))
        markExpressionAsNotTransformed()
    }


    private var needPostRewrite = false

    fun disablePostRewrite() {
        needPostRewrite = false
    }

    fun enablePostRewrite() {
        needPostRewrite = true
    }

    fun postRewriteEnabled(): Boolean = needPostRewrite

    override fun <T : KSort> rewrite(expr: KExpr<T>): KExpr<T> {
        currentFrameDepth = UNBOUND_REWRITE_DEPTH
        needPostRewrite = true
        return expr
    }

    override fun canPerformBoundedRewrite(): Boolean = rewriteDepth > 0

    override fun <T : KSort> boundedRewrite(allowedDepth: Int, expr: KExpr<T>): KExpr<T> {
        check(canPerformBoundedRewrite()) { "Bound rewrite depth limit reached" }
        currentFrameDepth = allowedDepth
        needPostRewrite = true
        return expr
    }

    companion object {
        private const val UNBOUND_REWRITE_DEPTH = 10000
    }
}

/**
 * Simplify an expression.
 * [preprocess] Rewrite an expression before simplification of an [args] (top-down).
 * [simplifier] Rewrite an expression after [args] simplification (bottom-up).
 *
 * See [simplifyExprBase] for the details.
 * */
inline fun <T : KSort, A : KSort> KExprSimplifierBase.simplifyExpr(
    expr: KExpr<T>,
    args: List<KExpr<A>>,
    preprocess: KContext.() -> KExpr<T> = { expr },
    crossinline simplifier: KContext.(List<KExpr<A>>) -> KExpr<T>
): KExpr<T> = simplifyExprBase(
    expr,
    { ctx.preprocess() },
    { transformExprAfterTransformed(expr, args) { tArgs -> ctx.simplifier(tArgs) } }
)

/**
 * Specialized version of [simplifyExpr] for expressions which are always
 * rewritten with another expression on the [preprocess] stage.
 * */
inline fun <T : KSort> KExprSimplifierBase.simplifyExpr(
    expr: KExpr<T>,
    preprocess: KContext.() -> KExpr<T>,
): KExpr<T> = simplifyExprBase(
    expr,
    { ctx.preprocess() },
    { error("Always preprocessed") }
)

/**
 * Specialized version of [simplifyExpr] for expressions with a single argument.
 * */
inline fun <T : KSort, A0 : KSort> KExprSimplifierBase.simplifyExpr(
    expr: KExpr<T>,
    a0: KExpr<A0>,
    preprocess: KContext.() -> KExpr<T> = { expr },
    crossinline simplifier: KContext.(KExpr<A0>) -> KExpr<T>
): KExpr<T> = simplifyExprBase(
    expr,
    { ctx.preprocess() },
    { transformExprAfterTransformed(expr, a0) { ta0 -> ctx.simplifier(ta0) } }
)

/**
 * Specialized version of [simplifyExpr] for expressions with two arguments.
 * */
inline fun <T : KSort, A0 : KSort, A1 : KSort> KExprSimplifierBase.simplifyExpr(
    expr: KExpr<T>,
    a0: KExpr<A0>,
    a1: KExpr<A1>,
    preprocess: KContext.() -> KExpr<T> = { expr },
    crossinline simplifier: KContext.(KExpr<A0>, KExpr<A1>) -> KExpr<T>
): KExpr<T> = simplifyExprBase(
    expr,
    { ctx.preprocess() },
    { transformExprAfterTransformed(expr, a0, a1) { ta0, ta1 -> ctx.simplifier(ta0, ta1) } }
)

/**
 * Specialized version of [simplifyExpr] for expressions with three arguments.
 * */
@Suppress("LongParameterList")
inline fun <T : KSort, A0 : KSort, A1 : KSort, A2 : KSort> KExprSimplifierBase.simplifyExpr(
    expr: KExpr<T>,
    a0: KExpr<A0>,
    a1: KExpr<A1>,
    a2: KExpr<A2>,
    preprocess: KContext.() -> KExpr<T> = { expr },
    crossinline simplifier: KContext.(KExpr<A0>, KExpr<A1>, KExpr<A2>) -> KExpr<T>
): KExpr<T> = simplifyExprBase(
    expr,
    { ctx.preprocess() },
    { transformExprAfterTransformed(expr, a0, a1, a2) { ta0, ta1, ta2 -> ctx.simplifier(ta0, ta1, ta2) } }
)

/**
 * Specialized version of [simplifyExpr] for expressions with four arguments.
 * */
@Suppress("LongParameterList")
inline fun <T : KSort, A0 : KSort, A1 : KSort, A2 : KSort, A3 : KSort> KExprSimplifierBase.simplifyExpr(
    expr: KExpr<T>,
    a0: KExpr<A0>,
    a1: KExpr<A1>,
    a2: KExpr<A2>,
    a3: KExpr<A3>,
    preprocess: KContext.() -> KExpr<T> = { expr },
    crossinline simplifier: KContext.(KExpr<A0>, KExpr<A1>, KExpr<A2>, KExpr<A3>) -> KExpr<T>
): KExpr<T> = simplifyExprBase(
    expr,
    { ctx.preprocess() },
    { transformExprAfterTransformed(expr, a0, a1, a2, a3) { ta0, ta1, ta2, ta3 -> ctx.simplifier(ta0, ta1, ta2, ta3) } }
)

/**
 * Simplify an expression.
 * 1. Preprocess. Rewrite an expression before simplification of an arguments (top-down).
 * 2. Simplify. Rewrite an expression after arguments simplification (bottom-up).
 * 3. Post rewrite. Perform a simplification of a simplification result.
 * */
inline fun <T : KSort> KExprSimplifierBase.simplifyExprBase(
    expr: KExpr<T>,
    preprocess: KExprSimplifier.() -> KExpr<T>,
    simplify: KExprSimplifier.() -> KExpr<T>
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

    val preprocessed = preprocess()
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
    val transformed = simplify()

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
