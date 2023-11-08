package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KApp
import io.ksmt.expr.KArray2Lambda
import io.ksmt.expr.KArray3Lambda
import io.ksmt.expr.KArrayLambda
import io.ksmt.expr.KArrayNLambda
import io.ksmt.expr.KDistinctExpr
import io.ksmt.expr.KEqExpr
import io.ksmt.expr.KExistentialQuantifier
import io.ksmt.expr.KExpr
import io.ksmt.expr.KInterpretedValue
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.expr.KUniversalQuantifier
import io.ksmt.expr.rewrite.KExprUninterpretedDeclCollector
import io.ksmt.expr.transformer.KNonRecursiveTransformerBase
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import io.ksmt.utils.uncheckedCast

open class KExprSimplifier(override val ctx: KContext) :
    KNonRecursiveTransformerBase(),
    KExprSimplifierBase,
    KBoolExprSimplifier,
    KArithExprSimplifier,
    KBvExprSimplifier,
    KFpExprSimplifier,
    KArrayExprSimplifier {

    fun <T : KSort> KContext.preprocess(expr: KEqExpr<T>): KExpr<KBoolSort> =
        if (expr.lhs == expr.rhs) trueExpr else expr

    @Suppress("UNCHECKED_CAST")
    fun <T : KSort> KContext.postRewriteEq(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> {
        if (lhs == rhs) return trueExpr

        return when (lhs.sort) {
            boolSort -> this@KExprSimplifier.simplifyEqBool(lhs.uncheckedCast(), rhs.uncheckedCast())
            intSort -> simplifyEqInt(lhs.uncheckedCast(), rhs.uncheckedCast())
            realSort -> simplifyEqReal(lhs.uncheckedCast(), rhs.uncheckedCast())
            is KBvSort -> simplifyEqBv(lhs as KExpr<KBvSort>, rhs.uncheckedCast())
            is KFpSort -> simplifyEqFp(lhs as KExpr<KFpSort>, rhs.uncheckedCast())
            is KArraySortBase<*> -> simplifyEqArray(lhs as KExpr<KArraySortBase<KSort>>, rhs.uncheckedCast())
            is KUninterpretedSort -> simplifyEqUninterpreted(lhs.uncheckedCast(), rhs.uncheckedCast())
            else -> withExpressionsOrdered(lhs, rhs, ::mkEqNoSimplify)
        }
    }

    override fun <T : KSort> transform(expr: KEqExpr<T>) =
        simplifyExpr(
            expr = expr,
            a0 = expr.lhs,
            a1 = expr.rhs,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteEq(l, r) }
        )

    fun <T : KSort> KContext.preprocess(expr: KDistinctExpr<T>): KExpr<KBoolSort> =
        when (expr.args.size) {
            0, 1 -> trueExpr
            2 -> !(expr.args[0] eq expr.args[1])
            else -> expr
        }

    fun <T : KSort> KContext.postRewriteDistinct(args: List<KExpr<T>>): KExpr<KBoolSort> {
        val distinct = checkAllExpressionsAreDistinct(args)

        return distinct?.expr ?: mkDistinctNoSimplify(args)
    }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>) =
        simplifyExpr(
            expr = expr,
            args = expr.args,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteDistinct(it) }
        )

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
                    && (allExpressionsAreConstants || visitedExprs.all { areDefinitelyDistinct(it, expr) })
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

    // We can skip values transformation since values can't be simplified
    override fun <T : KSort> exprTransformationRequired(expr: KExpr<T>): Boolean =
        expr !is KInterpretedValue<T>

    // Interpreted values can't be simplified.
    override fun <T : KSort> transformValue(expr: KInterpretedValue<T>): KExpr<T> = expr

    override fun <T : KSort, A : KSort> transformApp(expr: KApp<T, A>): KExpr<T> =
        simplifyExpr(expr, expr.args) { args ->
            expr.decl.apply(args)
        }

    // quantified expressions
    fun <D : KSort, R : KSort> KContext.preprocess(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>> = expr

    fun <D : KSort, R : KSort> KContext.postRewriteArrayLambda(
        bound: KDecl<D>,
        simplifiedBody: KExpr<R>
    ): KExpr<KArraySort<D, R>> = simplifyQuantifier(
        bounds = listOf(bound),
        simplifiedBody = simplifiedBody,
        eliminateQuantifier = { body ->
            val sort = mkArraySort(bound.sort, body.sort)
            mkArrayConst(sort, body)
        },
        buildQuantifier = { _, body -> mkArrayLambda(bound, body) }
    )

    override fun <D : KSort, R : KSort> transform(
        expr: KArrayLambda<D, R>
    ): KExpr<KArraySort<D, R>> = simplifyExpr(
        expr = expr,
        a0 = expr.body,
        preprocess = { preprocess(it) },
        simplifier = { body -> postRewriteArrayLambda(expr.indexVarDecl, body) }
    )

    fun <D0 : KSort, D1 : KSort, R : KSort> KContext.preprocess(
        expr: KArray2Lambda<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = expr

    fun <D0 : KSort, D1 : KSort, R : KSort> KContext.postRewriteArrayLambda(
        bound0: KDecl<D0>, bound1: KDecl<D1>,
        simplifiedBody: KExpr<R>
    ): KExpr<KArray2Sort<D0, D1, R>> = simplifyQuantifier(
        bounds = listOf(bound0, bound1),
        simplifiedBody = simplifiedBody,
        eliminateQuantifier = { body ->
            val sort = mkArraySort(bound0.sort, bound1.sort, body.sort)
            mkArrayConst(sort, body)
        },
        buildQuantifier = { _, body -> mkArrayLambda(bound0, bound1, body) }
    )

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Lambda<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = simplifyExpr(
        expr = expr,
        a0 = expr.body,
        preprocess = { preprocess(it) },
        simplifier = { body -> postRewriteArrayLambda(expr.indexVar0Decl, expr.indexVar1Decl, body) }
    )

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> KContext.preprocess(
        expr: KArray3Lambda<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = expr

    fun <D0 : KSort, D1 : KSort, D2: KSort, R : KSort> KContext.postRewriteArrayLambda(
        bound0: KDecl<D0>, bound1: KDecl<D1>, bound2: KDecl<D2>,
        simplifiedBody: KExpr<R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = simplifyQuantifier(
        bounds = listOf(bound0, bound1, bound2),
        simplifiedBody = simplifiedBody,
        eliminateQuantifier = { body ->
            val sort = mkArraySort(bound0.sort, bound1.sort, bound2.sort, body.sort)
            mkArrayConst(sort, body)
        },
        buildQuantifier = { _, body -> mkArrayLambda(bound0, bound1, bound2, body) }
    )


    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Lambda<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = simplifyExpr(
        expr = expr,
        a0 = expr.body,
        preprocess = { preprocess(it) },
        simplifier = { body ->
            postRewriteArrayLambda(expr.indexVar0Decl, expr.indexVar1Decl, expr.indexVar2Decl, body)
        }
    )

    fun <R : KSort> KContext.preprocess(expr: KArrayNLambda<R>): KExpr<KArrayNSort<R>> = expr

    fun <R : KSort> KContext.postRewriteArrayLambda(
        bounds: List<KDecl<*>>,
        simplifiedBody: KExpr<R>
    ): KExpr<KArrayNSort<R>> = simplifyQuantifier(
        bounds = bounds,
        simplifiedBody = simplifiedBody,
        eliminateQuantifier = { body ->
            val sort = mkArrayNSort(bounds.map { it.sort }, body.sort)
            mkArrayConst(sort, body)
        },
        buildQuantifier = { _, body -> mkArrayNLambda(bounds, body) }
    )

    override fun <R : KSort> transform(
        expr: KArrayNLambda<R>
    ): KExpr<KArrayNSort<R>> = simplifyExpr(
        expr = expr,
        a0 = expr.body,
        preprocess = { preprocess(it) },
        simplifier = { body -> postRewriteArrayLambda(expr.indexVarDeclarations, body) }
    )

    fun KContext.preprocess(expr: KExistentialQuantifier): KExpr<KBoolSort> = expr

    fun KContext.postRewriteExistentialQuantifier(
        bounds: List<KDecl<*>>,
        simplifiedBody: KExpr<KBoolSort>
    ): KExpr<KBoolSort> = simplifyQuantifier(
        bounds = bounds,
        simplifiedBody = simplifiedBody,
        eliminateQuantifier = { body -> body },
        buildQuantifier = { simplifiedBounds, body -> mkExistentialQuantifier(body, simplifiedBounds) }
    )

    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.body,
            preprocess = { preprocess(it) },
            simplifier = { body -> postRewriteExistentialQuantifier(expr.bounds, body) }
        )

    fun KContext.preprocess(expr: KUniversalQuantifier): KExpr<KBoolSort> = expr

    fun KContext.postRewriteUniversalQuantifier(
        bounds: List<KDecl<*>>,
        simplifiedBody: KExpr<KBoolSort>
    ): KExpr<KBoolSort> = simplifyQuantifier(
        bounds = bounds,
        simplifiedBody = simplifiedBody,
        eliminateQuantifier = { body -> body },
        buildQuantifier = { simplifiedBounds, body -> mkUniversalQuantifier(body, simplifiedBounds) }
    )

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.body,
            preprocess = { preprocess(it) },
            simplifier = { body -> postRewriteUniversalQuantifier(expr.bounds, body) }
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
    ): KExpr<KBoolSort> = with(ctx) {
        if (lhs is KUninterpretedSortValue && rhs is KUninterpretedSortValue) {
            return (lhs == rhs).expr
        }
        return withExpressionsOrdered(lhs, rhs, ctx::mkEqNoSimplify)
    }

    open fun areDefinitelyDistinctUninterpreted(
        lhs: KExpr<KUninterpretedSort>,
        rhs: KExpr<KUninterpretedSort>
    ): Boolean {
        if (lhs is KUninterpretedSortValue && rhs is KUninterpretedSortValue) {
            return lhs != rhs
        }
        return false
    }


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

        if (rewriteFrame.rewritten is KSimplifierAuxExpr) {
            eraseTransformationResult(rewriteFrame.rewritten)
        }

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
inline fun <T : KSort, P : KExpr<T>, A : KSort> KExprSimplifierBase.simplifyExpr(
    expr: P,
    args: List<KExpr<A>>,
    preprocess: KContext.(P) -> KExpr<T> = { expr },
    crossinline simplifier: KContext.(List<KExpr<A>>) -> KExpr<T>
): KExpr<T> = simplifyExprBase(
    expr,
    { ctx.preprocess(expr) },
    { transformExprAfterTransformed(expr, args) { tArgs -> ctx.simplifier(tArgs) } }
)

/**
 * Specialized version of [simplifyExpr] for expressions which are always
 * rewritten with another expression on the [preprocess] stage.
 * */
inline fun <T : KSort, P : KExpr<T>> KExprSimplifierBase.simplifyExpr(
    expr: P,
    preprocess: KContext.(P) -> KExpr<T>,
): KExpr<T> = simplifyExprBase(
    expr,
    { ctx.preprocess(expr) },
    { error("Always preprocessed") }
)

/**
 * Specialized version of [simplifyExpr] for expressions with a single argument.
 * */
inline fun <T : KSort, P : KExpr<T>, A0 : KSort> KExprSimplifierBase.simplifyExpr(
    expr: P,
    a0: KExpr<A0>,
    preprocess: KContext.(P) -> KExpr<T> = { expr },
    crossinline simplifier: KContext.(KExpr<A0>) -> KExpr<T>
): KExpr<T> = simplifyExprBase(
    expr,
    { ctx.preprocess(expr) },
    { transformExprAfterTransformed(expr, a0) { ta0 -> ctx.simplifier(ta0) } }
)

/**
 * Specialized version of [simplifyExpr] for expressions with two arguments.
 * */
inline fun <T : KSort, P : KExpr<T>, A0 : KSort, A1 : KSort> KExprSimplifierBase.simplifyExpr(
    expr: P,
    a0: KExpr<A0>,
    a1: KExpr<A1>,
    preprocess: KContext.(P) -> KExpr<T> = { expr },
    crossinline simplifier: KContext.(KExpr<A0>, KExpr<A1>) -> KExpr<T>
): KExpr<T> = simplifyExprBase(
    expr,
    { ctx.preprocess(expr) },
    { transformExprAfterTransformed(expr, a0, a1) { ta0, ta1 -> ctx.simplifier(ta0, ta1) } }
)

/**
 * Specialized version of [simplifyExpr] for expressions with three arguments.
 * */
@Suppress("LongParameterList")
inline fun <T : KSort, P: KExpr<T>, A0 : KSort, A1 : KSort, A2 : KSort> KExprSimplifierBase.simplifyExpr(
    expr: P,
    a0: KExpr<A0>,
    a1: KExpr<A1>,
    a2: KExpr<A2>,
    preprocess: KContext.(P) -> KExpr<T> = { expr },
    crossinline simplifier: KContext.(KExpr<A0>, KExpr<A1>, KExpr<A2>) -> KExpr<T>
): KExpr<T> = simplifyExprBase(
    expr,
    { ctx.preprocess(expr) },
    { transformExprAfterTransformed(expr, a0, a1, a2) { ta0, ta1, ta2 -> ctx.simplifier(ta0, ta1, ta2) } }
)

/**
 * Specialized version of [simplifyExpr] for expressions with four arguments.
 * */
@Suppress("LongParameterList")
inline fun <T : KSort, P : KExpr<T>, A0 : KSort, A1 : KSort, A2 : KSort, A3 : KSort> KExprSimplifierBase.simplifyExpr(
    expr: P,
    a0: KExpr<A0>,
    a1: KExpr<A1>,
    a2: KExpr<A2>,
    a3: KExpr<A3>,
    preprocess: KContext.(P) -> KExpr<T> = { expr },
    crossinline simplifier: KContext.(KExpr<A0>, KExpr<A1>, KExpr<A2>, KExpr<A3>) -> KExpr<T>
): KExpr<T> = simplifyExprBase(
    expr,
    { ctx.preprocess(expr) },
    { transformExprAfterTransformed(expr, a0, a1, a2, a3) { ta0, ta1, ta2, ta3 -> ctx.simplifier(ta0, ta1, ta2, ta3) } }
)

/**
 * Specialized version of [simplifyExpr] for expressions with five arguments.
 * */
@Suppress("LongParameterList")
inline fun <
    T : KSort,
    P : KExpr<T>,
    A0 : KSort,
    A1 : KSort,
    A2 : KSort,
    A3 : KSort,
    A4 : KSort
> KExprSimplifierBase.simplifyExpr(
    expr: P,
    a0: KExpr<A0>,
    a1: KExpr<A1>,
    a2: KExpr<A2>,
    a3: KExpr<A3>,
    a4: KExpr<A4>,
    preprocess: KContext.(P) -> KExpr<T> = { expr },
    crossinline simplifier: KContext.(KExpr<A0>, KExpr<A1>, KExpr<A2>, KExpr<A3>, KExpr<A4>) -> KExpr<T>
): KExpr<T> = simplifyExprBase(
    expr,
    { ctx.preprocess(expr) },
    { transformExprAfterTransformed(expr, a0, a1, a2, a3, a4) { ta0, ta1, ta2, ta3, ta4 ->
        ctx.simplifier(ta0, ta1, ta2, ta3, ta4)
    } }
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
