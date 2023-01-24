package org.ksmt.expr.rewrite

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KApp
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.rewrite.KExprUninterpretedDeclCollector.Companion.collectUninterpretedDeclarations
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import org.ksmt.utils.uncheckedCast

open class KExprSubstitutor(ctx: KContext) : KNonRecursiveTransformer(ctx) {
    private val exprExprSubstitution = hashMapOf<KExpr<*>, KExpr<*>>()
    private val declDeclSubstitution = hashMapOf<KDecl<*>, KDecl<*>>()

    /**
     * Substitute every occurrence of `from` in expression `expr` with `to`.
     */
    fun <T : KSort> substitute(from: KExpr<T>, to: KExpr<T>) {
        check(from.sort == to.sort) {
            "Substitution expression sort mismatch: from ${from.sort} to ${to.sort}"
        }
        exprExprSubstitution[from] = to
    }

    /**
     * Substitute every occurrence of declaration `from` in expression `expr` with `to`.
     */
    fun <T : KSort> substitute(from: KDecl<T>, to: KDecl<T>) {
        check(from.sort == to.sort) {
            "Substitution declaration sort mismatch: from ${from.sort} to ${to.sort}"
        }
        declDeclSubstitution[from] = to
    }

    override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> =
        exprExprSubstitution[expr]?.uncheckedCast() ?: expr

    override fun <T : KSort, A : KSort> transformApp(expr: KApp<T, KExpr<A>>): KExpr<T> =
        transformAppAfterArgsTransformed(expr) { transformedArgs ->
            val decl = declDeclSubstitution[expr.decl]?.uncheckedCast() ?: expr.decl
            val transformedApp = decl.apply(transformedArgs)
            return transformExpr(transformedApp)
        }

    override fun <D : KSort, R : KSort> transform(expr: KFunctionAsArray<D, R>): KExpr<KArraySort<D, R>> {
        val declSubstitution = declDeclSubstitution[expr.function]?.uncheckedCast() ?: expr.function
        return transformExpr(ctx.mkFunctionAsArray(declSubstitution))
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>> =
        transformQuantifiedExpression(expr, setOf(expr.indexVarDecl), expr.body) { body, bounds ->
            ctx.mkArrayLambda(bounds.single().uncheckedCast(), body)
        }

    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> =
        transformQuantifiedExpression(expr, expr.bounds.toSet(), expr.body) { body, bounds ->
            ctx.mkExistentialQuantifier(body, bounds.toList())
        }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> =
        transformQuantifiedExpression(expr, expr.bounds.toSet(), expr.body) { body, bounds ->
            ctx.mkUniversalQuantifier(body, bounds.toList())
        }

    private val unprocessedQuantifiers = hashMapOf<KExpr<*>, Pair<Set<KDecl<*>>, KExpr<*>>>()

    /**
     * Resolve quantifier bound vars shadowing.
     *
     * For example, we have an expression of the form
     * (and (f a) (exists (a) (g a)))
     * and we want substitute expression a with b.
     * In subexpression (g a) a is a quantified var and we must not substitute it.
     * So, in our case result must be:
     * (and (f b) (exists (a) (g a)))
     * */
    private inline fun <B : KSort, T: KSort> transformQuantifiedExpression(
        quantifiedExpr: KExpr<T>,
        quantifiedVars: Set<KDecl<*>>,
        body: KExpr<B>,
        crossinline quantifierBuilder: (KExpr<B>, Set<KDecl<*>>) -> KExpr<T>
    ): KExpr<T> {
        val (unshadowedBounds, unshadowedBody) = unprocessedQuantifiers.getOrPut(quantifiedExpr) {
            resolveQuantifierShadowedVars(quantifiedVars, body)
        }
        @Suppress("UNCHECKED_CAST")
        return transformExprAfterTransformed(quantifiedExpr, listOf(unshadowedBody as KExpr<B>)) { (transformedBody) ->
            unprocessedQuantifiers.remove(quantifiedExpr)
            quantifierBuilder(transformedBody, unshadowedBounds)
        }
    }

    private fun <B : KSort> resolveQuantifierShadowedVars(
        quantifiedVars: Set<KDecl<*>>,
        body: KExpr<B>,
    ): Pair<Set<KDecl<*>>, KExpr<*>> {
        val usedDeclarationsList = exprExprSubstitution.flatMap {
            collectUninterpretedDeclarations(it.key) + collectUninterpretedDeclarations(it.value)
        } + declDeclSubstitution.keys + declDeclSubstitution.values
        val usedDeclarations = usedDeclarationsList.toSet()

        val shadowedDeclarations = quantifiedVars.intersect(usedDeclarations)

        if (shadowedDeclarations.isEmpty()) {
            return quantifiedVars to body
        }

        val shadowedBoundsReplacement = shadowedDeclarations.associateWith { it.freshStub() }
        val unshadowedQuantifiedVars = quantifiedVars.mapTo(hashSetOf()) { shadowedBoundsReplacement[it] ?: it }
        val shadowedBoundRemover = KExprSubstitutor(ctx).also { remover ->
            shadowedBoundsReplacement.forEach {
                @Suppress("UNCHECKED_CAST")
                remover.substitute(it.key as KDecl<KSort>, it.value.uncheckedCast())
            }
        }
        val unshadowedBody = shadowedBoundRemover.apply(body)
        return unshadowedQuantifiedVars to unshadowedBody
    }

    private fun <T: KSort> KDecl<T>.freshStub(): KDecl<T> =
        if (this is KFuncDecl<T>) {
            ctx.mkFreshFuncDecl(name, sort, argSorts)
        } else {
            ctx.mkFreshConstDecl(name, sort)
        }
}
