package org.ksmt.solver.yices

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KApp
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import org.ksmt.utils.mkFreshConstDecl
import org.ksmt.utils.uncheckedCast

class KDeclSubstitutor(ctx: KContext) : KNonRecursiveTransformer(ctx) {
    private val substitution = hashMapOf<KDecl<*>, KDecl<*>>()

    fun <T : KSort> substitute(from: KDecl<T>, to: KDecl<T>) {
        substitution[from] = to
    }

    private fun <T : KSort> transformDecl(decl: KDecl<T>): KDecl<T> =
        (substitution[decl].uncheckedCast()) ?: decl

    override fun <T : KSort, A : KSort> transformApp(expr: KApp<T, A>): KExpr<T> =
        transformAppAfterArgsTransformed(expr) { transformedArgs ->
            val transformedDecl = transformDecl(expr.decl)

            ctx.mkApp(transformedDecl, transformedArgs)
        }

    override fun <D : KSort, R : KSort> transform(
        expr: KArrayLambda<D, R>
    ): KExpr<KArraySort<D, R>> = with(expr) {
        transformQuantifierAfterBodyTransformed(body, listOf(indexVarDecl)) { transformedBody, transformedBounds ->
            ctx.mkArrayLambda(transformedBounds.single().uncheckedCast(), transformedBody)
        }
    }

    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> = with(expr) {
        transformQuantifierAfterBodyTransformed(body, bounds) { transformedBody, transformedBounds ->
            ctx.mkExistentialQuantifier(transformedBody, transformedBounds)
        }
    }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> = with(expr) {
        transformQuantifierAfterBodyTransformed(body, bounds) { transformedBody, transformedBounds ->
            ctx.mkUniversalQuantifier(transformedBody, transformedBounds)
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KFunctionAsArray<D, R>): KExpr<KArraySort<D, R>> {
        val transformedFunction = transformDecl(expr.function)

        return ctx.mkFunctionAsArray(transformedFunction as KFuncDecl<R>)
    }

    @Suppress("UNCHECKED_CAST")
    private inline fun<T: KSort, S: KSort> transformQuantifierAfterBodyTransformed(
        body: KExpr<T>,
        bounds: List<KDecl<*>>,
        transformer: (KExpr<T>, List<KDecl<*>>) -> KExpr<S>
    ): KExpr<S> {
        val newSubstitutor = KDeclSubstitutor(ctx).also { currentSubstitutor ->
            val boundsSet = bounds.toHashSet()

            substitution
                .filterNot { (from, _) ->
                    boundsSet.contains(from)
                }.takeIf { it.isNotEmpty() }
                ?.forEach { (from, to) ->
                    currentSubstitutor.substitute(from as KDecl<KSort>, to as KDecl<KSort>)

                    if (boundsSet.contains(to) && !currentSubstitutor.substitution.contains(to))
                        currentSubstitutor.substitute(to, to.sort.mkFreshConstDecl(to.name))
                } ?: return transformer(body, bounds)
        }

        return transformer(newSubstitutor.apply(body), bounds.map { newSubstitutor.transformDecl(it) })
    }
}
