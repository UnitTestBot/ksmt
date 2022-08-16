package org.ksmt.expr.rewrite

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.transformer.KNonRecursiveAppTransformer
import org.ksmt.sort.KSort

/* Substitute every occurrence of `from` in expression `expr` with `to`.
* */
class KExprSubstitutor(ctx: KContext) : KNonRecursiveAppTransformer(ctx) {
    private val substitution = hashMapOf<KExpr<*>, KExpr<*>>()
    fun <T : KSort> substitute(from: KExpr<T>, to: KExpr<T>) {
        substitution[from] = to
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> =
        (substitution[expr] as? KExpr<T>) ?: expr
}
