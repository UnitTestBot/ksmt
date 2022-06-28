package org.ksmt.expr.rewrite

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KTransformer
import org.ksmt.sort.KSort

/* Substitute every occurrence of `from` in expression `expr` with `to`.
* */
class KExprSubstitutor(override val ctx: KContext) : KTransformer {
    private val substitution = hashMapOf<KExpr<*>, KExpr<*>>()
    fun <T : KSort> substitute(from: KExpr<T>, to: KExpr<T>) {
        substitution[from] = to
    }

    fun <T : KSort> apply(expr: KExpr<T>): KExpr<T> = expr.accept(this)

    @Suppress("UNCHECKED_CAST")
    override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> =
        (substitution[expr] as? KExpr<T>) ?: expr
}
