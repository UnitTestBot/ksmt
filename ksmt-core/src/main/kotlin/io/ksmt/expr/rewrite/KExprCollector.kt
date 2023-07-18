package io.ksmt.expr.rewrite

import io.ksmt.KContext
import io.ksmt.expr.*
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.sort.KSort

open class KExprCollector(ctx: KContext, private val predicate: (KExpr<*>) -> Boolean):
    KNonRecursiveTransformer(ctx)
{
    private val exprCollected = hashSetOf<KExpr<*>>()

    override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> {
        if (predicate(expr))
            exprCollected += expr
        return super.transformExpr(expr)
    }

    companion object{
        fun collectDeclarations(expr: KExpr<*>, predicate: (KExpr<*>) -> Boolean): Set<KExpr<*>> =
            KExprCollector(expr.ctx, predicate)
                .apply { apply(expr) }
                .exprCollected
    }
}
