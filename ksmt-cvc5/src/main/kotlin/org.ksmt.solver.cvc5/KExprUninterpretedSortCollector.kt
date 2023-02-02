package org.ksmt.solver.cvc5

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort

internal class KExprUninterpretedSortCollector(ctx: KContext) : KNonRecursiveTransformer(ctx) {
    private val uninterpretedSorts = hashSetOf<KUninterpretedSort>()

    override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> {
        if (expr.sort is KUninterpretedSort) uninterpretedSorts += expr.sort as KUninterpretedSort
        return super.transformExpr(expr)
    }

    companion object {
        fun collectUninterpretedSorts(expr: KExpr<*>): Set<KUninterpretedSort> =
            KExprUninterpretedSortCollector(expr.ctx)
                .apply { apply(expr) }
                .uninterpretedSorts
    }
}