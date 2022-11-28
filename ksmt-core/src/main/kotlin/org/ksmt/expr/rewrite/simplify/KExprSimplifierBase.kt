package org.ksmt.expr.rewrite.simplify

import org.ksmt.expr.KExpr
import org.ksmt.expr.transformer.KTransformer
import org.ksmt.sort.KSort

interface KExprSimplifierBase : KTransformer {
    fun <T : KSort> areDefinitelyDistinct(lhs: KExpr<T>, rhs: KExpr<T>): Boolean
    fun <T : KSort> rewrite(expr: KExpr<T>): KExpr<T>
}
