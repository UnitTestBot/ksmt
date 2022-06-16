package org.ksmt.expr.transformer

import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

interface KTransformer {
    fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> = expr
}
