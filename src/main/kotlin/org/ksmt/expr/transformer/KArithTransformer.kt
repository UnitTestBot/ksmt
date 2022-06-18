package org.ksmt.expr.transformer

import org.ksmt.expr.KAddArithExpr
import org.ksmt.expr.KExpr
import org.ksmt.sort.KArithSort

interface KArithTransformer : KAppTransformer {
    fun <T : KArithSort<T>> transformArithAdd(expr: KAddArithExpr<T>): KExpr<T> = transformApp(expr)
    fun <T : KArithSort<T>> transformArithAdd(expr: KAddArithExpr<T>, accept: KAddArithExpr<T>.() -> KExpr<T>): KExpr<T> = expr.accept()
}
