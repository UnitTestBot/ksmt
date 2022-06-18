package org.ksmt.expr.transformer

import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

interface KAppTransformer : KTransformer {
    fun <T : KSort> transformApp(expr: KApp<T, *>): KExpr<T> = transformExpr(expr)
    fun <T : KSort> transformApp(expr: KApp<T, *>, accept: KApp<T, *>.() -> KExpr<T>): KExpr<T> = expr.accept()
}
