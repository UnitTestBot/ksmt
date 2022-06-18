package org.ksmt.expr.transformer

import org.ksmt.expr.KAndExpr
import org.ksmt.expr.KExpr
import org.ksmt.expr.KOrExpr
import org.ksmt.sort.KBoolSort

interface KBoolTransformer : KAppTransformer {
    fun transformAnd(expr: KAndExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transformAnd(expr: KAndExpr, accept: KAndExpr.() -> KExpr<KBoolSort>): KExpr<KBoolSort> = expr.accept()
    fun transformOr(expr: KOrExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transformOr(expr: KOrExpr, accept: KOrExpr.() -> KExpr<KBoolSort>): KExpr<KBoolSort> = expr.accept()
}
