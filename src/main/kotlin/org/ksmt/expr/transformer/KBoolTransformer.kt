package org.ksmt.expr.transformer

import org.ksmt.expr.*
import org.ksmt.sort.KBoolSort

interface KBoolTransformer : KAppTransformer {
    fun transformAnd(expr: KAndExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transformAnd(expr: KAndExpr, accept: KAndExpr.() -> KExpr<KBoolSort>): KExpr<KBoolSort> = expr.accept()

    fun transformOrAfter(expr: KOrExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transformOrBefore(expr: KOrExpr): KExpr<KBoolSort> = expr.transform(this)

    fun transformNot(expr: KNotExpr): KExpr<KBoolSort> {
        val arg = expr.arg.accept(this)
        if (arg == expr.arg) return transformApp(expr)
        return transformApp(mkNot(arg))
    }
}
