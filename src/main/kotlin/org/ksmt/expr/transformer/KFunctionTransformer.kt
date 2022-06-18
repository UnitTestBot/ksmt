package org.ksmt.expr.transformer

import org.ksmt.expr.KConst
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFunctionApp
import org.ksmt.sort.KSort

interface KFunctionTransformer : KAppTransformer {
    fun <T : KSort> transformFunctionApp(expr: KFunctionApp<T>): KExpr<T> = transformApp(expr)
    fun <T : KSort> transformFunctionApp(expr: KFunctionApp<T>, accept: KFunctionApp<T>.() -> KExpr<T>): KExpr<T> = expr.accept()
    fun <T : KSort> transformConst(expr: KConst<T>): KExpr<T> = transformApp(expr)
    fun <T : KSort> transformConst(expr: KConst<T>, accept: KConst<T>.() -> KExpr<T>): KExpr<T> = expr.accept()
}
