package org.ksmt.expr

import org.ksmt.sort.*

interface KTransformer {
    fun transform(expr: KExpr<*>): Any = error("transformer is not implemented for expr $expr")
    fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> = expr

    // function transformers
    fun <T : KSort> transform(expr: KFunctionApp<T>) = transformApp(expr)
    fun <T : KSort> transform(expr: KConst<T>) = transformApp(expr)
    fun <T : KSort> transformApp(expr: KApp<T, *>): KExpr<T> {
        val args = expr.args.map { it.accept(this) }
        if (args == expr.args) return transformExpr(expr)
        return transformExpr(mkApp(expr.decl, args))
    }

    // bool transformers
    fun transform(expr: KAndExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KOrExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KNotExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KTrue): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KFalse): KExpr<KBoolSort> = transformApp(expr)
    fun <T : KSort> transform(expr: KEqExpr<T>): KExpr<KBoolSort> = transformApp(expr)

    // bit-vec transformers
    fun <T : KBVSort<KBVSize>> transform(expr: BitVecExpr<T>): KExpr<T> = transformApp(expr)

    // array transformers
    fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>): KExpr<KArraySort<D, R>> = transformApp(expr)
    fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> = transformApp(expr)

    // arith transformers
    fun <T : KArithSort<T>> transform(expr: KAddArithExpr<T>): KExpr<T> = transformApp(expr)
    fun <T : KArithSort<T>> transform(expr: KMulArithExpr<T>): KExpr<T> = transformApp(expr)
    fun <T : KArithSort<T>> transform(expr: KSubArithExpr<T>): KExpr<T> = transformApp(expr)
    fun <T : KArithSort<T>> transform(expr: KUnaryMinusArithExpr<T>): KExpr<T> = transformApp(expr)
    fun <T : KArithSort<T>> transform(expr: KDivArithExpr<T>): KExpr<T> = transformApp(expr)
    fun <T : KArithSort<T>> transform(expr: KPowerArithExpr<T>): KExpr<T> = transformApp(expr)
    fun <T : KArithSort<T>> transform(expr: KLtArithExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T : KArithSort<T>> transform(expr: KLeArithExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T : KArithSort<T>> transform(expr: KGtArithExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T : KArithSort<T>> transform(expr: KGeArithExpr<T>): KExpr<KBoolSort> = transformApp(expr)

    // integer transformers
    fun transform(expr: KModIntExpr): KExpr<KIntSort> = transformApp(expr)
    fun transform(expr: KRemIntExpr): KExpr<KIntSort> = transformApp(expr)
    fun transform(expr: KToRealIntExpr): KExpr<KRealSort> = transformApp(expr)
    fun transformIntNum(expr: KIntNumExpr): KExpr<KIntSort> = transformApp(expr)
    fun transform(expr: KInt32NumExpr): KExpr<KIntSort> = transformIntNum(expr)
    fun transform(expr: KInt64NumExpr): KExpr<KIntSort> = transformIntNum(expr)
    fun transform(expr: KIntBigNumExpr): KExpr<KIntSort> = transformIntNum(expr)

    // real transformers
    fun transform(expr: KToIntRealExpr): KExpr<KIntSort> = transformApp(expr)
    fun transform(expr: KIsIntRealExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KRealNumExpr): KExpr<KRealSort> = transformApp(expr)

    // quantifier transformers
    fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> {
        val body = expr.body.accept(this)
        if (body == expr.body) return transformExpr(expr)
        return transformExpr(mkExistentialQuantifier(body, expr.bounds))
    }

    fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> {
        val body = expr.body.accept(this)
        if (body == expr.body) return transformExpr(expr)
        return transformExpr(mkUniversalQuantifier(body, expr.bounds))
    }
}