package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KBv16Sort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBv64Sort
import org.ksmt.sort.KBv8Sort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort


interface KTransformer {
    val ctx: KContext
    fun transform(expr: KExpr<*>): Any = error("transformer is not implemented for expr $expr")
    fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> = expr

    // function transformers
    fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> = transformApp(expr)
    fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = transform(expr as KFunctionApp<T>)
    fun <T : KSort> transformApp(expr: KApp<T, *>): KExpr<T> = with(ctx) {
        val args = expr.args.map { it.accept(this@KTransformer) }
        if (args == expr.args) return transformExpr(expr)
        return transformExpr(mkApp(expr.decl, args))
    }

    // bool transformers
    fun transform(expr: KAndExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KOrExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KNotExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KImpliesExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KXorExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KTrue): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KFalse): KExpr<KBoolSort> = transformApp(expr)
    fun <T : KSort> transform(expr: KEqExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T : KSort> transform(expr: KDistinctExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T : KSort> transform(expr: KIteExpr<T>): KExpr<T> = transformApp(expr)

    // bit-vec transformers
    fun <T : KBvSort> transformBitVecValue(expr: KBitVecValue<T>): KExpr<T> = transformApp(expr)
    fun transform(expr: KBitVec1Value): KExpr<KBv1Sort> = transformBitVecValue(expr)
    fun transform(expr: KBitVec8Value): KExpr<KBv8Sort> = transformBitVecValue(expr)
    fun transform(expr: KBitVec16Value): KExpr<KBv16Sort> = transformBitVecValue(expr)
    fun transform(expr: KBitVec32Value): KExpr<KBv32Sort> = transformBitVecValue(expr)
    fun transform(expr: KBitVec64Value): KExpr<KBv64Sort> = transformBitVecValue(expr)
    fun transform(expr: KBitVecCustomValue): KExpr<KBvSort> = transformBitVecValue(expr)

    // bit-vec expressions transformers
    fun transform(expr: KBvNotExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvReductionAndExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvReductionOrExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvAndExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvOrExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvXorExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvNAndExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvNorExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvXNorExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvNegationExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvAddExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvSubExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvMulExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvUnsignedDivExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvSignedDivExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvUnsignedRemExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvSignedRemExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvSignedModExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvUnsignedLessExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KBvSignedLessExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KBvUnsignedLessOrEqualExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KBvSignedLessOrEqualExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KBvUnsignedGreaterOrEqualExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KBvSignedGreaterOrEqualExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KBvUnsignedGreaterExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KBvSignedGreaterExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KConcatExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KExtractExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KSignExtensionExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KZeroExtensionExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KRepeatExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvShiftLeftExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvLogicalShiftRightExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvArithShiftRightExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvRotateLeftExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvRotateRightExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBv2IntExpr): KExpr<KIntSort> = transformApp(expr)
    fun transform(expr: KBvAddNoOverflowExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KBvAddNoUnderflowExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KBvSubNoOverflowExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KBvSubNoUnderflowExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KBvDivNoOverflowExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KBvNegNoOverflowExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KBvMulNoOverflowExpr): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KBvMulNoUnderflowExpr): KExpr<KBoolSort> = transformApp(expr)

    // array transformers
    fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>): KExpr<KArraySort<D, R>> = transformApp(expr)
    fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> = transformApp(expr)
    fun <D : KSort, R : KSort> transform(expr: KArrayConst<D, R>): KExpr<KArraySort<D, R>> = transformApp(expr)
    fun <D : KSort, R : KSort> transform(expr: KFunctionAsArray<D, R>): KExpr<KArraySort<D, R>> = transformExpr(expr)
    fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>> = with(ctx) {
        val body = expr.body.accept(this@KTransformer)
        if (body == expr.body) return transformExpr(expr)
        return transformExpr(mkArrayLambda(expr.indexVarDecl, body))
    }

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
    fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> = with(ctx) {
        val body = expr.body.accept(this@KTransformer)
        if (body == expr.body) return transformExpr(expr)
        return transformExpr(mkExistentialQuantifier(body, expr.bounds))
    }

    fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> = with(ctx) {
        val body = expr.body.accept(this@KTransformer)
        if (body == expr.body) return transformExpr(expr)
        return transformExpr(mkUniversalQuantifier(body, expr.bounds))
    }
}
