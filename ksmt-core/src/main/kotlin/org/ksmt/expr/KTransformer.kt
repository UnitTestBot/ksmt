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
    fun <T: KBvSort> transform(expr: KBvNotExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvReductionAndExpr<T>): KExpr<KBv1Sort> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvReductionOrExpr<T>): KExpr<KBv1Sort> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvAndExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvOrExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvXorExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvNAndExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvNorExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvXNorExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvNegationExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvAddExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvSubExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvMulExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvUnsignedDivExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvSignedDivExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvUnsignedRemExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvSignedRemExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvSignedModExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvUnsignedLessExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvSignedLessExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvSignedGreaterExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun transform(expr: KBvConcatExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvExtractExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvSignExtensionExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvZeroExtensionExpr): KExpr<KBvSort> = transformApp(expr)
    fun transform(expr: KBvRepeatExpr): KExpr<KBvSort> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvShiftLeftExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvArithShiftRightExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvRotateLeftExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvRotateRightExpr<T>): KExpr<T> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>): KExpr<T> = transformApp(expr)
    fun transform(expr: KBv2IntExpr): KExpr<KIntSort> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvAddNoOverflowExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvSubNoOverflowExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvDivNoOverflowExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvNegNoOverflowExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvMulNoOverflowExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    fun <T: KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>): KExpr<KBoolSort> = transformApp(expr)

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
