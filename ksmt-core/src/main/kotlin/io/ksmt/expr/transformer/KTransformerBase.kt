package io.ksmt.expr.transformer

import io.ksmt.expr.KAddArithExpr
import io.ksmt.expr.KAndBinaryExpr
import io.ksmt.expr.KAndExpr
import io.ksmt.expr.KArray2Lambda
import io.ksmt.expr.KArray2Select
import io.ksmt.expr.KArray2Store
import io.ksmt.expr.KArray3Lambda
import io.ksmt.expr.KArray3Select
import io.ksmt.expr.KArray3Store
import io.ksmt.expr.KArrayConst
import io.ksmt.expr.KArrayLambda
import io.ksmt.expr.KArrayNLambda
import io.ksmt.expr.KArrayNSelect
import io.ksmt.expr.KArrayNStore
import io.ksmt.expr.KArraySelect
import io.ksmt.expr.KArrayStore
import io.ksmt.expr.KBitVec16Value
import io.ksmt.expr.KBitVec1Value
import io.ksmt.expr.KBitVec32Value
import io.ksmt.expr.KBitVec64Value
import io.ksmt.expr.KBitVec8Value
import io.ksmt.expr.KBitVecCustomValue
import io.ksmt.expr.KBv2IntExpr
import io.ksmt.expr.KBvAddExpr
import io.ksmt.expr.KBvAddNoOverflowExpr
import io.ksmt.expr.KBvAddNoUnderflowExpr
import io.ksmt.expr.KBvAndExpr
import io.ksmt.expr.KBvArithShiftRightExpr
import io.ksmt.expr.KBvConcatExpr
import io.ksmt.expr.KBvDivNoOverflowExpr
import io.ksmt.expr.KBvExtractExpr
import io.ksmt.expr.KBvLogicalShiftRightExpr
import io.ksmt.expr.KBvMulExpr
import io.ksmt.expr.KBvMulNoOverflowExpr
import io.ksmt.expr.KBvMulNoUnderflowExpr
import io.ksmt.expr.KBvNAndExpr
import io.ksmt.expr.KBvNegNoOverflowExpr
import io.ksmt.expr.KBvNegationExpr
import io.ksmt.expr.KBvNorExpr
import io.ksmt.expr.KBvNotExpr
import io.ksmt.expr.KBvOrExpr
import io.ksmt.expr.KBvReductionAndExpr
import io.ksmt.expr.KBvReductionOrExpr
import io.ksmt.expr.KBvRepeatExpr
import io.ksmt.expr.KBvRotateLeftExpr
import io.ksmt.expr.KBvRotateLeftIndexedExpr
import io.ksmt.expr.KBvRotateRightExpr
import io.ksmt.expr.KBvRotateRightIndexedExpr
import io.ksmt.expr.KBvShiftLeftExpr
import io.ksmt.expr.KBvSignExtensionExpr
import io.ksmt.expr.KBvSignedDivExpr
import io.ksmt.expr.KBvSignedGreaterExpr
import io.ksmt.expr.KBvSignedGreaterOrEqualExpr
import io.ksmt.expr.KBvSignedLessExpr
import io.ksmt.expr.KBvSignedLessOrEqualExpr
import io.ksmt.expr.KBvSignedModExpr
import io.ksmt.expr.KBvSignedRemExpr
import io.ksmt.expr.KBvSubExpr
import io.ksmt.expr.KBvSubNoOverflowExpr
import io.ksmt.expr.KBvSubNoUnderflowExpr
import io.ksmt.expr.KBvToFpExpr
import io.ksmt.expr.KBvUnsignedDivExpr
import io.ksmt.expr.KBvUnsignedGreaterExpr
import io.ksmt.expr.KBvUnsignedGreaterOrEqualExpr
import io.ksmt.expr.KBvUnsignedLessExpr
import io.ksmt.expr.KBvUnsignedLessOrEqualExpr
import io.ksmt.expr.KBvUnsignedRemExpr
import io.ksmt.expr.KBvXNorExpr
import io.ksmt.expr.KBvXorExpr
import io.ksmt.expr.KBvZeroExtensionExpr
import io.ksmt.expr.KConst
import io.ksmt.expr.KDistinctExpr
import io.ksmt.expr.KDivArithExpr
import io.ksmt.expr.KEqExpr
import io.ksmt.expr.KExistentialQuantifier
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFalse
import io.ksmt.expr.KFp128Value
import io.ksmt.expr.KFp16Value
import io.ksmt.expr.KFp32Value
import io.ksmt.expr.KFp64Value
import io.ksmt.expr.KFpAbsExpr
import io.ksmt.expr.KFpAddExpr
import io.ksmt.expr.KFpCustomSizeValue
import io.ksmt.expr.KFpDivExpr
import io.ksmt.expr.KFpEqualExpr
import io.ksmt.expr.KFpFromBvExpr
import io.ksmt.expr.KFpFusedMulAddExpr
import io.ksmt.expr.KFpGreaterExpr
import io.ksmt.expr.KFpGreaterOrEqualExpr
import io.ksmt.expr.KFpIsInfiniteExpr
import io.ksmt.expr.KFpIsNaNExpr
import io.ksmt.expr.KFpIsNegativeExpr
import io.ksmt.expr.KFpIsNormalExpr
import io.ksmt.expr.KFpIsPositiveExpr
import io.ksmt.expr.KFpIsSubnormalExpr
import io.ksmt.expr.KFpIsZeroExpr
import io.ksmt.expr.KFpLessExpr
import io.ksmt.expr.KFpLessOrEqualExpr
import io.ksmt.expr.KFpMaxExpr
import io.ksmt.expr.KFpMinExpr
import io.ksmt.expr.KFpMulExpr
import io.ksmt.expr.KFpNegationExpr
import io.ksmt.expr.KFpRemExpr
import io.ksmt.expr.KFpRoundToIntegralExpr
import io.ksmt.expr.KFpRoundingModeExpr
import io.ksmt.expr.KFpSqrtExpr
import io.ksmt.expr.KFpSubExpr
import io.ksmt.expr.KFpToBvExpr
import io.ksmt.expr.KFpToFpExpr
import io.ksmt.expr.KFpToIEEEBvExpr
import io.ksmt.expr.KFpToRealExpr
import io.ksmt.expr.KFunctionApp
import io.ksmt.expr.KFunctionAsArray
import io.ksmt.expr.KGeArithExpr
import io.ksmt.expr.KGtArithExpr
import io.ksmt.expr.KImpliesExpr
import io.ksmt.expr.KInt32NumExpr
import io.ksmt.expr.KInt64NumExpr
import io.ksmt.expr.KIntBigNumExpr
import io.ksmt.expr.KIsIntRealExpr
import io.ksmt.expr.KIteExpr
import io.ksmt.expr.KLeArithExpr
import io.ksmt.expr.KLtArithExpr
import io.ksmt.expr.KModIntExpr
import io.ksmt.expr.KMulArithExpr
import io.ksmt.expr.KNotExpr
import io.ksmt.expr.KOrBinaryExpr
import io.ksmt.expr.KOrExpr
import io.ksmt.expr.KPowerArithExpr
import io.ksmt.expr.KRealNumExpr
import io.ksmt.expr.KRealToFpExpr
import io.ksmt.expr.KRemIntExpr
import io.ksmt.expr.KSubArithExpr
import io.ksmt.expr.KToIntRealExpr
import io.ksmt.expr.KToRealIntExpr
import io.ksmt.expr.KTrue
import io.ksmt.expr.KUnaryMinusArithExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.expr.KUniversalQuantifier
import io.ksmt.expr.KXorExpr
import io.ksmt.sort.KArithSort
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBv16Sort
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBv32Sort
import io.ksmt.sort.KBv64Sort
import io.ksmt.sort.KBv8Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFp128Sort
import io.ksmt.sort.KFp16Sort
import io.ksmt.sort.KFp32Sort
import io.ksmt.sort.KFp64Sort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort


interface KTransformerBase {
    fun <T : KSort> apply(expr: KExpr<T>): KExpr<T> = expr.accept(this)

    fun transform(expr: KExpr<*>): Any = error("transformer is not implemented for expr $expr")

    // function transformers
    fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T>
    fun <T : KSort> transform(expr: KConst<T>): KExpr<T>

    // bool transformers
    fun transform(expr: KAndExpr): KExpr<KBoolSort>
    fun transform(expr: KAndBinaryExpr): KExpr<KBoolSort>
    fun transform(expr: KOrExpr): KExpr<KBoolSort>
    fun transform(expr: KOrBinaryExpr): KExpr<KBoolSort>
    fun transform(expr: KNotExpr): KExpr<KBoolSort>
    fun transform(expr: KImpliesExpr): KExpr<KBoolSort>
    fun transform(expr: KXorExpr): KExpr<KBoolSort>
    fun transform(expr: KTrue): KExpr<KBoolSort>
    fun transform(expr: KFalse): KExpr<KBoolSort>
    fun <T : KSort> transform(expr: KEqExpr<T>): KExpr<KBoolSort>
    fun <T : KSort> transform(expr: KDistinctExpr<T>): KExpr<KBoolSort>
    fun <T : KSort> transform(expr: KIteExpr<T>): KExpr<T>

    // bit-vec transformers
    fun transform(expr: KBitVec1Value): KExpr<KBv1Sort>
    fun transform(expr: KBitVec8Value): KExpr<KBv8Sort>
    fun transform(expr: KBitVec16Value): KExpr<KBv16Sort>
    fun transform(expr: KBitVec32Value): KExpr<KBv32Sort>
    fun transform(expr: KBitVec64Value): KExpr<KBv64Sort>
    fun transform(expr: KBitVecCustomValue): KExpr<KBvSort>

    // bit-vec expressions transformers
    fun <T : KBvSort> transform(expr: KBvNotExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>): KExpr<KBv1Sort>
    fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>): KExpr<KBv1Sort>
    fun <T : KBvSort> transform(expr: KBvAndExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvOrExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvXorExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvNAndExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvNorExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvXNorExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvNegationExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvAddExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvSubExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvMulExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>): KExpr<KBoolSort>
    fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>): KExpr<KBoolSort>
    fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>): KExpr<KBoolSort>
    fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>): KExpr<KBoolSort>
    fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>): KExpr<KBoolSort>
    fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>): KExpr<KBoolSort>
    fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>): KExpr<KBoolSort>
    fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>): KExpr<KBoolSort>
    fun transform(expr: KBvConcatExpr): KExpr<KBvSort>
    fun transform(expr: KBvExtractExpr): KExpr<KBvSort>
    fun transform(expr: KBvSignExtensionExpr): KExpr<KBvSort>
    fun transform(expr: KBvZeroExtensionExpr): KExpr<KBvSort>
    fun transform(expr: KBvRepeatExpr): KExpr<KBvSort>
    fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>): KExpr<T>
    fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>): KExpr<T>
    fun transform(expr: KBv2IntExpr): KExpr<KIntSort>
    fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>): KExpr<KBoolSort>
    fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>): KExpr<KBoolSort>
    fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>): KExpr<KBoolSort>
    fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort>
    fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>): KExpr<KBoolSort>
    fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>): KExpr<KBoolSort>
    fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>): KExpr<KBoolSort>
    fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>): KExpr<KBoolSort>

    // fp value transformers
    fun transform(expr: KFp16Value): KExpr<KFp16Sort>
    fun transform(expr: KFp32Value): KExpr<KFp32Sort>
    fun transform(expr: KFp64Value): KExpr<KFp64Sort>
    fun transform(expr: KFp128Value): KExpr<KFp128Sort>
    fun transform(expr: KFpCustomSizeValue): KExpr<KFpSort>

    // fp rounding mode
    fun transform(expr: KFpRoundingModeExpr): KExpr<KFpRoundingModeSort>

    // fp operations tranformation
    fun <T : KFpSort> transform(expr: KFpAbsExpr<T>): KExpr<T>
    fun <T : KFpSort> transform(expr: KFpNegationExpr<T>): KExpr<T>
    fun <T : KFpSort> transform(expr: KFpAddExpr<T>): KExpr<T>
    fun <T : KFpSort> transform(expr: KFpSubExpr<T>): KExpr<T>
    fun <T : KFpSort> transform(expr: KFpMulExpr<T>): KExpr<T>
    fun <T : KFpSort> transform(expr: KFpDivExpr<T>): KExpr<T>
    fun <T : KFpSort> transform(expr: KFpFusedMulAddExpr<T>): KExpr<T>
    fun <T : KFpSort> transform(expr: KFpSqrtExpr<T>): KExpr<T>
    fun <T : KFpSort> transform(expr: KFpRemExpr<T>): KExpr<T>
    fun <T : KFpSort> transform(expr: KFpRoundToIntegralExpr<T>): KExpr<T>
    fun <T : KFpSort> transform(expr: KFpMinExpr<T>): KExpr<T>
    fun <T : KFpSort> transform(expr: KFpMaxExpr<T>): KExpr<T>
    fun <T : KFpSort> transform(expr: KFpLessOrEqualExpr<T>): KExpr<KBoolSort>
    fun <T : KFpSort> transform(expr: KFpLessExpr<T>): KExpr<KBoolSort>
    fun <T : KFpSort> transform(expr: KFpGreaterOrEqualExpr<T>): KExpr<KBoolSort>
    fun <T : KFpSort> transform(expr: KFpGreaterExpr<T>): KExpr<KBoolSort>
    fun <T : KFpSort> transform(expr: KFpEqualExpr<T>): KExpr<KBoolSort>
    fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>): KExpr<KBoolSort>
    fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>): KExpr<KBoolSort>
    fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>): KExpr<KBoolSort>
    fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>): KExpr<KBoolSort>
    fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>): KExpr<KBoolSort>
    fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>): KExpr<KBoolSort>
    fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>): KExpr<KBoolSort>
    fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort>
    fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort>
    fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort>
    fun <T : KFpSort> transform(expr: KFpFromBvExpr<T>): KExpr<T>
    fun <T : KFpSort> transform(expr: KFpToFpExpr<T>): KExpr<T>
    fun <T : KFpSort> transform(expr: KRealToFpExpr<T>): KExpr<T>
    fun <T : KFpSort> transform(expr: KBvToFpExpr<T>): KExpr<T>

    // array transformers
    fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>): KExpr<KArraySort<D, R>>
    fun <D0 : KSort, D1 : KSort, R : KSort> transform(expr: KArray2Store<D0, D1, R>): KExpr<KArray2Sort<D0, D1, R>>

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Store<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>>

    fun <R : KSort> transform(expr: KArrayNStore<R>): KExpr<KArrayNSort<R>>

    fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R>
    fun <D0 : KSort, D1 : KSort, R : KSort> transform(expr: KArray2Select<D0, D1, R>): KExpr<R>
    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(expr: KArray3Select<D0, D1, D2, R>): KExpr<R>
    fun <R : KSort> transform(expr: KArrayNSelect<R>): KExpr<R>

    fun <A : KArraySortBase<R>, R : KSort> transform(expr: KArrayConst<A, R>): KExpr<A>
    fun <A : KArraySortBase<R>, R : KSort> transform(expr: KFunctionAsArray<A, R>): KExpr<A>

    fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>>
    fun <D0 : KSort, D1 : KSort, R : KSort> transform(expr: KArray2Lambda<D0, D1, R>): KExpr<KArray2Sort<D0, D1, R>>

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Lambda<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>>

    fun <R : KSort> transform(expr: KArrayNLambda<R>): KExpr<KArrayNSort<R>>

    // arith transformers
    fun <T : KArithSort> transform(expr: KAddArithExpr<T>): KExpr<T>
    fun <T : KArithSort> transform(expr: KMulArithExpr<T>): KExpr<T>
    fun <T : KArithSort> transform(expr: KSubArithExpr<T>): KExpr<T>
    fun <T : KArithSort> transform(expr: KUnaryMinusArithExpr<T>): KExpr<T>
    fun <T : KArithSort> transform(expr: KDivArithExpr<T>): KExpr<T>
    fun <T : KArithSort> transform(expr: KPowerArithExpr<T>): KExpr<T>
    fun <T : KArithSort> transform(expr: KLtArithExpr<T>): KExpr<KBoolSort>
    fun <T : KArithSort> transform(expr: KLeArithExpr<T>): KExpr<KBoolSort>
    fun <T : KArithSort> transform(expr: KGtArithExpr<T>): KExpr<KBoolSort>
    fun <T : KArithSort> transform(expr: KGeArithExpr<T>): KExpr<KBoolSort>

    // integer transformers
    fun transform(expr: KModIntExpr): KExpr<KIntSort>
    fun transform(expr: KRemIntExpr): KExpr<KIntSort>
    fun transform(expr: KToRealIntExpr): KExpr<KRealSort>
    fun transform(expr: KInt32NumExpr): KExpr<KIntSort>
    fun transform(expr: KInt64NumExpr): KExpr<KIntSort>
    fun transform(expr: KIntBigNumExpr): KExpr<KIntSort>

    // real transformers
    fun transform(expr: KToIntRealExpr): KExpr<KIntSort>
    fun transform(expr: KIsIntRealExpr): KExpr<KBoolSort>
    fun transform(expr: KRealNumExpr): KExpr<KRealSort>

    // quantifier transformers
    fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort>
    fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort>

    // uninterpreted sort value
    fun transform(expr: KUninterpretedSortValue): KExpr<KUninterpretedSort>
}
