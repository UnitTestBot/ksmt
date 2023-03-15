package org.ksmt.runner.serializer

import org.ksmt.expr.KAddArithExpr
import org.ksmt.expr.KAndBinaryExpr
import org.ksmt.expr.KAndExpr
import org.ksmt.expr.KArray2Lambda
import org.ksmt.expr.KArray2Select
import org.ksmt.expr.KArray2Store
import org.ksmt.expr.KArray3Lambda
import org.ksmt.expr.KArray3Select
import org.ksmt.expr.KArray3Store
import org.ksmt.expr.KArrayConst
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KArrayNLambda
import org.ksmt.expr.KArrayNSelect
import org.ksmt.expr.KArrayNStore
import org.ksmt.expr.KArraySelect
import org.ksmt.expr.KArrayStore
import org.ksmt.expr.KBitVec16Value
import org.ksmt.expr.KBitVec1Value
import org.ksmt.expr.KBitVec32Value
import org.ksmt.expr.KBitVec64Value
import org.ksmt.expr.KBitVec8Value
import org.ksmt.expr.KBitVecCustomValue
import org.ksmt.expr.KBv2IntExpr
import org.ksmt.expr.KBvAddExpr
import org.ksmt.expr.KBvAddNoOverflowExpr
import org.ksmt.expr.KBvAddNoUnderflowExpr
import org.ksmt.expr.KBvAndExpr
import org.ksmt.expr.KBvArithShiftRightExpr
import org.ksmt.expr.KBvConcatExpr
import org.ksmt.expr.KBvDivNoOverflowExpr
import org.ksmt.expr.KBvExtractExpr
import org.ksmt.expr.KBvLogicalShiftRightExpr
import org.ksmt.expr.KBvMulExpr
import org.ksmt.expr.KBvMulNoOverflowExpr
import org.ksmt.expr.KBvMulNoUnderflowExpr
import org.ksmt.expr.KBvNAndExpr
import org.ksmt.expr.KBvNegNoOverflowExpr
import org.ksmt.expr.KBvNegationExpr
import org.ksmt.expr.KBvNorExpr
import org.ksmt.expr.KBvNotExpr
import org.ksmt.expr.KBvOrExpr
import org.ksmt.expr.KBvReductionAndExpr
import org.ksmt.expr.KBvReductionOrExpr
import org.ksmt.expr.KBvRepeatExpr
import org.ksmt.expr.KBvRotateLeftExpr
import org.ksmt.expr.KBvRotateLeftIndexedExpr
import org.ksmt.expr.KBvRotateRightExpr
import org.ksmt.expr.KBvRotateRightIndexedExpr
import org.ksmt.expr.KBvShiftLeftExpr
import org.ksmt.expr.KBvSignExtensionExpr
import org.ksmt.expr.KBvSignedDivExpr
import org.ksmt.expr.KBvSignedGreaterExpr
import org.ksmt.expr.KBvSignedGreaterOrEqualExpr
import org.ksmt.expr.KBvSignedLessExpr
import org.ksmt.expr.KBvSignedLessOrEqualExpr
import org.ksmt.expr.KBvSignedModExpr
import org.ksmt.expr.KBvSignedRemExpr
import org.ksmt.expr.KBvSubExpr
import org.ksmt.expr.KBvSubNoOverflowExpr
import org.ksmt.expr.KBvSubNoUnderflowExpr
import org.ksmt.expr.KBvToFpExpr
import org.ksmt.expr.KBvUnsignedDivExpr
import org.ksmt.expr.KBvUnsignedGreaterExpr
import org.ksmt.expr.KBvUnsignedGreaterOrEqualExpr
import org.ksmt.expr.KBvUnsignedLessExpr
import org.ksmt.expr.KBvUnsignedLessOrEqualExpr
import org.ksmt.expr.KBvUnsignedRemExpr
import org.ksmt.expr.KBvXNorExpr
import org.ksmt.expr.KBvXorExpr
import org.ksmt.expr.KBvZeroExtensionExpr
import org.ksmt.expr.KConst
import org.ksmt.expr.KDistinctExpr
import org.ksmt.expr.KDivArithExpr
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFalse
import org.ksmt.expr.KFp128Value
import org.ksmt.expr.KFp16Value
import org.ksmt.expr.KFp32Value
import org.ksmt.expr.KFp64Value
import org.ksmt.expr.KFpAbsExpr
import org.ksmt.expr.KFpAddExpr
import org.ksmt.expr.KFpCustomSizeValue
import org.ksmt.expr.KFpDivExpr
import org.ksmt.expr.KFpEqualExpr
import org.ksmt.expr.KFpFromBvExpr
import org.ksmt.expr.KFpFusedMulAddExpr
import org.ksmt.expr.KFpGreaterExpr
import org.ksmt.expr.KFpGreaterOrEqualExpr
import org.ksmt.expr.KFpIsInfiniteExpr
import org.ksmt.expr.KFpIsNaNExpr
import org.ksmt.expr.KFpIsNegativeExpr
import org.ksmt.expr.KFpIsNormalExpr
import org.ksmt.expr.KFpIsPositiveExpr
import org.ksmt.expr.KFpIsSubnormalExpr
import org.ksmt.expr.KFpIsZeroExpr
import org.ksmt.expr.KFpLessExpr
import org.ksmt.expr.KFpLessOrEqualExpr
import org.ksmt.expr.KFpMaxExpr
import org.ksmt.expr.KFpMinExpr
import org.ksmt.expr.KFpMulExpr
import org.ksmt.expr.KFpNegationExpr
import org.ksmt.expr.KFpRemExpr
import org.ksmt.expr.KFpRoundToIntegralExpr
import org.ksmt.expr.KFpRoundingModeExpr
import org.ksmt.expr.KFpSqrtExpr
import org.ksmt.expr.KFpSubExpr
import org.ksmt.expr.KFpToBvExpr
import org.ksmt.expr.KFpToFpExpr
import org.ksmt.expr.KFpToIEEEBvExpr
import org.ksmt.expr.KFpToRealExpr
import org.ksmt.expr.KFunctionApp
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.expr.KGeArithExpr
import org.ksmt.expr.KGtArithExpr
import org.ksmt.expr.KImpliesExpr
import org.ksmt.expr.KInt32NumExpr
import org.ksmt.expr.KInt64NumExpr
import org.ksmt.expr.KIntBigNumExpr
import org.ksmt.expr.KIsIntRealExpr
import org.ksmt.expr.KIteExpr
import org.ksmt.expr.KLeArithExpr
import org.ksmt.expr.KLtArithExpr
import org.ksmt.expr.KModIntExpr
import org.ksmt.expr.KMulArithExpr
import org.ksmt.expr.KNotExpr
import org.ksmt.expr.KOrBinaryExpr
import org.ksmt.expr.KOrExpr
import org.ksmt.expr.KPowerArithExpr
import org.ksmt.expr.KRealNumExpr
import org.ksmt.expr.KRealToFpExpr
import org.ksmt.expr.KRemIntExpr
import org.ksmt.expr.KSubArithExpr
import org.ksmt.expr.KToIntRealExpr
import org.ksmt.expr.KToRealIntExpr
import org.ksmt.expr.KTrue
import org.ksmt.expr.KUnaryMinusArithExpr
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.KXorExpr
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv16Sort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBv64Sort
import org.ksmt.sort.KBv8Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFp128Sort
import org.ksmt.sort.KFp16Sort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort

class ExprKindMapper: KTransformerBase {

    fun getKind(expr: KExpr<*>): ExprKind {
        expr.accept(this)
        return exprKind
    }

    private lateinit var exprKind: ExprKind
    private fun <T> T.kind(kind: ExprKind): T {
        exprKind = kind
        return this
    }

    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> = expr.kind(ExprKind.FunctionApp)


    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = expr.kind(ExprKind.Const)


    override fun transform(expr: KAndExpr): KExpr<KBoolSort> = expr.kind(ExprKind.AndExpr)

    override fun transform(expr: KAndBinaryExpr): KExpr<KBoolSort> = expr.kind(ExprKind.AndBinaryExpr)

    override fun transform(expr: KOrExpr): KExpr<KBoolSort> = expr.kind(ExprKind.OrExpr)

    override fun transform(expr: KOrBinaryExpr): KExpr<KBoolSort> = expr.kind(ExprKind.OrBinaryExpr)

    override fun transform(expr: KNotExpr): KExpr<KBoolSort> = expr.kind(ExprKind.NotExpr)


    override fun transform(expr: KImpliesExpr): KExpr<KBoolSort> = expr.kind(ExprKind.ImpliesExpr)


    override fun transform(expr: KXorExpr): KExpr<KBoolSort> = expr.kind(ExprKind.XorExpr)


    override fun transform(expr: KTrue): KExpr<KBoolSort> = expr.kind(ExprKind.True)


    override fun transform(expr: KFalse): KExpr<KBoolSort> = expr.kind(ExprKind.False)


    override fun <T : KSort> transform(expr: KEqExpr<T>): KExpr<KBoolSort> = expr.kind(ExprKind.EqExpr)


    override fun <T : KSort> transform(expr: KDistinctExpr<T>): KExpr<KBoolSort> = expr.kind(ExprKind.DistinctExpr)


    override fun <T : KSort> transform(expr: KIteExpr<T>): KExpr<T> = expr.kind(ExprKind.IteExpr)


    override fun transform(expr: KBitVec1Value): KExpr<KBv1Sort> = expr.kind(ExprKind.BitVec1Value)


    override fun transform(expr: KBitVec8Value): KExpr<KBv8Sort> = expr.kind(ExprKind.BitVec8Value)


    override fun transform(expr: KBitVec16Value): KExpr<KBv16Sort> = expr.kind(ExprKind.BitVec16Value)


    override fun transform(expr: KBitVec32Value): KExpr<KBv32Sort> = expr.kind(ExprKind.BitVec32Value)


    override fun transform(expr: KBitVec64Value): KExpr<KBv64Sort> = expr.kind(ExprKind.BitVec64Value)


    override fun transform(expr: KBitVecCustomValue): KExpr<KBvSort> = expr.kind(ExprKind.BitVecCustomValue)


    override fun <T : KBvSort> transform(expr: KBvNotExpr<T>): KExpr<T> = expr.kind(ExprKind.BvNotExpr)


    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>): KExpr<KBv1Sort> =
        expr.kind(ExprKind.BvReductionAndExpr)


    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>): KExpr<KBv1Sort> =
        expr.kind(ExprKind.BvReductionOrExpr)


    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>): KExpr<T> = expr.kind(ExprKind.BvAndExpr)


    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>): KExpr<T> = expr.kind(ExprKind.BvOrExpr)


    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>): KExpr<T> = expr.kind(ExprKind.BvXorExpr)


    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>): KExpr<T> = expr.kind(ExprKind.BvNAndExpr)


    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>): KExpr<T> = expr.kind(ExprKind.BvNorExpr)


    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>): KExpr<T> = expr.kind(ExprKind.BvXNorExpr)


    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>): KExpr<T> = expr.kind(ExprKind.BvNegationExpr)


    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>): KExpr<T> = expr.kind(ExprKind.BvAddExpr)


    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>): KExpr<T> = expr.kind(ExprKind.BvSubExpr)


    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>): KExpr<T> = expr.kind(ExprKind.BvMulExpr)


    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>): KExpr<T> = expr.kind(ExprKind.BvUnsignedDivExpr)


    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>): KExpr<T> = expr.kind(ExprKind.BvSignedDivExpr)


    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>): KExpr<T> = expr.kind(ExprKind.BvUnsignedRemExpr)


    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>): KExpr<T> = expr.kind(ExprKind.BvSignedRemExpr)


    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>): KExpr<T> = expr.kind(ExprKind.BvSignedModExpr)


    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.BvUnsignedLessExpr)


    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.BvSignedLessExpr)


    override fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.BvUnsignedLessOrEqualExpr)


    override fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.BvSignedLessOrEqualExpr)


    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.BvUnsignedGreaterOrEqualExpr)


    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.BvSignedGreaterOrEqualExpr)


    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.BvUnsignedGreaterExpr)


    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.BvSignedGreaterExpr)


    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> = expr.kind(ExprKind.BvConcatExpr)


    override fun transform(expr: KBvExtractExpr): KExpr<KBvSort> = expr.kind(ExprKind.BvExtractExpr)


    override fun transform(expr: KBvSignExtensionExpr): KExpr<KBvSort> = expr.kind(ExprKind.BvSignExtensionExpr)


    override fun transform(expr: KBvZeroExtensionExpr): KExpr<KBvSort> = expr.kind(ExprKind.BvZeroExtensionExpr)


    override fun transform(expr: KBvRepeatExpr): KExpr<KBvSort> = expr.kind(ExprKind.BvRepeatExpr)


    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>): KExpr<T> = expr.kind(ExprKind.BvShiftLeftExpr)


    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>): KExpr<T> =
        expr.kind(ExprKind.BvLogicalShiftRightExpr)


    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>): KExpr<T> =
        expr.kind(ExprKind.BvArithShiftRightExpr)


    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>): KExpr<T> = expr.kind(ExprKind.BvRotateLeftExpr)


    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>): KExpr<T> =
        expr.kind(ExprKind.BvRotateLeftIndexedExpr)


    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>): KExpr<T> = expr.kind(ExprKind.BvRotateRightExpr)


    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>): KExpr<T> =
        expr.kind(ExprKind.BvRotateRightIndexedExpr)


    override fun transform(expr: KBv2IntExpr): KExpr<KIntSort> = expr.kind(ExprKind.Bv2IntExpr)


    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.BvAddNoOverflowExpr)


    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.BvAddNoUnderflowExpr)


    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.BvSubNoOverflowExpr)


    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.BvSubNoUnderflowExpr)


    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.BvDivNoOverflowExpr)


    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.BvNegNoOverflowExpr)


    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.BvMulNoOverflowExpr)


    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.BvMulNoUnderflowExpr)


    override fun transform(expr: KFp16Value): KExpr<KFp16Sort> = expr.kind(ExprKind.Fp16Value)


    override fun transform(expr: KFp32Value): KExpr<KFp32Sort> = expr.kind(ExprKind.Fp32Value)


    override fun transform(expr: KFp64Value): KExpr<KFp64Sort> = expr.kind(ExprKind.Fp64Value)


    override fun transform(expr: KFp128Value): KExpr<KFp128Sort> = expr.kind(ExprKind.Fp128Value)


    override fun transform(expr: KFpCustomSizeValue): KExpr<KFpSort> = expr.kind(ExprKind.FpCustomSizeValue)


    override fun transform(expr: KFpRoundingModeExpr): KExpr<KFpRoundingModeSort> =
        expr.kind(ExprKind.FpRoundingModeExpr)


    override fun <T : KFpSort> transform(expr: KFpAbsExpr<T>): KExpr<T> = expr.kind(ExprKind.FpAbsExpr)


    override fun <T : KFpSort> transform(expr: KFpNegationExpr<T>): KExpr<T> = expr.kind(ExprKind.FpNegationExpr)


    override fun <T : KFpSort> transform(expr: KFpAddExpr<T>): KExpr<T> = expr.kind(ExprKind.FpAddExpr)


    override fun <T : KFpSort> transform(expr: KFpSubExpr<T>): KExpr<T> = expr.kind(ExprKind.FpSubExpr)


    override fun <T : KFpSort> transform(expr: KFpMulExpr<T>): KExpr<T> = expr.kind(ExprKind.FpMulExpr)


    override fun <T : KFpSort> transform(expr: KFpDivExpr<T>): KExpr<T> = expr.kind(ExprKind.FpDivExpr)


    override fun <T : KFpSort> transform(expr: KFpFusedMulAddExpr<T>): KExpr<T> = expr.kind(ExprKind.FpFusedMulAddExpr)


    override fun <T : KFpSort> transform(expr: KFpSqrtExpr<T>): KExpr<T> = expr.kind(ExprKind.FpSqrtExpr)


    override fun <T : KFpSort> transform(expr: KFpRemExpr<T>): KExpr<T> = expr.kind(ExprKind.FpRemExpr)


    override fun <T : KFpSort> transform(expr: KFpRoundToIntegralExpr<T>): KExpr<T> =
        expr.kind(ExprKind.FpRoundToIntegralExpr)


    override fun <T : KFpSort> transform(expr: KFpMinExpr<T>): KExpr<T> = expr.kind(ExprKind.FpMinExpr)


    override fun <T : KFpSort> transform(expr: KFpMaxExpr<T>): KExpr<T> = expr.kind(ExprKind.FpMaxExpr)


    override fun <T : KFpSort> transform(expr: KFpLessOrEqualExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.FpLessOrEqualExpr)


    override fun <T : KFpSort> transform(expr: KFpLessExpr<T>): KExpr<KBoolSort> = expr.kind(ExprKind.FpLessExpr)


    override fun <T : KFpSort> transform(expr: KFpGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.FpGreaterOrEqualExpr)


    override fun <T : KFpSort> transform(expr: KFpGreaterExpr<T>): KExpr<KBoolSort> = expr.kind(ExprKind.FpGreaterExpr)


    override fun <T : KFpSort> transform(expr: KFpEqualExpr<T>): KExpr<KBoolSort> = expr.kind(ExprKind.FpEqualExpr)


    override fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.FpIsNormalExpr)


    override fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.FpIsSubnormalExpr)


    override fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>): KExpr<KBoolSort> = expr.kind(ExprKind.FpIsZeroExpr)


    override fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.FpIsInfiniteExpr)


    override fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>): KExpr<KBoolSort> = expr.kind(ExprKind.FpIsNaNExpr)


    override fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.FpIsNegativeExpr)


    override fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.FpIsPositiveExpr)


    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort> = expr.kind(ExprKind.FpToBvExpr)


    override fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort> = expr.kind(ExprKind.FpToRealExpr)


    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> = expr.kind(ExprKind.FpToIEEEBvExpr)


    override fun <T : KFpSort> transform(expr: KFpFromBvExpr<T>): KExpr<T> = expr.kind(ExprKind.FpFromBvExpr)


    override fun <T : KFpSort> transform(expr: KFpToFpExpr<T>): KExpr<T> = expr.kind(ExprKind.FpToFpExpr)


    override fun <T : KFpSort> transform(expr: KRealToFpExpr<T>): KExpr<T> = expr.kind(ExprKind.RealToFpExpr)


    override fun <T : KFpSort> transform(expr: KBvToFpExpr<T>): KExpr<T> = expr.kind(ExprKind.BvToFpExpr)


    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>): KExpr<KArraySort<D, R>> =
        expr.kind(ExprKind.ArrayStore)

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Store<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = expr.kind(ExprKind.Array2Store)

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Store<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = expr.kind(ExprKind.Array3Store)

    override fun <R : KSort> transform(expr: KArrayNStore<R>): KExpr<KArrayNSort<R>> =
        expr.kind(ExprKind.ArrayNStore)

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> = expr.kind(ExprKind.ArraySelect)

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Select<D0, D1, R>
    ): KExpr<R> = expr.kind(ExprKind.Array2Select)

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Select<D0, D1, D2, R>
    ): KExpr<R> = expr.kind(ExprKind.Array3Select)

    override fun <R : KSort> transform(expr: KArrayNSelect<R>): KExpr<R> = expr.kind(ExprKind.ArrayNSelect)

    override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KArrayConst<A, R>): KExpr<A> =
        expr.kind(ExprKind.ArrayConst)

    override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KFunctionAsArray<A, R>): KExpr<A> =
        expr.kind(ExprKind.FunctionAsArray)


    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>> =
        expr.kind(ExprKind.ArrayLambda)

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Lambda<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = expr.kind(ExprKind.Array2Lambda)

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Lambda<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = expr.kind(ExprKind.Array3Lambda)

    override fun <R : KSort> transform(expr: KArrayNLambda<R>): KExpr<KArrayNSort<R>> =
        expr.kind(ExprKind.ArrayNLambda)

    override fun <T : KArithSort> transform(expr: KAddArithExpr<T>): KExpr<T> = expr.kind(ExprKind.AddArithExpr)


    override fun <T : KArithSort> transform(expr: KMulArithExpr<T>): KExpr<T> = expr.kind(ExprKind.MulArithExpr)


    override fun <T : KArithSort> transform(expr: KSubArithExpr<T>): KExpr<T> = expr.kind(ExprKind.SubArithExpr)


    override fun <T : KArithSort> transform(expr: KUnaryMinusArithExpr<T>): KExpr<T> =
        expr.kind(ExprKind.UnaryMinusArithExpr)


    override fun <T : KArithSort> transform(expr: KDivArithExpr<T>): KExpr<T> = expr.kind(ExprKind.DivArithExpr)


    override fun <T : KArithSort> transform(expr: KPowerArithExpr<T>): KExpr<T> = expr.kind(ExprKind.PowerArithExpr)


    override fun <T : KArithSort> transform(expr: KLtArithExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.LtArithExpr)


    override fun <T : KArithSort> transform(expr: KLeArithExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.LeArithExpr)


    override fun <T : KArithSort> transform(expr: KGtArithExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.GtArithExpr)


    override fun <T : KArithSort> transform(expr: KGeArithExpr<T>): KExpr<KBoolSort> =
        expr.kind(ExprKind.GeArithExpr)


    override fun transform(expr: KModIntExpr): KExpr<KIntSort> = expr.kind(ExprKind.ModIntExpr)


    override fun transform(expr: KRemIntExpr): KExpr<KIntSort> = expr.kind(ExprKind.RemIntExpr)


    override fun transform(expr: KToRealIntExpr): KExpr<KRealSort> = expr.kind(ExprKind.ToRealIntExpr)


    override fun transform(expr: KInt32NumExpr): KExpr<KIntSort> = expr.kind(ExprKind.Int32NumExpr)


    override fun transform(expr: KInt64NumExpr): KExpr<KIntSort> = expr.kind(ExprKind.Int64NumExpr)


    override fun transform(expr: KIntBigNumExpr): KExpr<KIntSort> = expr.kind(ExprKind.IntBigNumExpr)


    override fun transform(expr: KToIntRealExpr): KExpr<KIntSort> = expr.kind(ExprKind.ToIntRealExpr)


    override fun transform(expr: KIsIntRealExpr): KExpr<KBoolSort> = expr.kind(ExprKind.IsIntRealExpr)


    override fun transform(expr: KRealNumExpr): KExpr<KRealSort> = expr.kind(ExprKind.RealNumExpr)


    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> = expr.kind(ExprKind.ExistentialQuantifier)


    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> = expr.kind(ExprKind.UniversalQuantifier)

}
