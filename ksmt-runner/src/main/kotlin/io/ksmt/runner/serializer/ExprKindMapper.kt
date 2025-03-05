package io.ksmt.runner.serializer

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
import io.ksmt.expr.KStringConcatExpr
import io.ksmt.expr.KStringLenExpr
import io.ksmt.expr.KStringToRegexExpr
import io.ksmt.expr.KStringInRegexExpr
import io.ksmt.expr.KStringSuffixOfExpr
import io.ksmt.expr.KStringPrefixOfExpr
import io.ksmt.expr.KStringLtExpr
import io.ksmt.expr.KStringLeExpr
import io.ksmt.expr.KStringGtExpr
import io.ksmt.expr.KStringGeExpr
import io.ksmt.expr.KStringContainsExpr
import io.ksmt.expr.KStringSingletonSubExpr
import io.ksmt.expr.KStringSubExpr
import io.ksmt.expr.KStringIndexOfExpr
import io.ksmt.expr.KStringIndexOfRegexExpr
import io.ksmt.expr.KStringReplaceExpr
import io.ksmt.expr.KStringReplaceAllExpr
import io.ksmt.expr.KStringReplaceWithRegexExpr
import io.ksmt.expr.KStringReplaceAllWithRegexExpr
import io.ksmt.expr.KStringToLowerExpr
import io.ksmt.expr.KStringToUpperExpr
import io.ksmt.expr.KStringReverseExpr
import io.ksmt.expr.KStringIsDigitExpr
import io.ksmt.expr.KStringToCodeExpr
import io.ksmt.expr.KStringFromCodeExpr
import io.ksmt.expr.KStringToIntExpr
import io.ksmt.expr.KStringFromIntExpr
import io.ksmt.expr.KStringLiteralExpr
import io.ksmt.expr.KRegexConcatExpr
import io.ksmt.expr.KRegexUnionExpr
import io.ksmt.expr.KRegexIntersectionExpr
import io.ksmt.expr.KRegexStarExpr
import io.ksmt.expr.KRegexCrossExpr
import io.ksmt.expr.KRegexDifferenceExpr
import io.ksmt.expr.KRegexComplementExpr
import io.ksmt.expr.KRegexOptionExpr
import io.ksmt.expr.KRegexRangeExpr
import io.ksmt.expr.KRegexPowerExpr
import io.ksmt.expr.KRegexLoopExpr
import io.ksmt.expr.KRegexEpsilon
import io.ksmt.expr.KRegexAll
import io.ksmt.expr.KRegexAllChar
import io.ksmt.expr.transformer.KTransformerBase
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
import io.ksmt.sort.KStringSort
import io.ksmt.sort.KRegexSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort

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


    override fun transform(expr: KStringConcatExpr): KExpr<KStringSort> = expr.kind(ExprKind.StringConcatExpr)


    override fun transform(expr: KStringLenExpr): KExpr<KIntSort> = expr.kind(ExprKind.StringLenExpr)


    override fun transform(expr: KStringToRegexExpr): KExpr<KRegexSort> = expr.kind(ExprKind.StringToRegexExpr)


    override fun transform(expr: KStringInRegexExpr): KExpr<KBoolSort> = expr.kind(ExprKind.StringInRegexExpr)


    override fun transform(expr: KStringSuffixOfExpr): KExpr<KBoolSort> = expr.kind(ExprKind.StringSuffixOfExpr)


    override fun transform(expr: KStringPrefixOfExpr): KExpr<KBoolSort> = expr.kind(ExprKind.StringPrefixOfExpr)


    override fun transform(expr: KStringLtExpr): KExpr<KBoolSort> = expr.kind(ExprKind.StringLtExpr)


    override fun transform(expr: KStringLeExpr): KExpr<KBoolSort> = expr.kind(ExprKind.StringLeExpr)


    override fun transform(expr: KStringGtExpr): KExpr<KBoolSort> = expr.kind(ExprKind.StringGtExpr)


    override fun transform(expr: KStringGeExpr): KExpr<KBoolSort> = expr.kind(ExprKind.StringGeExpr)


    override fun transform(expr: KStringContainsExpr): KExpr<KBoolSort> = expr.kind(ExprKind.StringContainsExpr)


    override fun transform(
        expr: KStringSingletonSubExpr
    ): KExpr<KStringSort> = expr.kind(ExprKind.StringSingletonSubExpr)


    override fun transform(expr: KStringSubExpr): KExpr<KStringSort> = expr.kind(ExprKind.StringSubExpr)


    override fun transform(expr: KStringIndexOfExpr): KExpr<KIntSort> = expr.kind(ExprKind.StringIndexOfExpr)


    override fun transform(expr: KStringIndexOfRegexExpr): KExpr<KIntSort> = expr.kind(ExprKind.StringIndexOfRegexExpr)


    override fun transform(expr: KStringReplaceExpr): KExpr<KStringSort> = expr.kind(ExprKind.StringReplaceExpr)


    override fun transform(expr: KStringReplaceAllExpr): KExpr<KStringSort> = expr.kind(ExprKind.StringReplaceAllExpr)


    override fun transform(
        expr: KStringReplaceWithRegexExpr
    ): KExpr<KStringSort> = expr.kind(ExprKind.StringReplaceWithRegexExpr)


    override fun transform(
        expr: KStringReplaceAllWithRegexExpr
    ): KExpr<KStringSort> = expr.kind(ExprKind.StringReplaceAllWithRegexExpr)


    override fun transform(expr: KStringToLowerExpr): KExpr<KStringSort> = expr.kind(ExprKind.StringToLowerExpr)


    override fun transform(expr: KStringToUpperExpr): KExpr<KStringSort> = expr.kind(ExprKind.StringToUpperExpr)


    override fun transform(expr: KStringReverseExpr): KExpr<KStringSort> = expr.kind(ExprKind.StringReverseExpr)


    override fun transform(expr: KStringIsDigitExpr): KExpr<KBoolSort> = expr.kind(ExprKind.StringIsDigitExpr)


    override fun transform(expr: KStringToCodeExpr): KExpr<KIntSort> = expr.kind(ExprKind.StringToCodeExpr)


    override fun transform(expr: KStringFromCodeExpr): KExpr<KStringSort> = expr.kind(ExprKind.StringFromCodeExpr)


    override fun transform(expr: KStringToIntExpr): KExpr<KIntSort> = expr.kind(ExprKind.StringToIntExpr)


    override fun transform(expr: KStringFromIntExpr): KExpr<KStringSort> = expr.kind(ExprKind.StringFromIntExpr)


    override fun transform(expr: KStringLiteralExpr): KExpr<KStringSort> = expr.kind(ExprKind.StringLiteralExpr)


    override fun transform(expr: KRegexConcatExpr): KExpr<KRegexSort> = expr.kind(ExprKind.RegexConcatExpr)


    override fun transform(expr: KRegexUnionExpr): KExpr<KRegexSort> = expr.kind(ExprKind.RegexUnionExpr)


    override fun transform(expr: KRegexIntersectionExpr): KExpr<KRegexSort> = expr.kind(ExprKind.RegexIntersectionExpr)


    override fun transform(expr: KRegexStarExpr): KExpr<KRegexSort> = expr.kind(ExprKind.RegexStarExpr)


    override fun transform(expr: KRegexCrossExpr): KExpr<KRegexSort> = expr.kind(ExprKind.RegexCrossExpr)


    override fun transform(expr: KRegexComplementExpr): KExpr<KRegexSort> = expr.kind(ExprKind.RegexComplementExpr)


    override fun transform(expr: KRegexDifferenceExpr): KExpr<KRegexSort> = expr.kind(ExprKind.RegexDifferenceExpr)


    override fun transform(expr: KRegexOptionExpr): KExpr<KRegexSort> = expr.kind(ExprKind.RegexOptionExpr)


    override fun transform(expr: KRegexRangeExpr): KExpr<KRegexSort> = expr.kind(ExprKind.RegexRangeExpr)


    override fun transform(expr: KRegexPowerExpr): KExpr<KRegexSort> = expr.kind(ExprKind.RegexPowerExpr)


    override fun transform(expr: KRegexLoopExpr): KExpr<KRegexSort> = expr.kind(ExprKind.RegexLoopExpr)


    override fun transform(expr: KRegexEpsilon): KExpr<KRegexSort> = expr.kind(ExprKind.RegexEpsilonExpr)


    override fun transform(expr: KRegexAll): KExpr<KRegexSort> = expr.kind(ExprKind.RegexAllExpr)


    override fun transform(expr: KRegexAllChar): KExpr<KRegexSort> = expr.kind(ExprKind.RegexAllCharExpr)


    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> = expr.kind(ExprKind.ExistentialQuantifier)


    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> = expr.kind(ExprKind.UniversalQuantifier)

    override fun transform(expr: KUninterpretedSortValue): KExpr<KUninterpretedSort> =
        expr.kind(ExprKind.UninterpretedSortValue)
}
