package io.ksmt.expr.transformer

import io.ksmt.expr.KAddArithExpr
import io.ksmt.expr.KAndBinaryExpr
import io.ksmt.expr.KAndExpr
import io.ksmt.expr.KApp
import io.ksmt.expr.KArray2Lambda
import io.ksmt.expr.KArray2Select
import io.ksmt.expr.KArray2Store
import io.ksmt.expr.KArray3Lambda
import io.ksmt.expr.KArray3Select
import io.ksmt.expr.KArray3Store
import io.ksmt.expr.KArrayConst
import io.ksmt.expr.KArrayLambda
import io.ksmt.expr.KArrayLambdaBase
import io.ksmt.expr.KArrayNLambda
import io.ksmt.expr.KArrayNSelect
import io.ksmt.expr.KArrayNStore
import io.ksmt.expr.KArraySelect
import io.ksmt.expr.KArraySelectBase
import io.ksmt.expr.KArrayStore
import io.ksmt.expr.KArrayStoreBase
import io.ksmt.expr.KBitVec16Value
import io.ksmt.expr.KBitVec1Value
import io.ksmt.expr.KBitVec32Value
import io.ksmt.expr.KBitVec64Value
import io.ksmt.expr.KBitVec8Value
import io.ksmt.expr.KBitVecCustomValue
import io.ksmt.expr.KBitVecValue
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
import io.ksmt.expr.KFpValue
import io.ksmt.expr.KFunctionApp
import io.ksmt.expr.KFunctionAsArray
import io.ksmt.expr.KGeArithExpr
import io.ksmt.expr.KGtArithExpr
import io.ksmt.expr.KImpliesExpr
import io.ksmt.expr.KInt32NumExpr
import io.ksmt.expr.KInt64NumExpr
import io.ksmt.expr.KIntBigNumExpr
import io.ksmt.expr.KIntNumExpr
import io.ksmt.expr.KInterpretedValue
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
import io.ksmt.expr.KStringReplaceExpr
import io.ksmt.expr.KStringReplaceAllExpr
import io.ksmt.expr.KStringReplaceWithRegexExpr
import io.ksmt.expr.KStringReplaceAllWithRegexExpr
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
import io.ksmt.expr.KRegexEpsilon
import io.ksmt.expr.KRegexAll
import io.ksmt.expr.KRegexAllChar
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


interface KVisitor<V> : KTransformer {
    fun <E : KExpr<*>> exprVisitResult(expr: E, result: V): Unit {}

    fun <T : KSort> visitExpr(expr: KExpr<T>): V

    override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> = visitExpr(expr, ::visitExpr)

    // function visitors
    fun <T : KSort, A : KSort> visitApp(expr: KApp<T, A>): V = visitExpr(expr)
    fun <T : KSort> visit(expr: KFunctionApp<T>): V = visitApp(expr)
    fun <T : KSort> visit(expr: KConst<T>): V = visit(expr as KFunctionApp<T>)
    fun <T : KSort> visitValue(expr: KInterpretedValue<T>): V = visitApp(expr)

    override fun <T : KSort, A : KSort> transformApp(expr: KApp<T, A>): KExpr<T> = visitExpr(expr, ::visitApp)
    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KSort> transformValue(expr: KInterpretedValue<T>): KExpr<T> = visitExpr(expr, ::visitValue)

    // bool visitors
    fun visit(expr: KAndExpr): V = visitApp(expr)
    fun visit(expr: KAndBinaryExpr): V = visit(expr as KAndExpr)
    fun visit(expr: KOrExpr): V = visitApp(expr)
    fun visit(expr: KOrBinaryExpr): V = visit(expr as KOrExpr)
    fun visit(expr: KNotExpr): V = visitApp(expr)
    fun visit(expr: KImpliesExpr): V = visitApp(expr)
    fun visit(expr: KXorExpr): V = visitApp(expr)
    fun visit(expr: KTrue): V = visitValue(expr)
    fun visit(expr: KFalse): V = visitValue(expr)
    fun <T : KSort> visit(expr: KEqExpr<T>): V = visitApp(expr)
    fun <T : KSort> visit(expr: KDistinctExpr<T>): V = visitApp(expr)
    fun <T : KSort> visit(expr: KIteExpr<T>): V = visitApp(expr)

    override fun transform(expr: KAndExpr): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KAndBinaryExpr): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KOrExpr): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KOrBinaryExpr): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KNotExpr): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KImpliesExpr): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KXorExpr): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KTrue): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KFalse): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KSort> transform(expr: KEqExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KSort> transform(expr: KDistinctExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KSort> transform(expr: KIteExpr<T>): KExpr<T> = visitExpr(expr, ::visit)

    // bit-vec visitors
    fun <T : KBvSort> visitBitVecValue(expr: KBitVecValue<T>): V = visitValue(expr)
    fun visit(expr: KBitVec1Value): V = visitBitVecValue(expr)
    fun visit(expr: KBitVec8Value): V = visitBitVecValue(expr)
    fun visit(expr: KBitVec16Value): V = visitBitVecValue(expr)
    fun visit(expr: KBitVec32Value): V = visitBitVecValue(expr)
    fun visit(expr: KBitVec64Value): V = visitBitVecValue(expr)
    fun visit(expr: KBitVecCustomValue): V = visitBitVecValue(expr)

    override fun <T : KBvSort> transformBitVecValue(expr: KBitVecValue<T>): KExpr<T> =
        visitExpr(expr, ::visitBitVecValue)

    override fun transform(expr: KBitVec1Value): KExpr<KBv1Sort> = visitExpr(expr, ::visit)
    override fun transform(expr: KBitVec8Value): KExpr<KBv8Sort> = visitExpr(expr, ::visit)
    override fun transform(expr: KBitVec16Value): KExpr<KBv16Sort> = visitExpr(expr, ::visit)
    override fun transform(expr: KBitVec32Value): KExpr<KBv32Sort> = visitExpr(expr, ::visit)
    override fun transform(expr: KBitVec64Value): KExpr<KBv64Sort> = visitExpr(expr, ::visit)
    override fun transform(expr: KBitVecCustomValue): KExpr<KBvSort> = visitExpr(expr, ::visit)

    // bit-vec expressions visitors
    fun <T : KBvSort> visit(expr: KBvNotExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvReductionAndExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvReductionOrExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvAndExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvOrExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvXorExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvNAndExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvNorExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvXNorExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvNegationExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvAddExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvSubExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvMulExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvUnsignedDivExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvSignedDivExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvUnsignedRemExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvSignedRemExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvSignedModExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvUnsignedLessExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvSignedLessExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvUnsignedLessOrEqualExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvSignedLessOrEqualExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvUnsignedGreaterOrEqualExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvSignedGreaterOrEqualExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvUnsignedGreaterExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvSignedGreaterExpr<T>): V = visitApp(expr)
    fun visit(expr: KBvConcatExpr): V = visitApp(expr)
    fun visit(expr: KBvExtractExpr): V = visitApp(expr)
    fun visit(expr: KBvSignExtensionExpr): V = visitApp(expr)
    fun visit(expr: KBvZeroExtensionExpr): V = visitApp(expr)
    fun visit(expr: KBvRepeatExpr): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvShiftLeftExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvLogicalShiftRightExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvArithShiftRightExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvRotateLeftExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvRotateLeftIndexedExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvRotateRightExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvRotateRightIndexedExpr<T>): V = visitApp(expr)
    fun visit(expr: KBv2IntExpr): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvAddNoOverflowExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvAddNoUnderflowExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvSubNoOverflowExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvSubNoUnderflowExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvDivNoOverflowExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvNegNoOverflowExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvMulNoOverflowExpr<T>): V = visitApp(expr)
    fun <T : KBvSort> visit(expr: KBvMulNoUnderflowExpr<T>): V = visitApp(expr)

    override fun <T : KBvSort> transform(expr: KBvNotExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>): KExpr<KBv1Sort> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>): KExpr<KBv1Sort> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>): KExpr<KBoolSort> = visitExpr(
        expr,
        ::visit
    )

    override fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>): KExpr<KBoolSort> = visitExpr(
        expr,
        ::visit
    )

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        visitExpr(expr, ::visit)

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> = visitExpr(
        expr,
        ::visit
    )

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>): KExpr<KBoolSort> = visitExpr(
        expr,
        ::visit
    )

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KBvExtractExpr): KExpr<KBvSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KBvSignExtensionExpr): KExpr<KBvSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KBvZeroExtensionExpr): KExpr<KBvSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KBvRepeatExpr): KExpr<KBvSort> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun transform(expr: KBv2IntExpr): KExpr<KIntSort> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)

    // fp value visitors
    fun <T : KFpSort> visitFpValue(expr: KFpValue<T>): V = visitValue(expr)
    fun visit(expr: KFp16Value): V = visitFpValue(expr)
    fun visit(expr: KFp32Value): V = visitFpValue(expr)
    fun visit(expr: KFp64Value): V = visitFpValue(expr)
    fun visit(expr: KFp128Value): V = visitFpValue(expr)
    fun visit(expr: KFpCustomSizeValue): V = visitFpValue(expr)

    override fun <T : KFpSort> transformFpValue(expr: KFpValue<T>): KExpr<T> =
        visitExpr(expr, ::visitFpValue)

    override fun transform(expr: KFp16Value): KExpr<KFp16Sort> = visitExpr(expr, ::visit)
    override fun transform(expr: KFp32Value): KExpr<KFp32Sort> = visitExpr(expr, ::visit)
    override fun transform(expr: KFp64Value): KExpr<KFp64Sort> = visitExpr(expr, ::visit)
    override fun transform(expr: KFp128Value): KExpr<KFp128Sort> = visitExpr(expr, ::visit)
    override fun transform(expr: KFpCustomSizeValue): KExpr<KFpSort> = visitExpr(expr, ::visit)

    // fp rounding mode
    fun visit(expr: KFpRoundingModeExpr): V = visitValue(expr)

    override fun transform(expr: KFpRoundingModeExpr): KExpr<KFpRoundingModeSort> =
        visitExpr(expr, ::visit)

    // fp operations visit
    fun <T : KFpSort> visit(expr: KFpAbsExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpNegationExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpAddExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpSubExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpMulExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpDivExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpFusedMulAddExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpSqrtExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpRemExpr<T>): V = visitApp(expr)

    fun <T : KFpSort> visit(expr: KFpRoundToIntegralExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpMinExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpMaxExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpLessOrEqualExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpLessExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpGreaterOrEqualExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpGreaterExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpEqualExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpIsNormalExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpIsSubnormalExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpIsZeroExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpIsInfiniteExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpIsNaNExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpIsNegativeExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpIsPositiveExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpToBvExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpToRealExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpToIEEEBvExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpFromBvExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KFpToFpExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KRealToFpExpr<T>): V = visitApp(expr)
    fun <T : KFpSort> visit(expr: KBvToFpExpr<T>): V = visitApp(expr)

    override fun <T : KFpSort> transform(expr: KFpAbsExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpNegationExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpAddExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpSubExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpMulExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpDivExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpFusedMulAddExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpSqrtExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpRemExpr<T>): KExpr<T> = visitExpr(expr, ::visit)

    override fun <T : KFpSort> transform(expr: KFpRoundToIntegralExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpMinExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpMaxExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpLessOrEqualExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpLessExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpGreaterOrEqualExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpGreaterExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpEqualExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpFromBvExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KFpToFpExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KRealToFpExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KFpSort> transform(expr: KBvToFpExpr<T>): KExpr<T> = visitExpr(expr, ::visit)

    // array visitors
    fun <A : KArraySortBase<R>, R : KSort> visitArrayStore(expr: KArrayStoreBase<A, R>): V =
        visitApp(expr)

    fun <D : KSort, R : KSort> visit(expr: KArrayStore<D, R>): V =
        visitArrayStore(expr)

    fun <D0 : KSort, D1 : KSort, R : KSort> visit(expr: KArray2Store<D0, D1, R>): V =
        visitArrayStore(expr)

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(expr: KArray3Store<D0, D1, D2, R>): V =
        visitArrayStore(expr)

    fun <R : KSort> visit(expr: KArrayNStore<R>): V =
        visitArrayStore(expr)

    fun <A : KArraySortBase<R>, R : KSort> visitArraySelect(expr: KArraySelectBase<A, R>): V =
        visitApp(expr)

    fun <D : KSort, R : KSort> visit(expr: KArraySelect<D, R>): V =
        visitArraySelect(expr)

    fun <D0 : KSort, D1 : KSort, R : KSort> visit(expr: KArray2Select<D0, D1, R>): V =
        visitArraySelect(expr)

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(expr: KArray3Select<D0, D1, D2, R>): V =
        visitArraySelect(expr)

    fun <R : KSort> visit(expr: KArrayNSelect<R>): V =
        visitArraySelect(expr)

    fun <A : KArraySortBase<R>, R : KSort> visit(expr: KArrayConst<A, R>): V =
        visitApp(expr)

    fun <A : KArraySortBase<R>, R : KSort> visit(expr: KFunctionAsArray<A, R>): V =
        visitExpr(expr)

    fun <A : KArraySortBase<R>, R : KSort> visitArrayLambda(expr: KArrayLambdaBase<A, R>): V =
        visitExpr(expr)

    fun <D : KSort, R : KSort> visit(expr: KArrayLambda<D, R>): V =
        visitArrayLambda(expr)

    fun <D0 : KSort, D1 : KSort, R : KSort> visit(expr: KArray2Lambda<D0, D1, R>): V =
        visitArrayLambda(expr)

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(expr: KArray3Lambda<D0, D1, D2, R>): V =
        visitArrayLambda(expr)

    fun <R : KSort> visit(expr: KArrayNLambda<R>): V =
        visitArrayLambda(expr)

    override fun <A : KArraySortBase<R>, R : KSort> transformArrayStore(
        expr: KArrayStoreBase<A, R>
    ): KExpr<A> = visitExpr(expr, ::visitArrayStore)

    override fun <D : KSort, R : KSort> transform(
        expr: KArrayStore<D, R>
    ): KExpr<KArraySort<D, R>> = visitExpr(expr, ::visit)

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Store<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = visitExpr(expr, ::visit)

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Store<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = visitExpr(expr, ::visit)

    override fun <R : KSort> transform(
        expr: KArrayNStore<R>
    ): KExpr<KArrayNSort<R>> = visitExpr(expr, ::visit)

    override fun <A : KArraySortBase<R>, R : KSort> transformArraySelect(
        expr: KArraySelectBase<A, R>
    ): KExpr<R> = visitExpr(expr, ::visitArraySelect)

    override fun <D : KSort, R : KSort> transform(
        expr: KArraySelect<D, R>
    ): KExpr<R> = visitExpr(expr, ::visit)

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Select<D0, D1, R>
    ): KExpr<R> = visitExpr(expr, ::visit)

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Select<D0, D1, D2, R>
    ): KExpr<R> = visitExpr(expr, ::visit)

    override fun <R : KSort> transform(
        expr: KArrayNSelect<R>
    ): KExpr<R> = visitExpr(expr, ::visit)

    override fun <A : KArraySortBase<R>, R : KSort> transform(
        expr: KArrayConst<A, R>
    ): KExpr<A> = visitExpr(expr, ::visit)

    override fun <A : KArraySortBase<R>, R : KSort> transform(
        expr: KFunctionAsArray<A, R>
    ): KExpr<A> = visitExpr(expr, ::visit)

    override fun <A : KArraySortBase<R>, R : KSort> transformArrayLambda(
        expr: KArrayLambdaBase<A, R>
    ): KExpr<A> = visitExpr(expr, ::visitArrayLambda)

    override fun <D : KSort, R : KSort> transform(
        expr: KArrayLambda<D, R>
    ): KExpr<KArraySort<D, R>> = visitExpr(expr, ::visit)

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Lambda<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = visitExpr(expr, ::visit)

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Lambda<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = visitExpr(expr, ::visit)

    override fun <R : KSort> transform(
        expr: KArrayNLambda<R>
    ): KExpr<KArrayNSort<R>> = visitExpr(expr, ::visit)

    // arith visitors
    fun <T : KArithSort> visit(expr: KAddArithExpr<T>): V = visitApp(expr)
    fun <T : KArithSort> visit(expr: KMulArithExpr<T>): V = visitApp(expr)
    fun <T : KArithSort> visit(expr: KSubArithExpr<T>): V = visitApp(expr)
    fun <T : KArithSort> visit(expr: KUnaryMinusArithExpr<T>): V = visitApp(expr)
    fun <T : KArithSort> visit(expr: KDivArithExpr<T>): V = visitApp(expr)
    fun <T : KArithSort> visit(expr: KPowerArithExpr<T>): V = visitApp(expr)
    fun <T : KArithSort> visit(expr: KLtArithExpr<T>): V = visitApp(expr)
    fun <T : KArithSort> visit(expr: KLeArithExpr<T>): V = visitApp(expr)
    fun <T : KArithSort> visit(expr: KGtArithExpr<T>): V = visitApp(expr)
    fun <T : KArithSort> visit(expr: KGeArithExpr<T>): V = visitApp(expr)

    override fun <T : KArithSort> transform(expr: KAddArithExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KArithSort> transform(expr: KMulArithExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KArithSort> transform(expr: KSubArithExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KArithSort> transform(expr: KUnaryMinusArithExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KArithSort> transform(expr: KDivArithExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KArithSort> transform(expr: KPowerArithExpr<T>): KExpr<T> = visitExpr(expr, ::visit)
    override fun <T : KArithSort> transform(expr: KLtArithExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KArithSort> transform(expr: KLeArithExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KArithSort> transform(expr: KGtArithExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun <T : KArithSort> transform(expr: KGeArithExpr<T>): KExpr<KBoolSort> = visitExpr(expr, ::visit)

    // integer visitors
    fun visit(expr: KModIntExpr): V = visitApp(expr)
    fun visit(expr: KRemIntExpr): V = visitApp(expr)
    fun visit(expr: KToRealIntExpr): V = visitApp(expr)

    override fun transform(expr: KModIntExpr): KExpr<KIntSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KRemIntExpr): KExpr<KIntSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KToRealIntExpr): KExpr<KRealSort> = visitExpr(expr, ::visit)

    fun visitIntNum(expr: KIntNumExpr): V = visitValue(expr)
    fun visit(expr: KInt32NumExpr): V = visitIntNum(expr)
    fun visit(expr: KInt64NumExpr): V = visitIntNum(expr)
    fun visit(expr: KIntBigNumExpr): V = visitIntNum(expr)

    override fun transformIntNum(expr: KIntNumExpr): KExpr<KIntSort> = visitExpr(expr, ::visitIntNum)
    override fun transform(expr: KInt32NumExpr): KExpr<KIntSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KInt64NumExpr): KExpr<KIntSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KIntBigNumExpr): KExpr<KIntSort> = visitExpr(expr, ::visit)

    // real visitors
    fun visit(expr: KToIntRealExpr): V = visitApp(expr)
    fun visit(expr: KIsIntRealExpr): V = visitApp(expr)
    fun visit(expr: KRealNumExpr): V = visitValue(expr)

    override fun transform(expr: KToIntRealExpr): KExpr<KIntSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KIsIntRealExpr): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KRealNumExpr): KExpr<KRealSort> = visitExpr(expr, ::visit)

    // string visitors
    fun visit(expr: KStringConcatExpr): V = visitApp(expr)
    fun visit(expr: KStringLenExpr): V = visitApp(expr)
    fun visit(expr: KStringToRegexExpr): V = visitApp(expr)
    fun visit(expr: KStringInRegexExpr): V = visitApp(expr)
    fun visit(expr: KStringSuffixOfExpr): V = visitApp(expr)
    fun visit(expr: KStringPrefixOfExpr): V = visitApp(expr)
    fun visit(expr: KStringLtExpr): V = visitApp(expr)
    fun visit(expr: KStringLeExpr): V = visitApp(expr)
    fun visit(expr: KStringGtExpr): V = visitApp(expr)
    fun visit(expr: KStringGeExpr): V = visitApp(expr)
    fun visit(expr: KStringContainsExpr): V = visitApp(expr)
    fun visit(expr: KStringSingletonSubExpr): V = visitApp(expr)
    fun visit(expr: KStringSubExpr): V = visitApp(expr)
    fun visit(expr: KStringIndexOfExpr): V = visitApp(expr)
    fun visit(expr: KStringReplaceExpr): V = visitApp(expr)
    fun visit(expr: KStringReplaceAllExpr): V = visitApp(expr)
    fun visit(expr: KStringReplaceWithRegexExpr): V = visitApp(expr)
    fun visit(expr: KStringReplaceAllWithRegexExpr): V = visitApp(expr)
    fun visit(expr: KStringIsDigitExpr): V = visitApp(expr)
    fun visit(expr: KStringToCodeExpr): V = visitApp(expr)
    fun visit(expr: KStringFromCodeExpr): V = visitApp(expr)
    fun visit(expr: KStringToIntExpr): V = visitApp(expr)
    fun visit(expr: KStringFromIntExpr): V = visitApp(expr)
    fun visit(expr: KStringLiteralExpr): V = visitValue(expr)

    override fun transform(expr: KStringConcatExpr): KExpr<KStringSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringLenExpr): KExpr<KIntSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringToRegexExpr): KExpr<KRegexSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringInRegexExpr): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringSuffixOfExpr): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringPrefixOfExpr): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringLtExpr): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringLeExpr): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringGtExpr): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringGeExpr): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringContainsExpr): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringSingletonSubExpr): KExpr<KStringSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringSubExpr): KExpr<KStringSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringIndexOfExpr): KExpr<KIntSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringReplaceExpr): KExpr<KStringSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringReplaceAllExpr): KExpr<KStringSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringReplaceWithRegexExpr): KExpr<KStringSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringReplaceAllWithRegexExpr): KExpr<KStringSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringIsDigitExpr): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringToCodeExpr): KExpr<KIntSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringFromCodeExpr): KExpr<KStringSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringToIntExpr): KExpr<KIntSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringFromIntExpr): KExpr<KStringSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KStringLiteralExpr): KExpr<KStringSort> = visitExpr(expr, ::visit)

    // regex visitors
    fun visit(expr: KRegexConcatExpr): V = visitApp(expr)
    fun visit(expr: KRegexUnionExpr): V = visitApp(expr)
    fun visit(expr: KRegexIntersectionExpr): V = visitApp(expr)
    fun visit(expr: KRegexStarExpr): V = visitApp(expr)
    fun visit(expr: KRegexCrossExpr): V = visitApp(expr)
    fun visit(expr: KRegexDifferenceExpr): V = visitApp(expr)
    fun visit(expr: KRegexComplementExpr): V = visitApp(expr)
    fun visit(expr: KRegexOptionExpr): V = visitApp(expr)
    fun visit(expr: KRegexRangeExpr): V = visitApp(expr)
    fun visit(expr: KRegexEpsilon): V = visitValue(expr)
    fun visit(expr: KRegexAll): V = visitValue(expr)
    fun visit(expr: KRegexAllChar): V = visitValue(expr)

    override fun transform(expr: KRegexConcatExpr): KExpr<KRegexSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KRegexUnionExpr): KExpr<KRegexSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KRegexIntersectionExpr): KExpr<KRegexSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KRegexStarExpr): KExpr<KRegexSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KRegexCrossExpr): KExpr<KRegexSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KRegexDifferenceExpr): KExpr<KRegexSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KRegexComplementExpr): KExpr<KRegexSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KRegexOptionExpr): KExpr<KRegexSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KRegexRangeExpr): KExpr<KRegexSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KRegexEpsilon): KExpr<KRegexSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KRegexAll): KExpr<KRegexSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KRegexAllChar): KExpr<KRegexSort> = visitExpr(expr, ::visit)

    // quantifier visitors
    fun visit(expr: KExistentialQuantifier): V = visitExpr(expr)
    fun visit(expr: KUniversalQuantifier): V = visitExpr(expr)

    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> = visitExpr(expr, ::visit)
    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> = visitExpr(expr, ::visit)

    // uninterpreted sort value
    fun visit(expr: KUninterpretedSortValue): V = visitValue(expr)

    override fun transform(expr: KUninterpretedSortValue): KExpr<KUninterpretedSort> = visitExpr(expr, ::visit)

}

inline fun <V, E : KExpr<*>> KVisitor<V>.visitExpr(expr: E, visitor: (E) -> V): E {
    val result = visitor(expr)
    exprVisitResult(expr, result)
    return expr
}
