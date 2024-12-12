package io.ksmt.expr.transformer

import io.ksmt.KContext
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


interface KTransformer : KTransformerBase {
    val ctx: KContext

    override fun transform(expr: KExpr<*>): Any = error("transformer is not implemented for expr $expr")

    fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> = expr

    // function transformers
    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> = transformApp(expr)
    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = transform(expr as KFunctionApp<T>)
    fun <T : KSort, A : KSort> transformApp(expr: KApp<T, A>): KExpr<T> = with(ctx) {
        val args = expr.args.map { it.accept(this@KTransformer) }

        return if (args == expr.args) {
            transformExpr(expr)
        } else {
            transformExpr(mkApp(expr.decl, args))
        }
    }

    fun <T : KSort> transformValue(expr: KInterpretedValue<T>): KExpr<T> = transformApp(expr)

    // bool transformers
    override fun transform(expr: KAndExpr): KExpr<KBoolSort> = transformApp(expr)
    override fun transform(expr: KAndBinaryExpr): KExpr<KBoolSort> = transform(expr as KAndExpr)
    override fun transform(expr: KOrExpr): KExpr<KBoolSort> = transformApp(expr)
    override fun transform(expr: KOrBinaryExpr): KExpr<KBoolSort> = transform(expr as KOrExpr)
    override fun transform(expr: KNotExpr): KExpr<KBoolSort> = transformApp(expr)
    override fun transform(expr: KImpliesExpr): KExpr<KBoolSort> = transformApp(expr)
    override fun transform(expr: KXorExpr): KExpr<KBoolSort> = transformApp(expr)
    override fun transform(expr: KTrue): KExpr<KBoolSort> = transformValue(expr)
    override fun transform(expr: KFalse): KExpr<KBoolSort> = transformValue(expr)
    override fun <T : KSort> transform(expr: KEqExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KSort> transform(expr: KDistinctExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KSort> transform(expr: KIteExpr<T>): KExpr<T> = transformApp(expr)

    // bit-vec transformers
    fun <T : KBvSort> transformBitVecValue(expr: KBitVecValue<T>): KExpr<T> = transformValue(expr)
    override fun transform(expr: KBitVec1Value): KExpr<KBv1Sort> = transformBitVecValue(expr)
    override fun transform(expr: KBitVec8Value): KExpr<KBv8Sort> = transformBitVecValue(expr)
    override fun transform(expr: KBitVec16Value): KExpr<KBv16Sort> = transformBitVecValue(expr)
    override fun transform(expr: KBitVec32Value): KExpr<KBv32Sort> = transformBitVecValue(expr)
    override fun transform(expr: KBitVec64Value): KExpr<KBv64Sort> = transformBitVecValue(expr)
    override fun transform(expr: KBitVecCustomValue): KExpr<KBvSort> = transformBitVecValue(expr)

    // bit-vec expressions transformers
    override fun <T : KBvSort> transform(expr: KBvNotExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>): KExpr<KBv1Sort> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>): KExpr<KBv1Sort> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> = transformApp(expr)
    override fun transform(expr: KBvExtractExpr): KExpr<KBvSort> = transformApp(expr)
    override fun transform(expr: KBvSignExtensionExpr): KExpr<KBvSort> = transformApp(expr)
    override fun transform(expr: KBvZeroExtensionExpr): KExpr<KBvSort> = transformApp(expr)
    override fun transform(expr: KBvRepeatExpr): KExpr<KBvSort> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>): KExpr<T> = transformApp(expr)
    override fun transform(expr: KBv2IntExpr): KExpr<KIntSort> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>): KExpr<KBoolSort> = transformApp(expr)

    // fp value transformers
    fun <T : KFpSort> transformFpValue(expr: KFpValue<T>): KExpr<T> = transformValue(expr)
    override fun transform(expr: KFp16Value): KExpr<KFp16Sort> = transformFpValue(expr)
    override fun transform(expr: KFp32Value): KExpr<KFp32Sort> = transformFpValue(expr)
    override fun transform(expr: KFp64Value): KExpr<KFp64Sort> = transformFpValue(expr)
    override fun transform(expr: KFp128Value): KExpr<KFp128Sort> = transformFpValue(expr)
    override fun transform(expr: KFpCustomSizeValue): KExpr<KFpSort> = transformFpValue(expr)

    // fp rounding mode
    override fun transform(expr: KFpRoundingModeExpr): KExpr<KFpRoundingModeSort> = transformValue(expr)

    // fp operations tranformation
    override fun <T : KFpSort> transform(expr: KFpAbsExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpNegationExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpAddExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpSubExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpMulExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpDivExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpFusedMulAddExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpSqrtExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpRemExpr<T>): KExpr<T> = transformApp(expr)

    @Suppress("MaxLineLength")
    override fun <T : KFpSort> transform(expr: KFpRoundToIntegralExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpMinExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpMaxExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpLessOrEqualExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpLessExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpGreaterOrEqualExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpGreaterExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpEqualExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpFromBvExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KFpToFpExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KRealToFpExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KFpSort> transform(expr: KBvToFpExpr<T>): KExpr<T> = transformApp(expr)

    // array transformers
    fun <A : KArraySortBase<R>, R : KSort> transformArrayStore(expr: KArrayStoreBase<A, R>): KExpr<A> =
        transformApp(expr)

    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>): KExpr<KArraySort<D, R>> =
        transformArrayStore(expr)

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Store<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = transformArrayStore(expr)

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Store<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = transformArrayStore(expr)

    override fun <R : KSort> transform(expr: KArrayNStore<R>): KExpr<KArrayNSort<R>> =
        transformArrayStore(expr)

    fun <A : KArraySortBase<R>, R : KSort> transformArraySelect(expr: KArraySelectBase<A, R>): KExpr<R> =
        transformApp(expr)

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> =
        transformArraySelect(expr)

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(expr: KArray2Select<D0, D1, R>): KExpr<R> =
        transformArraySelect(expr)

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Select<D0, D1, D2, R>
    ): KExpr<R> = transformArraySelect(expr)

    override fun <R : KSort> transform(expr: KArrayNSelect<R>): KExpr<R> =
        transformArraySelect(expr)

    override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KArrayConst<A, R>): KExpr<A> = transformApp(expr)

    override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KFunctionAsArray<A, R>): KExpr<A> =
        transformExpr(expr)

    fun <A : KArraySortBase<R>, R : KSort> transformArrayLambda(expr: KArrayLambdaBase<A, R>): KExpr<A> =
        transformExpr(expr)

    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>> = with(ctx) {
        val body = expr.body.accept(this@KTransformer)
        if (body == expr.body) return transformArrayLambda(expr)
        return transformArrayLambda(mkArrayLambda(expr.indexVarDecl, body))
    }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Lambda<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = with(ctx) {
        val body = expr.body.accept(this@KTransformer)
        if (body == expr.body) return transformArrayLambda(expr)
        return transformArrayLambda(mkArrayLambda(expr.indexVar0Decl, expr.indexVar1Decl, body))
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Lambda<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = with(ctx) {
        val body = expr.body.accept(this@KTransformer)
        if (body == expr.body) return transformArrayLambda(expr)
        return transformArrayLambda(mkArrayLambda(expr.indexVar0Decl, expr.indexVar1Decl, expr.indexVar2Decl, body))
    }

    override fun <R : KSort> transform(expr: KArrayNLambda<R>): KExpr<KArrayNSort<R>> = with(ctx) {
        val body = expr.body.accept(this@KTransformer)
        if (body == expr.body) return transformArrayLambda(expr)
        return transformArrayLambda(mkArrayNLambda(expr.indexVarDeclarations, body))
    }

    // arith transformers
    override fun <T : KArithSort> transform(expr: KAddArithExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KArithSort> transform(expr: KMulArithExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KArithSort> transform(expr: KSubArithExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KArithSort> transform(expr: KUnaryMinusArithExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KArithSort> transform(expr: KDivArithExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KArithSort> transform(expr: KPowerArithExpr<T>): KExpr<T> = transformApp(expr)
    override fun <T : KArithSort> transform(expr: KLtArithExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KArithSort> transform(expr: KLeArithExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KArithSort> transform(expr: KGtArithExpr<T>): KExpr<KBoolSort> = transformApp(expr)
    override fun <T : KArithSort> transform(expr: KGeArithExpr<T>): KExpr<KBoolSort> = transformApp(expr)

    // integer transformers
    override fun transform(expr: KModIntExpr): KExpr<KIntSort> = transformApp(expr)
    override fun transform(expr: KRemIntExpr): KExpr<KIntSort> = transformApp(expr)
    override fun transform(expr: KToRealIntExpr): KExpr<KRealSort> = transformApp(expr)

    fun transformIntNum(expr: KIntNumExpr): KExpr<KIntSort> = transformValue(expr)
    override fun transform(expr: KInt32NumExpr): KExpr<KIntSort> = transformIntNum(expr)
    override fun transform(expr: KInt64NumExpr): KExpr<KIntSort> = transformIntNum(expr)
    override fun transform(expr: KIntBigNumExpr): KExpr<KIntSort> = transformIntNum(expr)

    // real transformers
    override fun transform(expr: KToIntRealExpr): KExpr<KIntSort> = transformApp(expr)
    override fun transform(expr: KIsIntRealExpr): KExpr<KBoolSort> = transformApp(expr)
    override fun transform(expr: KRealNumExpr): KExpr<KRealSort> = transformValue(expr)

    // string transformers
    override fun transform(expr: KStringConcatExpr): KExpr<KStringSort> = transformApp(expr)
    override fun transform(expr: KStringLenExpr): KExpr<KIntSort> = transformApp(expr)
    override fun transform(expr: KStringToRegexExpr): KExpr<KRegexSort> = transformApp(expr)
    override fun transform(expr: KStringInRegexExpr): KExpr<KBoolSort> = transformApp(expr)
    override fun transform(expr: KStringSuffixOfExpr): KExpr<KBoolSort> = transformApp(expr)
    override fun transform(expr: KStringPrefixOfExpr): KExpr<KBoolSort> = transformApp(expr)
    override fun transform(expr: KStringLtExpr): KExpr<KBoolSort> = transformApp(expr)
    override fun transform(expr: KStringLeExpr): KExpr<KBoolSort> = transformApp(expr)
    override fun transform(expr: KStringGtExpr): KExpr<KBoolSort> = transformApp(expr)
    override fun transform(expr: KStringGeExpr): KExpr<KBoolSort> = transformApp(expr)
    override fun transform(expr: KStringContainsExpr): KExpr<KBoolSort> = transformApp(expr)
    override fun transform(expr: KStringSingletonSubExpr): KExpr<KStringSort> = transformApp(expr)
    override fun transform(expr: KStringSubExpr): KExpr<KStringSort> = transformApp(expr)
    override fun transform(expr: KStringIndexOfExpr): KExpr<KIntSort> = transformApp(expr)
    override fun transform(expr: KStringIndexOfRegexExpr): KExpr<KIntSort> = transformApp(expr)
    override fun transform(expr: KStringReplaceExpr): KExpr<KStringSort> = transformApp(expr)
    override fun transform(expr: KStringReplaceAllExpr): KExpr<KStringSort> = transformApp(expr)
    override fun transform(expr: KStringReplaceWithRegexExpr): KExpr<KStringSort> = transformApp(expr)
    override fun transform(expr: KStringReplaceAllWithRegexExpr): KExpr<KStringSort> = transformApp(expr)
    override fun transform(expr: KStringToLowerExpr): KExpr<KStringSort> = transformApp(expr)
    override fun transform(expr: KStringToUpperExpr): KExpr<KStringSort> = transformApp(expr)
    override fun transform(expr: KStringReverseExpr): KExpr<KStringSort> = transformApp(expr)
    override fun transform(expr: KStringIsDigitExpr): KExpr<KBoolSort> = transformApp(expr)
    override fun transform(expr: KStringToCodeExpr): KExpr<KIntSort> = transformApp(expr)
    override fun transform(expr: KStringFromCodeExpr): KExpr<KStringSort> = transformApp(expr)
    override fun transform(expr: KStringToIntExpr): KExpr<KIntSort> = transformApp(expr)
    override fun transform(expr: KStringFromIntExpr): KExpr<KStringSort> = transformApp(expr)
    override fun transform(expr: KStringLiteralExpr): KExpr<KStringSort> = transformValue(expr)

    // regex transformers
    override fun transform(expr: KRegexConcatExpr): KExpr<KRegexSort> = transformApp(expr)
    override fun transform(expr: KRegexUnionExpr): KExpr<KRegexSort> = transformApp(expr)
    override fun transform(expr: KRegexIntersectionExpr): KExpr<KRegexSort> = transformApp(expr)
    override fun transform(expr: KRegexStarExpr): KExpr<KRegexSort> = transformApp(expr)
    override fun transform(expr: KRegexCrossExpr): KExpr<KRegexSort> = transformApp(expr)
    override fun transform(expr: KRegexDifferenceExpr): KExpr<KRegexSort> = transformApp(expr)
    override fun transform(expr: KRegexComplementExpr): KExpr<KRegexSort> = transformApp(expr)
    override fun transform(expr: KRegexOptionExpr): KExpr<KRegexSort> = transformApp(expr)
    override fun transform(expr: KRegexRangeExpr): KExpr<KRegexSort> = transformApp(expr)
    override fun transform(expr: KRegexEpsilon): KExpr<KRegexSort> = transformValue(expr)
    override fun transform(expr: KRegexAll): KExpr<KRegexSort> = transformValue(expr)
    override fun transform(expr: KRegexAllChar): KExpr<KRegexSort> = transformValue(expr)

    // quantifier transformers
    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> = with(ctx) {
        val body = expr.body.accept(this@KTransformer)

        return if (body == expr.body) {
            transformExpr(expr)
        } else {
            transformExpr(mkExistentialQuantifier(body, expr.bounds))
        }
    }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> = with(ctx) {
        val body = expr.body.accept(this@KTransformer)

        return if (body == expr.body) {
            transformExpr(expr)
        } else {
            transformExpr(mkUniversalQuantifier(body, expr.bounds))
        }
    }

    // uninterpreted sort value
    override fun transform(expr: KUninterpretedSortValue): KExpr<KUninterpretedSort> = transformValue(expr)
}
