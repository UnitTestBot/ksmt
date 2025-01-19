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
import io.ksmt.expr.KArrayNLambda
import io.ksmt.expr.KArrayNSelect
import io.ksmt.expr.KArrayNStore
import io.ksmt.expr.KArraySelect
import io.ksmt.expr.KArrayStore
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
import io.ksmt.expr.KFpAbsExpr
import io.ksmt.expr.KFpAddExpr
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
import io.ksmt.expr.KFpSqrtExpr
import io.ksmt.expr.KFpSubExpr
import io.ksmt.expr.KFpToBvExpr
import io.ksmt.expr.KFpToFpExpr
import io.ksmt.expr.KFpToIEEEBvExpr
import io.ksmt.expr.KFpToRealExpr
import io.ksmt.expr.KFunctionApp
import io.ksmt.expr.KGeArithExpr
import io.ksmt.expr.KGtArithExpr
import io.ksmt.expr.KImpliesExpr
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
import io.ksmt.expr.KRealToFpExpr
import io.ksmt.expr.KRemIntExpr
import io.ksmt.expr.KSubArithExpr
import io.ksmt.expr.KToIntRealExpr
import io.ksmt.expr.KToRealIntExpr
import io.ksmt.expr.KUnaryMinusArithExpr
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
import io.ksmt.sort.KArithSort
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KStringSort
import io.ksmt.sort.KRegexSort
import io.ksmt.sort.KSort
import io.ksmt.utils.uncheckedCast

/**
 * Apply specialized non-recursive transformations for all KSMT expressions.
 * See [KNonRecursiveTransformerBase] for details.
 * */
abstract class KNonRecursiveTransformer(override val ctx: KContext) : KNonRecursiveTransformerBase(), KTransformer {

    override fun <T : KSort, A : KSort> transformApp(expr: KApp<T, A>): KExpr<T> = transformExpr(expr)

    // function transformers
    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.args, ::transformApp) { transformedArgs ->
            expr.decl.apply(transformedArgs)
        }

    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = transformApp(expr)

    // bool transformers
    override fun transform(expr: KAndExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.args, ::transformApp
        ) { args -> mkAnd(args, flat = false, order = false) }

    override fun transform(expr: KAndBinaryExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.lhs, expr.rhs, ::transformApp
        ) { l, r -> mkAnd(l, r, flat = false, order = false) }

    override fun transform(expr: KOrExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.args, ::transformApp
        ) { args -> mkOr(args, flat = false) }

    override fun transform(expr: KOrBinaryExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.lhs, expr.rhs, ::transformApp
        ) { l, r -> mkOr(l, r, flat = false, order = false) }

    override fun transform(expr: KNotExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.arg, ::transformApp, KContext::mkNot)

    override fun transform(expr: KImpliesExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.p, expr.q, ::transformApp, KContext::mkImplies)

    override fun transform(expr: KXorExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.a, expr.b, ::transformApp, KContext::mkXor)

    override fun <T : KSort> transform(expr: KEqExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.lhs, expr.rhs, ::transformApp
        ) { l, r -> mkEq(l, r, order = false) }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.args, ::transformApp
        ) { args -> mkDistinct(args, order = false) }

    override fun <T : KSort> transform(expr: KIteExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(
            expr,
            expr.condition,
            expr.trueBranch,
            expr.falseBranch,
            ::transformApp,
            KContext::mkIte
        )

    // bit-vec expressions transformers
    override fun <T : KBvSort> transform(expr: KBvNotExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp, KContext::mkBvNotExpr)

    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>): KExpr<KBv1Sort> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp, KContext::mkBvReductionAndExpr)

    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>): KExpr<KBv1Sort> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp, KContext::mkBvReductionOrExpr)

    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvAndExpr)

    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvOrExpr)

    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvXorExpr)

    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvNAndExpr)

    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvNorExpr)

    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvXNorExpr)

    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp, KContext::mkBvNegationExpr)

    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvAddExpr)

    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvSubExpr)

    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvMulExpr)

    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvUnsignedDivExpr)

    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvSignedDivExpr)

    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvUnsignedRemExpr)

    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvSignedRemExpr)

    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvSignedModExpr)

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvUnsignedLessExpr)

    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvSignedLessExpr)

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvUnsignedLessOrEqualExpr
        )

    override fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvSignedLessOrEqualExpr
        )

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvUnsignedGreaterOrEqualExpr
        )

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvSignedGreaterOrEqualExpr
        )

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvUnsignedGreaterExpr
        )

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvSignedGreaterExpr
        )

    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvConcatExpr)

    override fun transform(expr: KBvExtractExpr): KExpr<KBvSort> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp) { value ->
            mkBvExtractExpr(expr.high, expr.low, value)
        }

    override fun transform(expr: KBvSignExtensionExpr): KExpr<KBvSort> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp) { value ->
            mkBvSignExtensionExpr(expr.extensionSize, value)
        }

    override fun transform(expr: KBvZeroExtensionExpr): KExpr<KBvSort> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp) { value ->
            mkBvZeroExtensionExpr(expr.extensionSize, value)
        }

    override fun transform(expr: KBvRepeatExpr): KExpr<KBvSort> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp) { value ->
            mkBvRepeatExpr(expr.repeatNumber, value)
        }

    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg, expr.shift, ::transformApp, KContext::mkBvShiftLeftExpr)

    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(
            expr, expr.arg, expr.shift, ::transformApp, KContext::mkBvLogicalShiftRightExpr
        )

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(
            expr, expr.arg, expr.shift, ::transformApp, KContext::mkBvArithShiftRightExpr
        )

    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(
            expr, expr.arg, expr.rotation, ::transformApp, KContext::mkBvRotateLeftExpr
        )

    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp) { value ->
            mkBvRotateLeftIndexedExpr(expr.rotationNumber, value)
        }

    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(
            expr, expr.arg, expr.rotation, ::transformApp, KContext::mkBvRotateRightExpr
        )

    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp) { value ->
            mkBvRotateRightIndexedExpr(expr.rotationNumber, value)
        }

    override fun transform(expr: KBv2IntExpr): KExpr<KIntSort> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp) { value ->
            mkBv2IntExpr(value, expr.isSigned)
        }

    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp) { arg0, arg1 ->
            mkBvAddNoOverflowExpr(arg0, arg1, expr.isSigned)
        }

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvAddNoUnderflowExpr
        )

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvSubNoOverflowExpr
        )

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp) { arg0, arg1 ->
            mkBvSubNoUnderflowExpr(arg0, arg1, expr.isSigned)
        }

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvDivNoOverflowExpr
        )

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp, KContext::mkBvNegationNoOverflowExpr)

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp) { arg0, arg1 ->
            mkBvMulNoOverflowExpr(arg0, arg1, expr.isSigned)
        }

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkBvMulNoUnderflowExpr
        )

    // fp operations tranformation
    override fun <T : KFpSort> transform(expr: KFpAbsExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp, KContext::mkFpAbsExpr)

    override fun <T : KFpSort> transform(expr: KFpNegationExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp, KContext::mkFpNegationExpr)

    override fun <T : KFpSort> transform(expr: KFpAddExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(
            expr, expr.roundingMode, expr.arg0, expr.arg1, ::transformApp, KContext::mkFpAddExpr
        )

    override fun <T : KFpSort> transform(expr: KFpSubExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(
            expr, expr.roundingMode, expr.arg0, expr.arg1, ::transformApp, KContext::mkFpSubExpr
        )

    override fun <T : KFpSort> transform(expr: KFpMulExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(
            expr, expr.roundingMode, expr.arg0, expr.arg1, ::transformApp, KContext::mkFpMulExpr
        )

    override fun <T : KFpSort> transform(expr: KFpDivExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(
            expr, expr.roundingMode, expr.arg0, expr.arg1, ::transformApp, KContext::mkFpDivExpr
        )

    override fun <T : KFpSort> transform(expr: KFpFusedMulAddExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(
            expr, expr.roundingMode, expr.arg0, expr.arg1, expr.arg2, ::transformApp, KContext::mkFpFusedMulAddExpr
        )

    override fun <T : KFpSort> transform(expr: KFpSqrtExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(
            expr, expr.roundingMode, expr.value, ::transformApp, KContext::mkFpSqrtExpr
        )

    override fun <T : KFpSort> transform(expr: KFpRemExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkFpRemExpr)

    override fun <T : KFpSort> transform(expr: KFpRoundToIntegralExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(
            expr, expr.roundingMode, expr.value, ::transformApp, KContext::mkFpRoundToIntegralExpr
        )

    override fun <T : KFpSort> transform(expr: KFpMinExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkFpMinExpr)

    override fun <T : KFpSort> transform(expr: KFpMaxExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkFpMaxExpr)

    override fun <T : KFpSort> transform(expr: KFpLessOrEqualExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkFpLessOrEqualExpr)

    override fun <T : KFpSort> transform(expr: KFpLessExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkFpLessExpr)

    override fun <T : KFpSort> transform(expr: KFpGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkFpGreaterOrEqualExpr
        )

    override fun <T : KFpSort> transform(expr: KFpGreaterExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkFpGreaterExpr)

    override fun <T : KFpSort> transform(expr: KFpEqualExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkFpEqualExpr)

    override fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp, KContext::mkFpIsNormalExpr)

    override fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp, KContext::mkFpIsSubnormalExpr)

    override fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp, KContext::mkFpIsZeroExpr)

    override fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp, KContext::mkFpIsInfiniteExpr)

    override fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp, KContext::mkFpIsNaNExpr)

    override fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp, KContext::mkFpIsNegativeExpr)

    override fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp, KContext::mkFpIsPositiveExpr)

    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort> =
        transformExprAfterTransformedDefault(expr, expr.roundingMode, expr.value, ::transformApp) { rm, value ->
            mkFpToBvExpr(rm, value, expr.bvSize, expr.isSigned)
        }

    override fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp, KContext::mkFpToRealExpr)

    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp, KContext::mkFpToIEEEBvExpr)

    override fun <T : KFpSort> transform(expr: KFpFromBvExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(
            expr, expr.sign, expr.biasedExponent, expr.significand, ::transformApp, KContext::mkFpFromBvExpr
        )

    override fun <T : KFpSort> transform(expr: KFpToFpExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(
            expr, expr.roundingMode, expr.value, ::transformApp
        ) { rm, value -> mkFpToFpExpr(expr.sort, rm, value) }

    override fun <T : KFpSort> transform(expr: KRealToFpExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(
            expr, expr.roundingMode, expr.value, ::transformApp
        ) { rm, value -> mkRealToFpExpr(expr.sort, rm, value) }

    override fun <T : KFpSort> transform(expr: KBvToFpExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(
            expr, expr.roundingMode, expr.value, ::transformApp
        ) { rm, value -> mkBvToFpExpr(expr.sort, rm, value, expr.signed) }

    // array transformers
    override fun <D : KSort, R : KSort> transform(
        expr: KArrayStore<D, R>
    ): KExpr<KArraySort<D, R>> = transformExprAfterTransformedDefault(
        expr, expr.array, expr.index, expr.value, ::transformArrayStore, KContext::mkArrayStore
    )

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Store<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = transformExprAfterTransformedDefault(
        expr, expr.array, expr.index0, expr.index1, expr.value, ::transformArrayStore, KContext::mkArrayStore
    )

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Store<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = transformExprAfterTransformedDefault(
        expr, expr.array, expr.index0, expr.index1, expr.index2, expr.value,
        ::transformArrayStore, KContext::mkArrayStore
    )

    override fun <R : KSort> transform(
        expr: KArrayNStore<R>
    ): KExpr<KArrayNSort<R>> = transformExprAfterTransformedDefault(
        expr, expr.args, ::transformArrayStore
    ) { args ->
        mkArrayNStore(
            array = args.first().uncheckedCast(),
            indices = args.subList(fromIndex = 1, toIndex = args.size - 1).uncheckedCast(),
            value = args.last().uncheckedCast()
        )
    }

    override fun <D : KSort, R : KSort> transform(
        expr: KArraySelect<D, R>
    ): KExpr<R> = transformExprAfterTransformedDefault(
        expr, expr.array, expr.index, ::transformArraySelect, KContext::mkArraySelect
    )

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Select<D0, D1, R>
    ): KExpr<R> = transformExprAfterTransformedDefault(
        expr, expr.array, expr.index0, expr.index1, ::transformArraySelect, KContext::mkArraySelect
    )

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Select<D0, D1, D2, R>
    ): KExpr<R> = transformExprAfterTransformedDefault(
        expr, expr.array, expr.index0, expr.index1, expr.index2, ::transformArraySelect, KContext::mkArraySelect
    )

    override fun <R : KSort> transform(
        expr: KArrayNSelect<R>
    ): KExpr<R> = transformExprAfterTransformedDefault(
        expr, expr.args, ::transformArraySelect
    ) { args ->
        mkArrayNSelect(
            array = args.first().uncheckedCast(),
            indices = args.drop(1)
        )
    }

    override fun <D : KSort, R : KSort> transform(
        expr: KArrayLambda<D, R>
    ): KExpr<KArraySort<D, R>> = transformExprAfterTransformedDefault(
        expr, expr.body, ::transformArrayLambda
    ) { body ->
        mkArrayLambda(expr.indexVarDecl, body)
    }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Lambda<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = transformExprAfterTransformedDefault(
        expr, expr.body, ::transformArrayLambda
    ) { body ->
        mkArrayLambda(expr.indexVar0Decl, expr.indexVar1Decl, body)
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Lambda<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = transformExprAfterTransformedDefault(
        expr, expr.body, ::transformArrayLambda
    ) { body ->
        mkArrayLambda(expr.indexVar0Decl, expr.indexVar1Decl, expr.indexVar2Decl, body)
    }

    override fun <R : KSort> transform(
        expr: KArrayNLambda<R>
    ): KExpr<KArrayNSort<R>> = transformExprAfterTransformedDefault(
        expr, expr.body, ::transformArrayLambda
    ) { body ->
        mkArrayNLambda(expr.indexVarDeclarations, body)
    }

    override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KArrayConst<A, R>): KExpr<A> =
        transformExprAfterTransformedDefault(expr, expr.value, ::transformApp) { value ->
            mkArrayConst(expr.sort, value)
        }

    // arith transformers
    override fun <T : KArithSort> transform(expr: KAddArithExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.args, ::transformApp, KContext::mkArithAdd)

    override fun <T : KArithSort> transform(expr: KMulArithExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.args, ::transformApp, KContext::mkArithMul)

    override fun <T : KArithSort> transform(expr: KSubArithExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.args, ::transformApp, KContext::mkArithSub)

    override fun <T : KArithSort> transform(expr: KUnaryMinusArithExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.arg, ::transformApp, KContext::mkArithUnaryMinus)

    override fun <T : KArithSort> transform(expr: KDivArithExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.lhs, expr.rhs, ::transformApp, KContext::mkArithDiv)

    override fun <T : KArithSort> transform(expr: KPowerArithExpr<T>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.lhs, expr.rhs, ::transformApp, KContext::mkArithPower)

    override fun <T : KArithSort> transform(expr: KLtArithExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.lhs, expr.rhs, ::transformApp, KContext::mkArithLt)

    override fun <T : KArithSort> transform(expr: KLeArithExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.lhs, expr.rhs, ::transformApp, KContext::mkArithLe)

    override fun <T : KArithSort> transform(expr: KGtArithExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.lhs, expr.rhs, ::transformApp, KContext::mkArithGt)

    override fun <T : KArithSort> transform(expr: KGeArithExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.lhs, expr.rhs, ::transformApp, KContext::mkArithGe)

    // integer transformers
    override fun transform(expr: KModIntExpr): KExpr<KIntSort> =
        transformExprAfterTransformedDefault(expr, expr.lhs, expr.rhs, ::transformApp, KContext::mkIntMod)

    override fun transform(expr: KRemIntExpr): KExpr<KIntSort> =
        transformExprAfterTransformedDefault(expr, expr.lhs, expr.rhs, ::transformApp, KContext::mkIntRem)

    override fun transform(expr: KToRealIntExpr): KExpr<KRealSort> =
        transformExprAfterTransformedDefault(expr, expr.arg, ::transformApp, KContext::mkIntToReal)

    // real transformers
    override fun transform(expr: KToIntRealExpr): KExpr<KIntSort> =
        transformExprAfterTransformedDefault(expr, expr.arg, ::transformApp, KContext::mkRealToInt)

    override fun transform(expr: KIsIntRealExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.arg, ::transformApp, KContext::mkRealIsInt)

    // string transformers
    override fun transform(expr: KStringConcatExpr): KExpr<KStringSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkStringConcat
        )

    override fun transform(expr: KStringLenExpr): KExpr<KIntSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg, ::transformApp, KContext::mkStringLen
        )

    override fun transform(expr: KStringToRegexExpr): KExpr<KRegexSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg, ::transformApp, KContext::mkStringToRegex
        )

    override fun transform(expr: KStringInRegexExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkStringInRegex
        )

    override fun transform(expr: KStringSuffixOfExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkStringSuffixOf
        )

    override fun transform(expr: KStringPrefixOfExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkStringPrefixOf
        )

    override fun transform(expr: KStringLtExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkStringLt
        )

    override fun transform(expr: KStringLeExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkStringLe
        )

    override fun transform(expr: KStringGtExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkStringGt
        )

    override fun transform(expr: KStringGeExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkStringGe
        )

    override fun transform(expr: KStringContainsExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkStringContains
        )

    override fun transform(expr: KStringSingletonSubExpr): KExpr<KStringSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkStringSingletonSub
        )

    override fun transform(expr: KStringSubExpr): KExpr<KStringSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, expr.arg2, ::transformApp, KContext::mkStringSub
        )

    override fun transform(expr: KStringIndexOfExpr): KExpr<KIntSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, expr.arg2, ::transformApp, KContext::mkStringIndexOf
        )

    override fun transform(expr: KStringIndexOfRegexExpr): KExpr<KIntSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, expr.arg2, ::transformApp, KContext::mkStringIndexOfRegex
        )

    override fun transform(expr: KStringReplaceExpr): KExpr<KStringSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, expr.arg2, ::transformApp, KContext::mkStringReplace
        )

    override fun transform(expr: KStringReplaceAllExpr): KExpr<KStringSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, expr.arg2, ::transformApp, KContext::mkStringReplaceAll
        )

    override fun transform(expr: KStringReplaceWithRegexExpr): KExpr<KStringSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, expr.arg2, ::transformApp, KContext::mkStringReplaceWithRegex
        )

    override fun transform(expr: KStringReplaceAllWithRegexExpr): KExpr<KStringSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, expr.arg2, ::transformApp, KContext::mkStringReplaceAllWithRegex
        )

    override fun transform(expr: KStringToLowerExpr): KExpr<KStringSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg, ::transformApp, KContext::mkStringToLower
        )

    override fun transform(expr: KStringToUpperExpr): KExpr<KStringSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg, ::transformApp, KContext::mkStringToUpper
        )

    override fun transform(expr: KStringReverseExpr): KExpr<KStringSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg, ::transformApp, KContext::mkStringReverse
        )

    override fun transform(expr: KStringIsDigitExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg, ::transformApp, KContext::mkStringIsDigit
        )

    override fun transform(expr: KStringToCodeExpr): KExpr<KIntSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg, ::transformApp, KContext::mkStringToCode
        )

    override fun transform(expr: KStringFromCodeExpr): KExpr<KStringSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg, ::transformApp, KContext::mkStringFromCode
        )

    override fun transform(expr: KStringToIntExpr): KExpr<KIntSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg, ::transformApp, KContext::mkStringToCode
        )

    override fun transform(expr: KStringFromIntExpr): KExpr<KStringSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg, ::transformApp, KContext::mkStringFromCode
        )

    // regex transformers
    override fun transform(expr: KRegexConcatExpr): KExpr<KRegexSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkRegexConcat
        )

    override fun transform(expr: KRegexUnionExpr): KExpr<KRegexSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkRegexUnion
        )

    override fun transform(expr: KRegexIntersectionExpr): KExpr<KRegexSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkRegexIntersection
        )

    override fun transform(expr: KRegexStarExpr): KExpr<KRegexSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg, ::transformApp, KContext::mkRegexStar
        )

    override fun transform(expr: KRegexCrossExpr): KExpr<KRegexSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg, ::transformApp, KContext::mkRegexCross
        )

    override fun transform(expr: KRegexDifferenceExpr): KExpr<KRegexSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkRegexDifference
        )

    override fun transform(expr: KRegexComplementExpr): KExpr<KRegexSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg, ::transformApp, KContext::mkRegexComplement
        )

    override fun transform(expr: KRegexOptionExpr): KExpr<KRegexSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg, ::transformApp, KContext::mkRegexOption
        )

    override fun transform(expr: KRegexRangeExpr): KExpr<KRegexSort> =
        transformExprAfterTransformedDefault(
            expr, expr.arg0, expr.arg1, ::transformApp, KContext::mkRegexRange
        )

    override fun transform(expr: KRegexPowerExpr): KExpr<KRegexSort> =
        transformExprAfterTransformedDefault(expr, expr.arg, ::transformApp) {
            arg -> mkRegexPower(expr.power, arg)
        }

    override fun transform(expr: KRegexLoopExpr): KExpr<KRegexSort> =
        transformExprAfterTransformedDefault(expr, expr.arg, ::transformApp) {
            arg -> mkRegexLoop(expr.from, expr.to, arg)
        }

    // quantified expressions
    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.body, ::transformExpr) { body ->
            mkExistentialQuantifier(body, expr.bounds)
        }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.body, ::transformExpr) { body ->
            mkUniversalQuantifier(body, expr.bounds)
        }

    // utils
    fun <T : KSort> transformExprDefault(expr: KExpr<T>): KExpr<T> = when (expr) {
        is KInterpretedValue<T> -> transformValue(expr)
        is KApp<T, *> -> transformApp(expr)
        else -> transformExpr(expr)
    }

    /**
     * Transform expressions [dependencies] before expression transformation.
     * If all dependencies remain unchanged after transformation
     * invoke [ifNotTransformed] on the original expression and return it result.
     * Otherwise, apply [transformer] to the modified dependencies.
     * */
    inline fun <In : KExpr<T>, Out : KExpr<T>, T : KSort, A : KSort> transformExprAfterTransformedDefault(
        expr: In,
        dependencies: List<KExpr<A>>,
        ifNotTransformed: (In) -> KExpr<T>,
        transformer: KContext.(List<KExpr<A>>) -> Out
    ): KExpr<T> = transformExprAfterTransformed(expr, dependencies) { transformedDependencies ->
        if (transformedDependencies == dependencies) {
            return ifNotTransformed(expr)
        }

        val transformedExpr = ctx.transformer(transformedDependencies)

        return transformExprDefault(transformedExpr)
    }

    /**
     * Specialized version of [transformExprAfterTransformedDefault] for expression with single argument.
     * */
    inline fun <In : KExpr<T>, Out : KExpr<T>, T : KSort, A : KSort> transformExprAfterTransformedDefault(
        expr: In,
        dependency: KExpr<A>,
        ifNotTransformed: (In) -> KExpr<T>,
        transformer: KContext.(KExpr<A>) -> Out
    ): KExpr<T> = transformExprAfterTransformed(expr, dependency) { td ->
        if (td == dependency) {
            return ifNotTransformed(expr)
        }

        val transformedExpr = ctx.transformer(td)

        return transformExprDefault(transformedExpr)
    }

    /**
     * Specialized version of [transformExprAfterTransformedDefault] for expression with two arguments.
     * */
    inline fun <In : KExpr<T>, Out : KExpr<T>, T : KSort, A0 : KSort, A1 : KSort>
    transformExprAfterTransformedDefault(
        expr: In,
        dependency0: KExpr<A0>,
        dependency1: KExpr<A1>,
        ifNotTransformed: (In) -> KExpr<T>,
        transformer: KContext.(KExpr<A0>, KExpr<A1>) -> Out
    ): KExpr<T> = transformExprAfterTransformed(expr, dependency0, dependency1) { td0, td1 ->
        if (td0 == dependency0 && td1 == dependency1) {
            return ifNotTransformed(expr)
        }

        val transformedExpr = ctx.transformer(td0, td1)

        return transformExprDefault(transformedExpr)
    }

    /**
     * Specialized version of [transformExprAfterTransformedDefault] for expression with three arguments.
     * */
    @Suppress("LongParameterList")
    inline fun <In : KExpr<T>, Out : KExpr<T>, T : KSort, A0 : KSort, A1 : KSort, A2 : KSort>
    transformExprAfterTransformedDefault(
        expr: In,
        dependency0: KExpr<A0>,
        dependency1: KExpr<A1>,
        dependency2: KExpr<A2>,
        ifNotTransformed: (In) -> KExpr<T>,
        transformer: KContext.(KExpr<A0>, KExpr<A1>, KExpr<A2>) -> Out
    ): KExpr<T> = transformExprAfterTransformed(expr, dependency0, dependency1, dependency2) { td0, td1, td2 ->
        if (td0 == dependency0 && td1 == dependency1 && td2 == dependency2) {
            return ifNotTransformed(expr)
        }

        val transformedExpr = ctx.transformer(td0, td1, td2)

        return transformExprDefault(transformedExpr)
    }

    /**
     * Specialized version of [transformExprAfterTransformedDefault] for expression with four arguments.
     * */
    @Suppress("LongParameterList", "ComplexCondition")
    inline fun <In : KExpr<T>, Out : KExpr<T>, T : KSort, A0 : KSort, A1 : KSort, A2 : KSort, A3 : KSort>
    transformExprAfterTransformedDefault(
        expr: In,
        dependency0: KExpr<A0>,
        dependency1: KExpr<A1>,
        dependency2: KExpr<A2>,
        dependency3: KExpr<A3>,
        ifNotTransformed: (In) -> KExpr<T>,
        transformer: KContext.(KExpr<A0>, KExpr<A1>, KExpr<A2>, KExpr<A3>) -> Out
    ): KExpr<T> =
        transformExprAfterTransformed(expr, dependency0, dependency1, dependency2, dependency3) { td0, td1, td2, td3 ->
            if (td0 == dependency0 && td1 == dependency1 && td2 == dependency2 && td3 == dependency3) {
                return ifNotTransformed(expr)
            }

            val transformedExpr = ctx.transformer(td0, td1, td2, td3)

            return transformExprDefault(transformedExpr)
        }

    /**
     * Specialized version of [transformExprAfterTransformedDefault] for expression with five arguments.
     * */
    @Suppress("LongParameterList", "ComplexCondition")
    inline fun <
        In : KExpr<T>, Out : KExpr<T>,
        T : KSort, A0 : KSort, A1 : KSort, A2 : KSort, A3 : KSort, A4 : KSort
    > transformExprAfterTransformedDefault(
        expr: In,
        d0: KExpr<A0>,
        d1: KExpr<A1>,
        d2: KExpr<A2>,
        d3: KExpr<A3>,
        d4: KExpr<A4>,
        ifNotTransformed: (In) -> KExpr<T>,
        transformer: KContext.(KExpr<A0>, KExpr<A1>, KExpr<A2>, KExpr<A3>, KExpr<A4>) -> Out
    ): KExpr<T> =
        transformExprAfterTransformed(
            expr, d0, d1, d2, d3, d4
        ) { td0, td1, td2, td3, td4 ->
            if (td0 == d0 && td1 == d1 && td2 == d2 && td3 == d3 && td4 == d4) {
                return ifNotTransformed(expr)
            }

            val transformedExpr = ctx.transformer(td0, td1, td2, td3, td4)

            return transformExprDefault(transformedExpr)
        }

}
