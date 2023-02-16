package org.ksmt.expr.transformer

import org.ksmt.KContext
import org.ksmt.expr.KAddArithExpr
import org.ksmt.expr.KAndExpr
import org.ksmt.expr.KApp
import org.ksmt.expr.KArrayConst
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KArraySelect
import org.ksmt.expr.KArrayStore
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
import org.ksmt.expr.KFpAbsExpr
import org.ksmt.expr.KFpAddExpr
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
import org.ksmt.expr.KFpSqrtExpr
import org.ksmt.expr.KFpSubExpr
import org.ksmt.expr.KFpToBvExpr
import org.ksmt.expr.KFpToFpExpr
import org.ksmt.expr.KFpToIEEEBvExpr
import org.ksmt.expr.KFpToRealExpr
import org.ksmt.expr.KFunctionApp
import org.ksmt.expr.KGeArithExpr
import org.ksmt.expr.KGtArithExpr
import org.ksmt.expr.KImpliesExpr
import org.ksmt.expr.KInterpretedValue
import org.ksmt.expr.KIsIntRealExpr
import org.ksmt.expr.KIteExpr
import org.ksmt.expr.KLeArithExpr
import org.ksmt.expr.KLtArithExpr
import org.ksmt.expr.KModIntExpr
import org.ksmt.expr.KMulArithExpr
import org.ksmt.expr.KNotExpr
import org.ksmt.expr.KOrExpr
import org.ksmt.expr.KPowerArithExpr
import org.ksmt.expr.KRealToFpExpr
import org.ksmt.expr.KRemIntExpr
import org.ksmt.expr.KSubArithExpr
import org.ksmt.expr.KToIntRealExpr
import org.ksmt.expr.KToRealIntExpr
import org.ksmt.expr.KUnaryMinusArithExpr
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.KXorExpr
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort

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
        transformExprAfterTransformedDefault(expr, expr.args, ::transformApp, KContext::mkAnd)

    override fun transform(expr: KOrExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.args, ::transformApp, KContext::mkOr)

    override fun transform(expr: KNotExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.arg, ::transformApp, KContext::mkNot)

    override fun transform(expr: KImpliesExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.p, expr.q, ::transformApp, KContext::mkImplies)

    override fun transform(expr: KXorExpr): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.a, expr.b, ::transformApp, KContext::mkXor)

    override fun <T : KSort> transform(expr: KEqExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.lhs, expr.rhs, ::transformApp, KContext::mkEq)

    override fun <T : KSort> transform(expr: KDistinctExpr<T>): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.args, ::transformApp, KContext::mkDistinct)

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
    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>): KExpr<KArraySort<D, R>> =
        transformExprAfterTransformedDefault(
            expr, expr.array, expr.index, expr.value, ::transformApp, KContext::mkArrayStore
        )

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> =
        transformExprAfterTransformedDefault(expr, expr.array, expr.index, ::transformApp, KContext::mkArraySelect)

    override fun <D : KSort, R : KSort> transform(expr: KArrayConst<D, R>): KExpr<KArraySort<D, R>> =
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

    // quantified expressions
    override fun <D : KSort, R : KSort> transform(
        expr: KArrayLambda<D, R>
    ): KExpr<KArraySort<D, R>> = transformExprAfterTransformedDefault(expr, expr.body, ::transformExpr) { body ->
        mkArrayLambda(expr.indexVarDecl, body)
    }

    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.body, ::transformExpr) { body ->
            mkExistentialQuantifier(body, expr.bounds)
        }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.body, ::transformExpr) { body ->
            mkUniversalQuantifier(body, expr.bounds)
        }

    // utils
    private fun <T : KSort> transformExprDefault(expr: KExpr<T>): KExpr<T> = when (expr) {
        is KInterpretedValue<T> -> transformValue(expr)
        is KApp<T, *> -> transformApp(expr)
        else -> transformExpr(expr)
    }

    private inline fun <In : KExpr<T>, Out : KExpr<T>, T : KSort, A : KSort> transformExprAfterTransformedDefault(
        expr: In,
        dependencies: List<KExpr<A>>,
        ifNotTransformed: (In) -> KExpr<T>,
        transformer: KContext.(List<KExpr<A>>) -> Out
    ): KExpr<T> = transformExprAfterTransformed(expr, dependencies) { transformedDependencies ->
        if (transformedDependencies == dependencies)
            return ifNotTransformed(expr)

        val transformedExpr = ctx.transformer(transformedDependencies)

        return transformExprDefault(transformedExpr)
    }

    private inline fun <In : KExpr<T>, Out : KExpr<T>, T : KSort, A : KSort> transformExprAfterTransformedDefault(
        expr: In,
        dependency: KExpr<A>,
        ifNotTransformed: (In) -> KExpr<T>,
        transformer: KContext.(KExpr<A>) -> Out
    ): KExpr<T> = transformExprAfterTransformed(expr, dependency) { td ->
        if (td == dependency)
            return ifNotTransformed(expr)

        val transformedExpr = ctx.transformer(td)

        return transformExprDefault(transformedExpr)
    }

    private inline fun <In : KExpr<T>, Out : KExpr<T>, T : KSort, A0 : KSort, A1 : KSort>
    transformExprAfterTransformedDefault(
        expr: In,
        dependency0: KExpr<A0>,
        dependency1: KExpr<A1>,
        ifNotTransformed: (In) -> KExpr<T>,
        transformer: KContext.(KExpr<A0>, KExpr<A1>) -> Out
    ): KExpr<T> = transformExprAfterTransformed(expr, dependency0, dependency1) { td0, td1 ->
        if (td0 == dependency0 && td1 == dependency1)
            return ifNotTransformed(expr)

        val transformedExpr = ctx.transformer(td0, td1)

        return transformExprDefault(transformedExpr)
    }

    @Suppress("LongParameterList")
    private inline fun <In : KExpr<T>, Out : KExpr<T>, T : KSort, A0 : KSort, A1 : KSort, A2 : KSort>
    transformExprAfterTransformedDefault(
        expr: In,
        dependency0: KExpr<A0>,
        dependency1: KExpr<A1>,
        dependency2: KExpr<A2>,
        ifNotTransformed: (In) -> KExpr<T>,
        transformer: KContext.(KExpr<A0>, KExpr<A1>, KExpr<A2>) -> Out
    ): KExpr<T> = transformExprAfterTransformed(expr, dependency0, dependency1, dependency2) { td0, td1, td2 ->
        if (td0 == dependency0 && td1 == dependency1 && td2 == dependency2)
            return ifNotTransformed(expr)

        val transformedExpr = ctx.transformer(td0, td1, td2)

        return transformExprDefault(transformedExpr)
    }

    @Suppress("LongParameterList", "ComplexCondition")
    private inline fun <In : KExpr<T>, Out : KExpr<T>, T : KSort, A0 : KSort, A1 : KSort, A2 : KSort, A3 : KSort>
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
            if (td0 == dependency0 && td1 == dependency1 && td2 == dependency2 && td3 == dependency3)
                return ifNotTransformed(expr)

            val transformedExpr = ctx.transformer(td0, td1, td2, td3)

            return transformExprDefault(transformedExpr)
        }

}
