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
import io.ksmt.expr.KSuffixOfExpr
import io.ksmt.expr.KPrefixOfExpr
import io.ksmt.expr.KStringLtExpr
import io.ksmt.expr.KStringLeExpr
import io.ksmt.expr.KStringGtExpr
import io.ksmt.expr.KStringGeExpr
import io.ksmt.expr.KStringContainsExpr
import io.ksmt.expr.KSingletonSubstringExpr
import io.ksmt.expr.KSubstringExpr
import io.ksmt.expr.KIndexOfExpr
import io.ksmt.expr.KStringReplaceExpr
import io.ksmt.expr.KStringReplaceAllExpr
import io.ksmt.expr.KStringReplaceWithRegexExpr
import io.ksmt.expr.KStringReplaceAllWithRegexExpr
import io.ksmt.expr.KStringIsDigitExpr
import io.ksmt.expr.KStringToCodeExpr
import io.ksmt.expr.KStringFromCodeExpr
import io.ksmt.expr.KStringToIntExpr
import io.ksmt.expr.KStringFromIntExpr
import io.ksmt.expr.KRegexConcatExpr
import io.ksmt.expr.KRegexUnionExpr
import io.ksmt.expr.KRegexIntersectionExpr
import io.ksmt.expr.KRegexKleeneClosureExpr
import io.ksmt.expr.KRegexKleeneCrossExpr
import io.ksmt.expr.KRegexDifferenceExpr
import io.ksmt.expr.KRegexComplementExpr
import io.ksmt.expr.KRegexOptionExpr
import io.ksmt.expr.KRangeExpr
import io.ksmt.sort.KArithSort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KSort

/**
 * Apply specialized non-recursive visit for all KSMT expressions.
 * See [KNonRecursiveVisitorBase] for details.
 * */
abstract class KNonRecursiveVisitor<V : Any>(
    override val ctx: KContext
) : KNonRecursiveVisitorBase<V>() {

    /**
     * Provides a default value for leaf expressions
     * that don't have a special visit and are not accepted by the generic visitor [visitExpr].
     *
     * @see [visitExprAfterVisitedDefault]
     * */
    abstract fun <T : KSort> defaultValue(expr: KExpr<T>): V

    /**
     * Merge subexpressions visit results.
     *
     * @see [visitExprAfterVisitedDefault]
     * */
    abstract fun mergeResults(left: V, right: V): V

    override fun <T : KSort> visitExpr(expr: KExpr<T>): KExprVisitResult<V> = KExprVisitResult.EMPTY

    override fun <T : KSort, A : KSort> visitApp(expr: KApp<T, A>): KExprVisitResult<V> = visitExpr(expr)

    // function visitors
    override fun <T : KSort> visit(expr: KFunctionApp<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.args, ::visitApp)

    override fun <T : KSort> visit(expr: KConst<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, ::visitApp)

    override fun <T : KSort> visitValue(expr: KInterpretedValue<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, ::visitApp)

    // bool visitors
    override fun visit(expr: KAndExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.args, ::visitApp
        )

    override fun visit(expr: KAndBinaryExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.lhs, expr.rhs, ::visitApp
        )

    override fun visit(expr: KOrExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.args, ::visitApp
        )

    override fun visit(expr: KOrBinaryExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.lhs, expr.rhs, ::visitApp
        )

    override fun visit(expr: KNotExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg, ::visitApp)

    override fun visit(expr: KImpliesExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.p, expr.q, ::visitApp)

    override fun visit(expr: KXorExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.a, expr.b, ::visitApp)

    override fun <T : KSort> visit(expr: KEqExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.lhs, expr.rhs, ::visitApp
        )

    override fun <T : KSort> visit(expr: KDistinctExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.args, ::visitApp
        )

    override fun <T : KSort> visit(expr: KIteExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr,
            expr.condition,
            expr.trueBranch,
            expr.falseBranch,
            ::visitApp
        )

    // bit-vec expressions visitors
    override fun <T : KBvSort> visit(expr: KBvNotExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvReductionAndExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvReductionOrExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvAndExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvOrExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvXorExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvNAndExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvNorExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvXNorExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvNegationExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvAddExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvSubExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvMulExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvUnsignedDivExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvSignedDivExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvUnsignedRemExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvSignedRemExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvSignedModExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvUnsignedLessExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvSignedLessExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvUnsignedLessOrEqualExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.arg0, expr.arg1, ::visitApp
        )

    override fun <T : KBvSort> visit(expr: KBvSignedLessOrEqualExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.arg0, expr.arg1, ::visitApp
        )

    override fun <T : KBvSort> visit(expr: KBvUnsignedGreaterOrEqualExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.arg0, expr.arg1, ::visitApp
        )

    override fun <T : KBvSort> visit(expr: KBvSignedGreaterOrEqualExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.arg0, expr.arg1, ::visitApp
        )

    override fun <T : KBvSort> visit(expr: KBvUnsignedGreaterExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.arg0, expr.arg1, ::visitApp
        )

    override fun <T : KBvSort> visit(expr: KBvSignedGreaterExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.arg0, expr.arg1, ::visitApp
        )

    override fun visit(expr: KBvConcatExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun visit(expr: KBvExtractExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun visit(expr: KBvSignExtensionExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun visit(expr: KBvZeroExtensionExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun visit(expr: KBvRepeatExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvShiftLeftExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg, expr.shift, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvLogicalShiftRightExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.arg, expr.shift, ::visitApp
        )

    override fun <T : KBvSort> visit(expr: KBvArithShiftRightExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.arg, expr.shift, ::visitApp
        )

    override fun <T : KBvSort> visit(expr: KBvRotateLeftExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.arg, expr.rotation, ::visitApp
        )

    override fun <T : KBvSort> visit(expr: KBvRotateLeftIndexedExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvRotateRightExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.arg, expr.rotation, ::visitApp
        )

    override fun <T : KBvSort> visit(expr: KBvRotateRightIndexedExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun visit(expr: KBv2IntExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvAddNoOverflowExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvAddNoUnderflowExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.arg0, expr.arg1, ::visitApp
        )

    override fun <T : KBvSort> visit(expr: KBvSubNoOverflowExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.arg0, expr.arg1, ::visitApp
        )

    override fun <T : KBvSort> visit(expr: KBvSubNoUnderflowExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvDivNoOverflowExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.arg0, expr.arg1, ::visitApp
        )

    override fun <T : KBvSort> visit(expr: KBvNegNoOverflowExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvMulNoOverflowExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KBvSort> visit(expr: KBvMulNoUnderflowExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.arg0, expr.arg1, ::visitApp
        )

    // fp operations tranformation
    override fun <T : KFpSort> visit(expr: KFpAbsExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpNegationExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpAddExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.roundingMode, expr.arg0, expr.arg1, ::visitApp
        )

    override fun <T : KFpSort> visit(expr: KFpSubExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.roundingMode, expr.arg0, expr.arg1, ::visitApp
        )

    override fun <T : KFpSort> visit(expr: KFpMulExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.roundingMode, expr.arg0, expr.arg1, ::visitApp
        )

    override fun <T : KFpSort> visit(expr: KFpDivExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.roundingMode, expr.arg0, expr.arg1, ::visitApp
        )

    override fun <T : KFpSort> visit(expr: KFpFusedMulAddExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.roundingMode, expr.arg0, expr.arg1, expr.arg2, ::visitApp
        )

    override fun <T : KFpSort> visit(expr: KFpSqrtExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.roundingMode, expr.value, ::visitApp
        )

    override fun <T : KFpSort> visit(expr: KFpRemExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpRoundToIntegralExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.roundingMode, expr.value, ::visitApp
        )

    override fun <T : KFpSort> visit(expr: KFpMinExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpMaxExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpLessOrEqualExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpLessExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpGreaterOrEqualExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.arg0, expr.arg1, ::visitApp
        )

    override fun <T : KFpSort> visit(expr: KFpGreaterExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpEqualExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpIsNormalExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpIsSubnormalExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpIsZeroExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpIsInfiniteExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpIsNaNExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpIsNegativeExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpIsPositiveExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpToBvExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.roundingMode, expr.value, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpToRealExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpToIEEEBvExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    override fun <T : KFpSort> visit(expr: KFpFromBvExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.sign, expr.biasedExponent, expr.significand, ::visitApp
        )

    override fun <T : KFpSort> visit(expr: KFpToFpExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.roundingMode, expr.value, ::visitApp
        )

    override fun <T : KFpSort> visit(expr: KRealToFpExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.roundingMode, expr.value, ::visitApp
        )

    override fun <T : KFpSort> visit(expr: KBvToFpExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(
            expr, expr.roundingMode, expr.value, ::visitApp
        )

    // array visitors
    override fun <D : KSort, R : KSort> visit(
        expr: KArrayStore<D, R>
    ): KExprVisitResult<V> = visitExprAfterVisitedDefault(
        expr, expr.array, expr.index, expr.value, ::visitArrayStore
    )

    override fun <D0 : KSort, D1 : KSort, R : KSort> visit(
        expr: KArray2Store<D0, D1, R>
    ): KExprVisitResult<V> = visitExprAfterVisitedDefault(
        expr, expr.array, expr.index0, expr.index1, expr.value, ::visitArrayStore
    )

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(
        expr: KArray3Store<D0, D1, D2, R>
    ): KExprVisitResult<V> = visitExprAfterVisitedDefault(
        expr, expr.array, expr.index0, expr.index1, expr.index2, expr.value,
        ::visitArrayStore
    )

    override fun <R : KSort> visit(
        expr: KArrayNStore<R>
    ): KExprVisitResult<V> = visitExprAfterVisitedDefault(
        expr, expr.args, ::visitArrayStore
    )

    override fun <D : KSort, R : KSort> visit(
        expr: KArraySelect<D, R>
    ): KExprVisitResult<V> = visitExprAfterVisitedDefault(
        expr, expr.array, expr.index, ::visitArraySelect
    )

    override fun <D0 : KSort, D1 : KSort, R : KSort> visit(
        expr: KArray2Select<D0, D1, R>
    ): KExprVisitResult<V> = visitExprAfterVisitedDefault(
        expr, expr.array, expr.index0, expr.index1, ::visitArraySelect
    )

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(
        expr: KArray3Select<D0, D1, D2, R>
    ): KExprVisitResult<V> = visitExprAfterVisitedDefault(
        expr, expr.array, expr.index0, expr.index1, expr.index2, ::visitArraySelect
    )

    override fun <R : KSort> visit(
        expr: KArrayNSelect<R>
    ): KExprVisitResult<V> = visitExprAfterVisitedDefault(
        expr, expr.args, ::visitArraySelect
    )

    override fun <D : KSort, R : KSort> visit(
        expr: KArrayLambda<D, R>
    ): KExprVisitResult<V> = visitExprAfterVisitedDefault(
        expr, expr.body, ::visitArrayLambda
    )

    override fun <D0 : KSort, D1 : KSort, R : KSort> visit(
        expr: KArray2Lambda<D0, D1, R>
    ): KExprVisitResult<V> = visitExprAfterVisitedDefault(
        expr, expr.body, ::visitArrayLambda
    )

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(
        expr: KArray3Lambda<D0, D1, D2, R>
    ): KExprVisitResult<V> = visitExprAfterVisitedDefault(
        expr, expr.body, ::visitArrayLambda
    )

    override fun <R : KSort> visit(
        expr: KArrayNLambda<R>
    ): KExprVisitResult<V> = visitExprAfterVisitedDefault(
        expr, expr.body, ::visitArrayLambda
    )

    override fun <A : KArraySortBase<R>, R : KSort> visit(expr: KArrayConst<A, R>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.value, ::visitApp)

    // arith visitors
    override fun <T : KArithSort> visit(expr: KAddArithExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.args, ::visitApp)

    override fun <T : KArithSort> visit(expr: KMulArithExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.args, ::visitApp)

    override fun <T : KArithSort> visit(expr: KSubArithExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.args, ::visitApp)

    override fun <T : KArithSort> visit(expr: KUnaryMinusArithExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg, ::visitApp)

    override fun <T : KArithSort> visit(expr: KDivArithExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.lhs, expr.rhs, ::visitApp)

    override fun <T : KArithSort> visit(expr: KPowerArithExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.lhs, expr.rhs, ::visitApp)

    override fun <T : KArithSort> visit(expr: KLtArithExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.lhs, expr.rhs, ::visitApp)

    override fun <T : KArithSort> visit(expr: KLeArithExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.lhs, expr.rhs, ::visitApp)

    override fun <T : KArithSort> visit(expr: KGtArithExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.lhs, expr.rhs, ::visitApp)

    override fun <T : KArithSort> visit(expr: KGeArithExpr<T>): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.lhs, expr.rhs, ::visitApp)

    // integer visitors
    override fun visit(expr: KModIntExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.lhs, expr.rhs, ::visitApp)

    override fun visit(expr: KRemIntExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.lhs, expr.rhs, ::visitApp)

    override fun visit(expr: KToRealIntExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg, ::visitApp)

    // real visitors
    override fun visit(expr: KToIntRealExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg, ::visitApp)

    override fun visit(expr: KIsIntRealExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg, ::visitApp)

    // string visitors
    override fun visit(expr: KStringConcatExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun visit(expr: KStringLenExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg, ::visitApp)

    override fun visit(expr: KStringToRegexExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg, ::visitApp)

    override fun visit(expr: KStringInRegexExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun visit(expr: KSuffixOfExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun visit(expr: KPrefixOfExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun visit(expr: KStringLtExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.lhs, expr.rhs, ::visitApp)

    override fun visit(expr: KStringLeExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.lhs, expr.rhs, ::visitApp)

    override fun visit(expr: KStringGtExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.lhs, expr.rhs, ::visitApp)

    override fun visit(expr: KStringGeExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.lhs, expr.rhs, ::visitApp)

    override fun visit(expr: KStringContainsExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.lhs, expr.rhs, ::visitApp)

    override fun visit(expr: KSingletonSubstringExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun visit(expr: KSubstringExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, expr.arg2, ::visitApp)

    override fun visit(expr: KIndexOfExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, expr.arg2, ::visitApp)

    override fun visit(expr: KStringReplaceExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, expr.arg2, ::visitApp)

    override fun visit(expr: KStringReplaceAllExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, expr.arg2, ::visitApp)

    override fun visit(expr: KStringReplaceWithRegexExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, expr.arg2, ::visitApp)

    override fun visit(expr: KStringReplaceAllWithRegexExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, expr.arg2, ::visitApp)

    override fun visit(expr: KStringIsDigitExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg, ::visitApp)

    override fun visit(expr: KStringToCodeExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg, ::visitApp)

    override fun visit(expr: KStringFromCodeExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg, ::visitApp)

    override fun visit(expr: KStringToIntExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg, ::visitApp)

    override fun visit(expr: KStringFromIntExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg, ::visitApp)

    // regex visitors
    override fun visit(expr: KRegexConcatExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun visit(expr: KRegexUnionExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun visit(expr: KRegexIntersectionExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun visit(expr: KRegexKleeneClosureExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg, ::visitApp)

    override fun visit(expr: KRegexKleeneCrossExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg, ::visitApp)

    override fun visit(expr: KRegexDifferenceExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    override fun visit(expr: KRegexComplementExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg, ::visitApp)

    override fun visit(expr: KRegexOptionExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg, ::visitApp)

    override fun visit(expr: KRangeExpr): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.arg0, expr.arg1, ::visitApp)

    // quantified expressions
    override fun visit(expr: KExistentialQuantifier): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.body, ::visitExpr)

    override fun visit(expr: KUniversalQuantifier): KExprVisitResult<V> =
        visitExprAfterVisitedDefault(expr, expr.body, ::visitExpr)

    /**
     * 1. Visit [dependencies] and merge their results.
     * 2. Visit [expr] with generic visitor [visitDefault].
     * Merge result with dependencies visit results.
     * 3. If merged result is empty return [defaultValue].
     * */
    inline fun <E : KExpr<*>> visitExprAfterVisitedDefault(
        expr: E,
        dependencies: List<KExpr<*>>,
        visitDefault: (E) -> KExprVisitResult<V>
    ): KExprVisitResult<V> = visitExprAfterVisited(expr, dependencies) { visitedDependencies ->
        val defaultRes = visitDefault(expr)
        if (defaultRes.dependencyVisitRequired) {
            return defaultRes
        }

        val dependenciesResults = visitedDependencies.reduceOrNull(::mergeResults)
        if (dependenciesResults == null) {
            if (defaultRes.hasResult) {
                defaultRes.result
            } else {
                defaultValue(expr)
            }
        } else {
            if (defaultRes.hasResult) {
                mergeResults(dependenciesResults, defaultRes.result)
            } else {
                dependenciesResults
            }
        }
    }

    /**
     * Specialized version of [visitExprAfterVisitedDefault] for expression with zero arguments.
     * */
    inline fun <E : KExpr<*>> visitExprAfterVisitedDefault(
        expr: E,
        visitDefault: (E) -> KExprVisitResult<V>
    ): KExprVisitResult<V> {
        val defaultRes = visitDefault(expr)
        if (defaultRes.isNotEmpty) {
            return defaultRes
        }

        return saveVisitResult(expr, defaultValue(expr))
    }

    /**
     * Specialized version of [visitExprAfterVisitedDefault] for expression with single argument.
     * */
    inline fun <E : KExpr<*>> visitExprAfterVisitedDefault(
        expr: E,
        dependency: KExpr<*>,
        visitDefault: (E) -> KExprVisitResult<V>
    ): KExprVisitResult<V> = visitExprAfterVisited(expr, dependency) { dr ->
        val defaultRes = visitDefault(expr)
        if (defaultRes.dependencyVisitRequired) {
            return defaultRes
        }
        if (defaultRes.hasResult) {
            mergeResults(dr, defaultRes.result)
        } else {
            dr
        }
    }

    /**
     * Specialized version of [visitExprAfterVisitedDefault] for expression with two arguments.
     * */
    inline fun <E : KExpr<*>> visitExprAfterVisitedDefault(
        expr: E,
        dependency0: KExpr<*>,
        dependency1: KExpr<*>,
        visitDefault: (E) -> KExprVisitResult<V>
    ): KExprVisitResult<V> = visitExprAfterVisited(expr, dependency0, dependency1) { dr0, dr1 ->
        val defaultRes = visitDefault(expr)
        if (defaultRes.dependencyVisitRequired) {
            return defaultRes
        }

        val dependencyResults = mergeResults(dr0, dr1)
        if (defaultRes.hasResult) {
            mergeResults(dependencyResults, defaultRes.result)
        } else {
            dependencyResults
        }
    }

    /**
     * Specialized version of [visitExprAfterVisitedDefault] for expression with three arguments.
     * */
    @Suppress("LongParameterList")
    inline fun <E : KExpr<*>> visitExprAfterVisitedDefault(
        expr: E,
        dependency0: KExpr<*>,
        dependency1: KExpr<*>,
        dependency2: KExpr<*>,
        visitDefault: (E) -> KExprVisitResult<V>
    ): KExprVisitResult<V> = visitExprAfterVisited(expr, dependency0, dependency1, dependency2) { dr0, dr1, dr2 ->
        val defaultRes = visitDefault(expr)
        if (defaultRes.dependencyVisitRequired) {
            return defaultRes
        }

        val dependencyResults = mergeResults(mergeResults(dr0, dr1), dr2)
        if (defaultRes.hasResult) {
            mergeResults(dependencyResults, defaultRes.result)
        } else {
            dependencyResults
        }
    }

    /**
     * Specialized version of [visitExprAfterVisitedDefault] for expression with four arguments.
     * */
    @Suppress("LongParameterList", "ComplexCondition")
    inline fun <E : KExpr<*>> visitExprAfterVisitedDefault(
        expr: E,
        dependency0: KExpr<*>,
        dependency1: KExpr<*>,
        dependency2: KExpr<*>,
        dependency3: KExpr<*>,
        visitDefault: (E) -> KExprVisitResult<V>
    ): KExprVisitResult<V> =
        visitExprAfterVisited(expr, dependency0, dependency1, dependency2, dependency3) { dr0, dr1, dr2, dr3 ->
            val defaultRes = visitDefault(expr)
            if (defaultRes.dependencyVisitRequired) {
                return defaultRes
            }

            val dependencyResults = mergeResults(mergeResults(dr0, dr1), mergeResults(dr2, dr3))
            if (defaultRes.hasResult) {
                mergeResults(dependencyResults, defaultRes.result)
            } else {
                dependencyResults
            }
        }

    /**
     * Specialized version of [visitExprAfterVisitedDefault] for expression with five arguments.
     * */
    @Suppress("LongParameterList", "ComplexCondition")
    inline fun <E : KExpr<*>> visitExprAfterVisitedDefault(
        expr: E,
        d0: KExpr<*>,
        d1: KExpr<*>,
        d2: KExpr<*>,
        d3: KExpr<*>,
        d4: KExpr<*>,
        visitDefault: (E) -> KExprVisitResult<V>
    ): KExprVisitResult<V> = visitExprAfterVisited(
        expr, d0, d1, d2, d3, d4
    ) { dr0, dr1, dr2, dr3, dr4 ->
        val defaultRes = visitDefault(expr)
        if (defaultRes.dependencyVisitRequired) {
            return defaultRes
        }

        val dependencyResults = mergeResults(
            mergeResults(mergeResults(dr0, dr1), mergeResults(dr2, dr3)),
            dr4
        )
        if (defaultRes.hasResult) {
            mergeResults(dependencyResults, defaultRes.result)
        } else {
            dependencyResults
        }
    }
}
