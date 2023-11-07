package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KApp
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
import io.ksmt.expr.KBvUnsignedDivExpr
import io.ksmt.expr.KBvUnsignedGreaterExpr
import io.ksmt.expr.KBvUnsignedGreaterOrEqualExpr
import io.ksmt.expr.KBvUnsignedLessExpr
import io.ksmt.expr.KBvUnsignedLessOrEqualExpr
import io.ksmt.expr.KBvUnsignedRemExpr
import io.ksmt.expr.KBvXNorExpr
import io.ksmt.expr.KBvXorExpr
import io.ksmt.expr.KBvZeroExtensionExpr
import io.ksmt.expr.KEqExpr
import io.ksmt.expr.KExpr
import io.ksmt.expr.KIteExpr
import io.ksmt.expr.KNotExpr
import io.ksmt.expr.printer.ExpressionPrinter
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KIntSort
import io.ksmt.utils.uncheckedCast

@Suppress(
    "LargeClass",
    "LongMethod",
    "ComplexMethod"
)
interface KBvExprSimplifier : KExprSimplifierBase {

    fun <T : KBvSort> simplifyEqBv(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> = with(ctx) {
        simplifyEqBvLight(lhs, rhs) { lhs2, rhs2 ->
            simplifyEqBvConcat(
                lhs = lhs2,
                rhs = rhs2,
                rewriteBvExtractExpr = { high, low, value -> rewrite(KBvExtractExpr(ctx, high, low, value)) },
                rewriteBvEq = { l, r -> rewrite(KEqExpr(ctx, l, r)) },
                rewriteFlatAnd = { args -> rewrite(mkAndAuxExpr(args)) }
            ) { lhs3, rhs3 ->
                withExpressionsOrdered(lhs3, rhs3, ::mkEqNoSimplify)
            }
        }
    }

    fun <T : KBvSort> areDefinitelyDistinctBv(lhs: KExpr<T>, rhs: KExpr<T>): Boolean {
        if (lhs is KBitVecValue<T> && rhs is KBitVecValue<T>) {
            return lhs != rhs
        }
        return false
    }

    fun <T : KBvSort> KContext.preprocess(expr: KBvUnsignedLessOrEqualExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KBvSort> KContext.postRewriteBvUnsignedLessOrEqualExpr(
        lhs: KExpr<T>,
        rhs: KExpr<T>
    ): KExpr<KBoolSort> = simplifyBvUnsignedLessOrEqualExpr(lhs, rhs)

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvUnsignedLessOrEqualExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvSignedLessOrEqualExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KBvSort> KContext.postRewriteBvSignedLessOrEqualExpr(
        lhs: KExpr<T>,
        rhs: KExpr<T>
    ): KExpr<KBoolSort> = simplifyBvSignedLessOrEqualExpr(lhs, rhs)

    override fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvSignedLessOrEqualExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvUnsignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        rewriteBvUnsignedGreaterOrEqualExpr(
            lhs = expr.arg0,
            rhs = expr.arg1,
            rewriteBvUnsignedLessOrEqualExpr = { l, r -> KBvUnsignedLessOrEqualExpr(this, l, r) }
        )

    fun <T : KBvSort> KContext.postRewriteBvUnsignedGreaterOrEqualExpr(
        lhs: KExpr<T>,
        rhs: KExpr<T>
    ): KExpr<KBoolSort> = error("Always preprocessed")

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvUnsignedGreaterOrEqualExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvUnsignedLessExpr<T>): KExpr<KBoolSort> =
        rewriteBvUnsignedLessExpr(
            lhs = expr.arg0,
            rhs = expr.arg1,
            rewriteBvUnsignedLessOrEqualExpr = { l, r -> KBvUnsignedLessOrEqualExpr(this, l, r) },
            rewriteNot = { KNotExpr(this, it) }
        )

    fun <T : KBvSort> KContext.postRewriteBvUnsignedLessExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        error("Always preprocessed")

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvUnsignedLessExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvUnsignedGreaterExpr<T>): KExpr<KBoolSort> =
        rewriteBvUnsignedGreaterExpr(
            lhs = expr.arg0,
            rhs = expr.arg1,
            rewriteBvUnsignedLessOrEqualExpr = { l, r -> KBvUnsignedLessOrEqualExpr(this, l, r) },
            rewriteNot = { KNotExpr(this, it) }
        )

    fun <T : KBvSort> KContext.postRewriteBvUnsignedGreaterExpr(
        lhs: KExpr<T>,
        rhs: KExpr<T>
    ): KExpr<KBoolSort> = error("Always preprocessed")

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvUnsignedGreaterExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvSignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        rewriteBvSignedGreaterOrEqualExpr(
            lhs = expr.arg0,
            rhs = expr.arg1,
            rewriteBvSignedLessOrEqualExpr = { l, r -> KBvSignedLessOrEqualExpr(this, l, r) }
        )

    fun <T : KBvSort> KContext.postRewriteBvSignedGreaterOrEqualExpr(
        lhs: KExpr<T>,
        rhs: KExpr<T>
    ): KExpr<KBoolSort> = error("Always preprocessed")

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvSignedGreaterOrEqualExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvSignedLessExpr<T>): KExpr<KBoolSort> =
        rewriteBvSignedLessExpr(
            lhs = expr.arg0,
            rhs = expr.arg1,
            rewriteBvSignedLessOrEqualExpr = { l, r -> KBvSignedLessOrEqualExpr(this, l, r) },
            rewriteNot = { KNotExpr(this, it) }
        )

    fun <T : KBvSort> KContext.postRewriteBvSignedLessExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        error("Always preprocessed")

    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvSignedLessExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvSignedGreaterExpr<T>): KExpr<KBoolSort> =
        rewriteBvSignedGreaterExpr(
            lhs = expr.arg0,
            rhs = expr.arg1,
            rewriteBvSignedLessOrEqualExpr = { l, r -> KBvSignedLessOrEqualExpr(this, l, r) },
            rewriteNot = { KNotExpr(this, it) }
        )

    fun <T : KBvSort> KContext.postRewriteBvSignedGreaterExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        error("Always preprocessed")

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvSignedGreaterExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvAddExpr<T>): KExpr<T> = flatBvAdd(expr)

    fun <T : KBvSort> KContext.postRewriteBvAddExpr(args: List<KExpr<T>>): KExpr<T> =
        simplifyFlatBvAddExpr(args) { it.reduceBinaryBvExpr(::mkBvAddExprNoSimplify) }

    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            args = expr.args,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteBvAddExpr(it) }
        )

    private fun <T : KBvSort> transform(expr: SimplifierFlatBvAddExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.args) { flatten ->
            postRewriteBvAddExpr(flatten)
        }

    fun <T : KBvSort> KContext.preprocess(expr: KBvSubExpr<T>): KExpr<T> =
        rewriteBvSubExpr(
            lhs = expr.arg0,
            rhs = expr.arg1,
            rewriteBvAddExpr = { l, r -> KBvAddExpr(this, l, r) },
            rewriteBvNegationExpr = { KBvNegationExpr(this, it) }
        )

    fun <T : KBvSort> KContext.postRewriteBvSubExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
        error("Always preprocessed")

    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvSubExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvMulExpr<T>): KExpr<T> =
        flatBvMul(expr)

    fun <T : KBvSort> KContext.postRewriteBvMulExpr(args: List<KExpr<T>>): KExpr<T> =
        simplifyFlatBvMulExpr(args) { negateResult, resultParts ->
            val value = resultParts.reduceBinaryBvExpr(::mkBvMulExprNoSimplify)

            if (negateResult) {
                mkBvNegationExprNoSimplify(value)
            } else {
                value
            }
        }

    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            args = expr.args,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteBvMulExpr(it) }
        )

    private fun <T : KBvSort> transform(expr: SimplifierFlatBvMulExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.args) { flatten ->
            postRewriteBvMulExpr(flatten)
        }

    fun <T : KBvSort> KContext.preprocess(expr: KBvNegationExpr<T>): KExpr<T> =
        expr

    fun <T : KBvSort> KContext.postRewriteBvNegationExpr(arg: KExpr<T>): KExpr<T> =
        simplifyBvNegationExpr(arg)

    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteBvNegationExpr(it) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvSignedDivExpr<T>): KExpr<T> = expr

    fun <T : KBvSort> KContext.postRewriteBvSignedDivExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
        simplifyBvSignedDivExpr(lhs, rhs)

    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvSignedDivExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvUnsignedDivExpr<T>): KExpr<T> = expr

    fun <T : KBvSort> KContext.postRewriteBvUnsignedDivExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
        simplifyBvUnsignedDivExpr(lhs, rhs)

    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvUnsignedDivExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvSignedRemExpr<T>): KExpr<T> = expr

    fun <T : KBvSort> KContext.postRewriteBvSignedRemExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
        simplifyBvSignedRemExpr(lhs, rhs)

    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvSignedRemExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvUnsignedRemExpr<T>): KExpr<T> = expr

    fun <T : KBvSort> KContext.postRewriteBvUnsignedRemExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
        simplifyBvUnsignedRemExpr(lhs, rhs)

    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvUnsignedRemExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvSignedModExpr<T>): KExpr<T> = expr

    fun <T : KBvSort> KContext.postRewriteBvSignedModExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
        simplifyBvSignedModExpr(lhs, rhs)

    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvSignedModExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvNotExpr<T>): KExpr<T> = expr

    fun <T : KBvSort> KContext.postRewriteBvNotExpr(arg: KExpr<T>): KExpr<T> =
        simplifyBvNotExprLight(arg) { arg2 ->
            if (canPerformBoundedRewrite()) {
                simplifyBvNotExprConcat(
                    arg = arg2,
                    rewriteBvNotExpr = { boundedRewrite(KBvNotExpr(ctx, it)) },
                    rewriteBvConcatExpr = { args -> boundedRewrite(SimplifierFlatBvConcatExpr(ctx, arg2.sort, args)) }
                ) { arg3 ->
                    simplifyBvNotExprIte(
                        arg = arg3,
                        rewriteBvNotExpr = { v -> boundedRewrite(KBvNotExpr(ctx, v)) },
                        rewriteBvIte = { c, t, f -> boundedRewrite(KIteExpr(ctx, c, t, f)) },
                        cont = ::mkBvNotExprNoSimplify
                    )
                }
            } else {
                mkBvNotExprNoSimplify(arg2)
            }
        }

    override fun <T : KBvSort> transform(expr: KBvNotExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteBvNotExpr(it) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvOrExpr<T>): KExpr<T> = flatBvOr(expr)

    fun <T : KBvSort> KContext.postRewriteBvOrExpr(args: List<KExpr<T>>): KExpr<T> =
        simplifyFlatBvOrExpr(args) { resultParts ->
            if (canPerformBoundedRewrite()) {
                simplifyFlatBvOrExprDistributeOverConcat(
                    args = resultParts,
                    rewriteBvExtractExpr = { h, l, v -> boundedRewrite(KBvExtractExpr(ctx, h, l, v)) },
                    rewriteFlatBvOrExpr = { boundedRewrite(it.reduceBinaryBvExpr { a, b -> KBvOrExpr(ctx, a, b) }) },
                    rewriteBvConcatExpr = { l, r -> boundedRewrite(KBvConcatExpr(ctx, l, r)) },
                    cont = { it.reduceBinaryBvExpr(::mkBvOrExprNoSimplify) }
                )
            } else {
                resultParts.reduceBinaryBvExpr(::mkBvOrExprNoSimplify)
            }
        }

    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            args = expr.args,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteBvOrExpr(it) }
        )

    private fun <T : KBvSort> transform(expr: SimplifierFlatBvOrExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.args) { flatten ->
            postRewriteBvOrExpr(flatten)
        }

    fun <T : KBvSort> KContext.preprocess(expr: KBvXorExpr<T>): KExpr<T> = flatBvXor(expr)

    fun <T : KBvSort> KContext.postRewriteBvXorExpr(args: List<KExpr<T>>): KExpr<T> =
        simplifyFlatBvXorExpr(args) { negateResult, resultParts ->
            if (canPerformBoundedRewrite()) {
                simplifyFlatBvXorExprDistributeOverConcat(
                    args = resultParts,
                    rewriteBvExtractExpr = { h, l, v -> boundedRewrite(KBvExtractExpr(ctx, h, l, v)) },
                    rewriteFlatBvXorExpr = { boundedRewrite(it.reduceBinaryBvExpr { a, b -> KBvXorExpr(ctx, a, b) }) },
                    rewriteBvConcatExpr = { l, r ->
                        val preResult = KBvConcatExpr(ctx, l, r)
                        boundedRewrite(if (negateResult) KBvNotExpr(ctx, preResult) else preResult)
                    },
                    cont = {
                        val preResult = it.reduceBinaryBvExpr(::mkBvXorExprNoSimplify)
                        if (negateResult) mkBvNotExprNoSimplify(preResult) else preResult
                    }
                )
            } else {
                val preResult = resultParts.reduceBinaryBvExpr(::mkBvXorExprNoSimplify)
                return if (negateResult) mkBvNotExprNoSimplify(preResult) else preResult
            }
        }

    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            args = expr.args,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteBvXorExpr(it) }
        )

    private fun <T : KBvSort> transform(expr: SimplifierFlatBvXorExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.args) { flatten ->
            postRewriteBvXorExpr(flatten)
        }

    fun <T : KBvSort> KContext.preprocess(expr: KBvAndExpr<T>): KExpr<T> = flatBvAnd(expr)

    fun <T : KBvSort> KContext.postRewriteBvAndExpr(args: List<KExpr<T>>): KExpr<T> =
        simplifyFlatBvAndExpr(args) { resultParts ->
            if (canPerformBoundedRewrite()) {
                simplifyFlatBvAndExprDistributeOverConcat(
                    args = resultParts,
                    rewriteBvExtractExpr = { h, l, v -> boundedRewrite(KBvExtractExpr(ctx, h, l, v)) },
                    rewriteFlatBvAndExpr = { boundedRewrite(it.reduceBinaryBvExpr { a, b -> KBvAndExpr(ctx, a, b) }) },
                    rewriteBvConcatExpr = { l, r -> boundedRewrite(KBvConcatExpr(ctx, l, r)) },
                    cont = { it.reduceBinaryBvExpr(::mkBvAndExprNoSimplify) }
                )
            } else {
                resultParts.reduceBinaryBvExpr(::mkBvAndExprNoSimplify)
            }
        }

    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            args = expr.args,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteBvAndExpr(it) }
        )

    private fun <T : KBvSort> transform(expr: SimplifierFlatBvAndExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.args) { flatten ->
            postRewriteBvAndExpr(flatten)
        }

    fun <T : KBvSort> KContext.preprocess(expr: KBvNAndExpr<T>): KExpr<T> =
        rewriteBvNAndExpr(
            lhs = expr.arg0,
            rhs = expr.arg1,
            rewriteBvOrExpr = { l, r -> KBvOrExpr(this, l, r) },
            rewriteBvNotExpr = { KBvNotExpr(this, it) }
        )

    fun <T : KBvSort> KContext.postRewriteBvNAndExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
        error("Always preprocessed")

    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvNAndExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvNorExpr<T>): KExpr<T> =
        rewriteBvNorExpr(
            lhs = expr.arg0,
            rhs = expr.arg1,
            rewriteBvOrExpr = { l, r -> KBvOrExpr(this, l, r) },
            rewriteBvNotExpr = { KBvNotExpr(this, it) }
        )

    fun <T : KBvSort> KContext.postRewriteBvNorExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
        error("Always preprocessed")

    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvNorExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvXNorExpr<T>): KExpr<T> =
        rewriteBvXNorExpr(
            lhs = expr.arg0,
            rhs = expr.arg1,
            rewriteBvXorExpr = { l, r -> KBvXorExpr(this, l, r) },
            rewriteBvNotExpr = { KBvNotExpr(this, it) }
        )

    fun <T : KBvSort> KContext.postRewriteBvXNorExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
        error("Always preprocessed")

    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvXNorExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvReductionAndExpr<T>): KExpr<KBv1Sort> = expr

    fun <T : KBvSort> KContext.postRewriteBvReductionAndExpr(arg: KExpr<T>): KExpr<KBv1Sort> =
        simplifyBvReductionAndExpr(arg)

    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>): KExpr<KBv1Sort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteBvReductionAndExpr(it) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvReductionOrExpr<T>): KExpr<KBv1Sort> = expr

    fun <T : KBvSort> KContext.postRewriteBvReductionOrExpr(arg: KExpr<T>): KExpr<KBv1Sort> =
        simplifyBvReductionOrExpr(arg)

    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>): KExpr<KBv1Sort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteBvReductionOrExpr(it) }
        )

    fun KContext.preprocess(expr: KBvConcatExpr): KExpr<KBvSort> = flatConcat(expr)

    fun KContext.postRewriteBvConcatExpr(args: List<KExpr<KBvSort>>): KExpr<KBvSort> =
        simplifyFlatBvConcatExpr(
            args = args,
            rewriteBvExtractExpr = { h, l, v -> rewrite(KBvExtractExpr(ctx, h, l, v)) },
            rewriteFlatBvConcatExpr = { parts ->
                rewrite(SimplifierFlatBvConcatExpr(ctx, mkBvSort(parts.sumOf { it.sort.sizeBits }), parts))
            },
            cont = { it.reduceBinaryBvExpr(::mkBvConcatExprNoSimplify) }
        )

    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> =
        simplifyExpr(
            expr = expr,
            args = expr.args,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteBvConcatExpr(it) }
        )

    @Suppress("LoopWithTooManyJumpStatements")
    private fun transform(expr: SimplifierFlatBvConcatExpr): KExpr<KBvSort> =
        simplifyExpr(expr, expr.args) { flatten ->
            postRewriteBvConcatExpr(flatten)
        }

    fun KContext.preprocess(expr: KBvExtractExpr): KExpr<KBvSort> = expr

    fun KContext.postRewriteBvExtractExpr(high: Int, low: Int, value: KExpr<KBvSort>): KExpr<KBvSort> =
        simplifyBvExtractExprLight(high, low, value) { high2, low2, value2 ->
            simplifyBvExtractExprNestedExtract(
                high = high2,
                low = low2,
                value = value2,
                rewriteBvExtractExpr = { h, l, v -> rewrite(KBvExtractExpr(ctx, h, l, v)) }
            ) { high3, low3, value3 ->
                if (canPerformBoundedRewrite()) {
                    simplifyBvExtractExprTryRewrite(
                        high = high3,
                        low = low3,
                        value = value3,
                        rewriteBvExtractExpr = { h, l, v -> boundedRewrite(KBvExtractExpr(ctx, h, l, v)) },
                        rewriteFlatBvConcatExpr = { args ->
                            boundedRewrite(SimplifierFlatBvConcatExpr(ctx, value3.sort, args))
                        },
                        rewriteBvNotExpr = { arg -> boundedRewrite(KBvNotExpr(ctx, arg)) },
                        rewriteBvOrExpr = { l, r -> boundedRewrite(KBvOrExpr(ctx, l, r)) },
                        rewriteBvAndExpr = { l, r -> boundedRewrite(KBvAndExpr(ctx, l, r)) },
                        rewriteBvXorExpr = { l, r -> boundedRewrite(KBvXorExpr(ctx, l, r)) },
                        rewriteBvAddExpr = { l, r -> boundedRewrite(KBvAddExpr(ctx, l, r)) },
                        rewriteBvMulExpr = { l, r -> boundedRewrite(KBvMulExpr(ctx, l, r)) },
                        cont = ::mkBvExtractExprNoSimplify
                    )
                } else {
                    mkBvExtractExprNoSimplify(high3, low3, value3)
                }
            }
        }

    override fun transform(expr: KBvExtractExpr): KExpr<KBvSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteBvExtractExpr(expr.high, expr.low, it) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvShiftLeftExpr<T>): KExpr<T> = expr

    fun <T : KBvSort> KContext.postRewriteBvShiftLeftExpr(arg: KExpr<T>, shift: KExpr<T>): KExpr<T> =
        simplifyBvShiftLeftExprLight(arg, shift) { arg2, shift2 ->
            simplifyBvShiftLeftExprConstShift(
                lhs = arg2,
                shift = shift2,
                rewriteBvExtractExpr = { h, l, v -> rewrite(KBvExtractExpr(ctx, h, l, v)) },
                rewriteBvConcatExpr = { l, r -> rewrite(KBvConcatExpr(ctx, l, r)) }
            ) { arg3, shift3 ->
                simplifyBvShiftLeftExprNestedShiftLeft(
                    lhs = arg3,
                    shift = shift3,
                    rewriteBvAddExpr = { l, r -> rewrite(KBvAddExpr(ctx, l, r)) },
                    rewriteBvUnsignedLessOrEqualExpr = { l, r -> rewrite(KBvUnsignedLessOrEqualExpr(ctx, l, r)) },
                    rewriteBvShiftLeftExpr = { a, shift -> rewrite(KBvShiftLeftExpr(ctx, a, shift)) },
                    rewriteIte = { c, t, f ->  rewrite(KIteExpr(ctx, c, t, f)) },
                    cont = ::mkBvShiftLeftExprNoSimplify
                )
            }
        }

    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg,
            a1 = expr.shift,
            preprocess = { preprocess(it) },
            simplifier = { arg, shift -> postRewriteBvShiftLeftExpr(arg, shift) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvLogicalShiftRightExpr<T>): KExpr<T> = expr

    fun <T : KBvSort> KContext.postRewriteBvLogicalShiftRightExpr(arg: KExpr<T>, shift: KExpr<T>): KExpr<T> =
        simplifyBvLogicalShiftRightExprLight(arg, shift) { arg2, shift2 ->
            simplifyBvLogicalShiftRightExprConstShift(
                arg = arg2,
                shift = shift2,
                rewriteBvExtractExpr = { high, low, arg -> rewrite(KBvExtractExpr(ctx, high, low, arg)) },
                rewriteBvConcatExpr = { l, r -> rewrite(KBvConcatExpr(ctx, l, r)) },
                cont = ::mkBvLogicalShiftRightExprNoSimplify
            )
        }

    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg,
            a1 = expr.shift,
            preprocess = { preprocess(it) },
            simplifier = { arg, shift -> postRewriteBvLogicalShiftRightExpr(arg, shift) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvArithShiftRightExpr<T>): KExpr<T> = expr

    fun <T : KBvSort> KContext.postRewriteBvArithShiftRightExpr(arg: KExpr<T>, shift: KExpr<T>): KExpr<T> =
        simplifyBvArithShiftRightExpr(arg, shift)

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg,
            a1 = expr.shift,
            preprocess = { preprocess(it) },
            simplifier = { arg, shift -> postRewriteBvArithShiftRightExpr(arg, shift) }
        )

    fun KContext.preprocess(expr: KBvRepeatExpr): KExpr<KBvSort> = expr

    fun <T : KBvSort> KContext.postRewriteBvRepeatExpr(repeatNumber: Int, arg: KExpr<T>): KExpr<KBvSort> =
        simplifyBvRepeatExprLight(repeatNumber, arg) { repeatNumber2, arg2 ->
            rewriteBvRepeatExpr(repeatNumber2, arg2.uncheckedCast()) { args ->
                rewrite(
                    SimplifierFlatBvConcatExpr(
                        ctx = ctx,
                        sort = mkBvSort(arg.sort.sizeBits * repeatNumber.toUInt()),
                        args = args
                    )
                )
            }
        }

    override fun transform(expr: KBvRepeatExpr): KExpr<KBvSort> =
        simplifyExpr(
            expr,
            expr.value,
            preprocess = { preprocess(it) },
            simplifier = { v -> postRewriteBvRepeatExpr(expr.repeatNumber, v) }
        )

    fun KContext.preprocess(expr: KBvZeroExtensionExpr): KExpr<KBvSort> = expr

    fun <T : KBvSort> KContext.postRewriteBvZeroExtensionExpr(extensionSize: Int, value: KExpr<T>): KExpr<KBvSort> =
        simplifyBvZeroExtensionExprLight(extensionSize, value) { extensionSize2, arg2 ->
            rewriteBvZeroExtensionExpr(extensionSize2, arg2) { l, r -> rewrite(KBvConcatExpr(ctx, l, r)) }
        }

    override fun transform(expr: KBvZeroExtensionExpr): KExpr<KBvSort> =
        simplifyExpr(
            expr,
            expr.value,
            preprocess = { preprocess(it) },
            simplifier = { v -> postRewriteBvZeroExtensionExpr(expr.extensionSize, v) }
        )

    fun KContext.preprocess(expr: KBvSignExtensionExpr): KExpr<KBvSort> = expr

    fun  <T : KBvSort> KContext.postRewriteBvSignExtensionExpr(
        extensionSize: Int,
        value: KExpr<T>
    ): KExpr<KBvSort> = simplifyBvSignExtensionExprLight(extensionSize, value, ::mkBvSignExtensionExprNoSimplify)

    override fun transform(expr: KBvSignExtensionExpr): KExpr<KBvSort> =
        simplifyExpr(
            expr,
            expr.value,
            preprocess = { preprocess(it) },
            simplifier = { v -> postRewriteBvSignExtensionExpr(expr.extensionSize, v) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvRotateLeftIndexedExpr<T>): KExpr<T> = expr

    // (rotateLeft a x) ==> (concat (extract [size-1-x:0] a) (extract [size-1:size-x] a))
    fun <T : KBvSort> KContext.postRewriteBvRotateLeftIndexedExpr(rotation: Int, value: KExpr<T>): KExpr<T> =
        simplifyBvRotateLeftIndexedExpr(rotation, value)

    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>): KExpr<T> =
        simplifyExpr(
            expr,
            expr.value,
            preprocess = { preprocess(it) },
            simplifier = { v -> postRewriteBvRotateLeftIndexedExpr(expr.rotationNumber, v) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvRotateLeftExpr<T>): KExpr<T> = expr

    fun <T : KBvSort> KContext.postRewriteBvRotateLeftExpr(value: KExpr<T>, rotation: KExpr<T>): KExpr<T> =
        simplifyBvRotateLeftExpr(value, rotation)

    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg,
            a1 = expr.rotation,
            preprocess = { preprocess(it) },
            simplifier = { v, rotation -> postRewriteBvRotateLeftExpr(v, rotation) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvRotateRightIndexedExpr<T>): KExpr<T> = expr

    // (rotateRight a x) ==> (rotateLeft a (- size x))
    fun <T : KBvSort> KContext.postRewriteBvRotateRightIndexedExpr(rotation: Int, value: KExpr<T>): KExpr<T> =
        simplifyBvRotateRightIndexedExpr(rotation, value)

    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteBvRotateRightIndexedExpr(expr.rotationNumber, it) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvRotateRightExpr<T>): KExpr<T> = expr

    fun <T : KBvSort> KContext.postRewriteBvRotateRightExpr(value: KExpr<T>, rotation: KExpr<T>): KExpr<T> =
        simplifyBvRotateRightExpr(value, rotation)

    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg,
            a1 = expr.rotation,
            preprocess = { preprocess(it) },
            simplifier = { v, rotation -> postRewriteBvRotateRightExpr(v, rotation) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvAddNoOverflowExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KBvSort> KContext.postRewriteBvAddNoOverflowExpr(
        lhs: KExpr<T>,
        rhs: KExpr<T>,
        isSigned: Boolean
    ): KExpr<KBoolSort> = rewriteBvAddNoOverflowExpr(lhs, rhs, isSigned)

    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvAddNoOverflowExpr(l, r, expr.isSigned) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvAddNoUnderflowExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KBvSort> KContext.postRewriteBvAddNoUnderflowExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        rewriteBvAddNoUnderflowExpr(lhs, rhs)

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvAddNoUnderflowExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvSubNoOverflowExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KBvSort> KContext.postRewriteBvSubNoOverflowExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        rewriteBvSubNoOverflowExpr(lhs, rhs)

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvSubNoOverflowExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KBvSort> KContext.postRewriteBvSubNoUnderflowExpr(
        lhs: KExpr<T>,
        rhs: KExpr<T>,
        isSigned: Boolean
    ): KExpr<KBoolSort> = rewriteBvSubNoUnderflowExpr(lhs, rhs, isSigned)

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvSubNoUnderflowExpr(l, r, expr.isSigned) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvNegNoOverflowExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KBvSort> KContext.postRewriteBvNegNoOverflowExpr(arg: KExpr<T>): KExpr<KBoolSort> =
        rewriteBvNegNoOverflowExpr(arg)

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteBvNegNoOverflowExpr(it) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvDivNoOverflowExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KBvSort> KContext.postRewriteBvDivNoOverflowExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        rewriteBvDivNoOverflowExpr(lhs, rhs)

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvDivNoOverflowExpr(l, r) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvMulNoOverflowExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KBvSort> KContext.postRewriteBvMulNoOverflowExpr(
        lhs: KExpr<T>,
        rhs: KExpr<T>,
        isSigned: Boolean
    ): KExpr<KBoolSort> = rewriteBvMulNoOverflowExpr(lhs, rhs, isSigned)

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvMulNoOverflowExpr(l, r, expr.isSigned) }
        )

    fun <T : KBvSort> KContext.preprocess(expr: KBvMulNoUnderflowExpr<T>): KExpr<KBoolSort> = expr

    fun <T : KBvSort> KContext.postRewriteBvMulNoUnderflowExpr(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        rewriteBvMulNoUnderflowExpr(lhs, rhs)

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg0,
            a1 = expr.arg1,
            preprocess = { preprocess(it) },
            simplifier = { l, r -> postRewriteBvMulNoUnderflowExpr(l, r) }
        )

    fun KContext.preprocess(expr: KBv2IntExpr): KExpr<KIntSort> = expr

    fun <T : KBvSort> KContext.postRewriteBv2IntExpr(value: KExpr<T>, isSigned: Boolean): KExpr<KIntSort> =
        simplifyBv2IntExpr(value, isSigned)

    override fun transform(expr: KBv2IntExpr): KExpr<KIntSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteBv2IntExpr(it, expr.isSigned) }
        )

    private fun flatConcat(expr: KBvConcatExpr): SimplifierFlatBvConcatExpr {
        val flatten = flatBinaryBvExpr<KBvConcatExpr>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 },
            getRhs = { it.arg1 }
        )
        return SimplifierFlatBvConcatExpr(ctx, expr.sort, flatten)
    }

    @Suppress("UNCHECKED_CAST")
    private fun <S : KBvSort> flatBvAdd(expr: KBvAddExpr<S>): SimplifierFlatBvAddExpr<S> {
        val flatten = flatBinaryBvExpr<KBvAddExpr<*>>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 as KExpr<KBvSort> },
            getRhs = { it.arg1 as KExpr<KBvSort> }
        )
        return SimplifierFlatBvAddExpr(ctx, flatten).uncheckedCast()
    }

    @Suppress("UNCHECKED_CAST")
    private fun <S : KBvSort> flatBvMul(expr: KBvMulExpr<S>): SimplifierFlatBvMulExpr<S> {
        val flatten = flatBinaryBvExpr<KBvMulExpr<*>>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 as KExpr<KBvSort> },
            getRhs = { it.arg1 as KExpr<KBvSort> }
        )
        return SimplifierFlatBvMulExpr(ctx, flatten).uncheckedCast()
    }

    @Suppress("UNCHECKED_CAST")
    private fun <S : KBvSort> flatBvOr(expr: KBvOrExpr<S>): SimplifierFlatBvOrExpr<S> {
        val flatten = flatBinaryBvExpr<KBvOrExpr<*>>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 as KExpr<KBvSort> },
            getRhs = { it.arg1 as KExpr<KBvSort> }
        )
        return SimplifierFlatBvOrExpr(ctx, flatten).uncheckedCast()
    }

    @Suppress("UNCHECKED_CAST")
    private fun <S : KBvSort> flatBvAnd(expr: KBvAndExpr<S>): SimplifierFlatBvAndExpr<S> {
        val flatten = flatBinaryBvExpr<KBvAndExpr<*>>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 as KExpr<KBvSort> },
            getRhs = { it.arg1 as KExpr<KBvSort> }
        )
        return SimplifierFlatBvAndExpr(ctx, flatten).uncheckedCast()
    }

    @Suppress("UNCHECKED_CAST")
    private fun <S : KBvSort> flatBvXor(expr: KBvXorExpr<S>): SimplifierFlatBvXorExpr<S> {
        val flatten = flatBinaryBvExpr<KBvXorExpr<*>>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 as KExpr<KBvSort> },
            getRhs = { it.arg1 as KExpr<KBvSort> }
        )
        return SimplifierFlatBvXorExpr(ctx, flatten).uncheckedCast()
    }

    /**
     * Reduce to an almost balanced expression tree.
     * */
    private inline fun <T : KBvSort> List<KExpr<T>>.reduceBinaryBvExpr(
        crossinline reducer: (KExpr<T>, KExpr<T>) -> KExpr<T>
    ): KExpr<T> {
        if (isEmpty()) {
            throw NoSuchElementException("List is empty")
        }
        val result = toMutableList()
        var size = size
        while (size > 1) {
            val unpairedLastElement = size % 2 == 1
            val realSize = if (unpairedLastElement) size - 1 else size
            var writeIdx = 0
            var readIdx = 0
            while (readIdx < realSize) {
                val arg0 = result[readIdx]
                val arg1 = result[readIdx + 1]
                result[writeIdx] = reducer(arg0, arg1)
                readIdx += 2
                writeIdx++
            }
            if (unpairedLastElement) {
                result[writeIdx] = result[size - 1]
                writeIdx++
            }
            size = writeIdx
        }
        return result[0]
    }

    /**
     * Auxiliary expression to store n-ary bv-add.
     * @see [SimplifierAuxExpression]
     * */
    private class SimplifierFlatBvAddExpr<T : KBvSort>(
        ctx: KContext,
        override val args: List<KExpr<T>>
    ) : KApp<T, T>(ctx), KSimplifierAuxExpr {

        override val decl: KDecl<T>
            get() = ctx.mkBvAddDecl(sort, sort)

        override val sort: T
            get() = args.first().sort

        override fun accept(transformer: KTransformerBase): KExpr<T> {
            transformer as KBvExprSimplifier
            return transformer.transform(this)
        }
    }

    /**
     * Auxiliary expression to store n-ary bv-mul.
     * @see [SimplifierAuxExpression]
     * */
    private class SimplifierFlatBvMulExpr<T : KBvSort>(
        ctx: KContext,
        override val args: List<KExpr<T>>
    ) : KApp<T, T>(ctx), KSimplifierAuxExpr {

        override val decl: KDecl<T>
            get() = ctx.mkBvMulDecl(sort, sort)

        override val sort: T
            get() = args.first().sort

        override fun accept(transformer: KTransformerBase): KExpr<T> {
            transformer as KBvExprSimplifier
            return transformer.transform(this)
        }
    }

    /**
     * Auxiliary expression to store n-ary bv-or.
     * @see [SimplifierAuxExpression]
     * */
    private class SimplifierFlatBvOrExpr<T : KBvSort>(
        ctx: KContext,
        override val args: List<KExpr<T>>
    ) : KApp<T, T>(ctx), KSimplifierAuxExpr {

        override val decl: KDecl<T>
            get() = ctx.mkBvOrDecl(sort, sort)

        override val sort: T
            get() = args.first().sort

        override fun accept(transformer: KTransformerBase): KExpr<T> {
            transformer as KBvExprSimplifier
            return transformer.transform(this)
        }
    }

    /**
     * Auxiliary expression to store n-ary bv-and.
     * @see [SimplifierAuxExpression]
     * */
    private class SimplifierFlatBvAndExpr<T : KBvSort>(
        ctx: KContext,
        override val args: List<KExpr<T>>
    ) : KApp<T, T>(ctx), KSimplifierAuxExpr {

        override val decl: KDecl<T>
            get() = ctx.mkBvAndDecl(sort, sort)

        override val sort: T
            get() = args.first().sort

        override fun accept(transformer: KTransformerBase): KExpr<T> {
            transformer as KBvExprSimplifier
            return transformer.transform(this)
        }
    }

    /**
     * Auxiliary expression to store n-ary bv-xor.
     * @see [SimplifierAuxExpression]
     * */
    private class SimplifierFlatBvXorExpr<T : KBvSort>(
        ctx: KContext,
        override val args: List<KExpr<T>>
    ) : KApp<T, T>(ctx), KSimplifierAuxExpr {

        override val decl: KDecl<T>
            get() = ctx.mkBvXorDecl(sort, sort)

        override val sort: T
            get() = args.first().sort

        override fun accept(transformer: KTransformerBase): KExpr<T> {
            transformer as KBvExprSimplifier
            return transformer.transform(this)
        }
    }

    /**
     * Auxiliary expression to store n-ary bv-concat.
     * @see [SimplifierAuxExpression]
     * */
    private class SimplifierFlatBvConcatExpr(
        ctx: KContext,
        override val sort: KBvSort,
        override val args: List<KExpr<KBvSort>>
    ) : KApp<KBvSort, KBvSort>(ctx), KSimplifierAuxExpr {

        // We have no decl, but we don't care since decl is unused
        override val decl: KDecl<KBvSort>
            get() = error("Decl of SimplifierFlatBvConcatExpr should not be used")

        override fun accept(transformer: KTransformerBase): KExpr<KBvSort> {
            transformer as KBvExprSimplifier
            return transformer.transform(this)
        }

        override fun print(printer: ExpressionPrinter) {
            with(printer) {
                append("(concat")
                for (arg in args) {
                    append(" ")
                    append(arg)
                }
                append(")")
            }
        }
    }
}

