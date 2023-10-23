package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KAndExpr
import io.ksmt.expr.KApp
import io.ksmt.expr.KEqExpr
import io.ksmt.expr.KExpr
import io.ksmt.expr.KImpliesExpr
import io.ksmt.expr.KIteExpr
import io.ksmt.expr.KNotExpr
import io.ksmt.expr.KOrBinaryExpr
import io.ksmt.expr.KOrExpr
import io.ksmt.expr.KXorExpr
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KSort

interface KBoolExprSimplifier : KExprSimplifierBase {

    fun KContext.preprocess(expr: KAndExpr): KExpr<KBoolSort> =
        SimplifierStagedBooleanOperationExpr(
            ctx,
            operation = SimplifierBooleanOperation.AND,
            neutralElement = trueExpr,
            zeroElement = falseExpr,
            args = expr.args
        )

    fun KContext.postRewriteAnd(args: List<KExpr<KBoolSort>>): KExpr<KBoolSort> = simplifyAnd(args)

    override fun transform(expr: KAndExpr): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            args = expr.args,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteAnd(it) }
        )

    fun KContext.preprocess(expr: KOrExpr): KExpr<KBoolSort> =
        SimplifierStagedBooleanOperationExpr(
            ctx,
            operation = SimplifierBooleanOperation.OR,
            neutralElement = falseExpr,
            zeroElement = trueExpr,
            args = expr.args
        )

    fun KContext.postRewriteOr(args: List<KExpr<KBoolSort>>): KExpr<KBoolSort> = simplifyOr(args)

    override fun transform(expr: KOrExpr): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            args = expr.args,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteOr(it) }
        )

    /**
     * Perform simplification of AND/OR expression arguments one by one.
     * */
    private fun transform(expr: SimplifierStagedBooleanOperationExpr) =
        if (!expr.hasUnprocessedArgument()) {
            when (expr.operation) {
                SimplifierBooleanOperation.AND -> ctx.postRewriteAnd(expr.simplifiedArgs)
                SimplifierBooleanOperation.OR -> ctx.postRewriteOr(expr.simplifiedArgs)
            }
        } else {
            this as KExprSimplifier
            stagedBooleanOperationStep(expr)
        }

    private fun KExprSimplifier.stagedBooleanOperationStep(
        expr: SimplifierStagedBooleanOperationExpr
    ): KExpr<KBoolSort> {
        val argument = expr.currentArgument()
        val simplifiedArgument = transformedExpr(argument)

        // Simplify argument and retry expr simplification
        if (simplifiedArgument == null) {
            markExpressionAsNotTransformed()
            expr.transformAfter(argument)
            return expr
        }

        if (simplifiedArgument == expr.zeroElement) return expr.zeroElement

        if (simplifiedArgument != expr.neutralElement) {
            expr.simplifiedArgs.add(simplifiedArgument)
        }

        // Select next argument to simplify
        expr.processNextArgument()

        // Repeat simplification with next argument
        markExpressionAsNotTransformed()
        retryExprTransformation(expr)

        return expr
    }

    fun KContext.preprocess(expr: KNotExpr): KExpr<KBoolSort> = expr
    fun KContext.postRewriteNot(arg: KExpr<KBoolSort>): KExpr<KBoolSort> = simplifyNot(arg)

    override fun transform(expr: KNotExpr) =
        simplifyExpr(
            expr,
            expr.arg,
            preprocess = { preprocess(it) },
            simplifier = { postRewriteNot(it) }
        )

    fun <T : KSort> KContext.preprocess(expr: KIteExpr<T>): KExpr<T> =
        SimplifierStagedIteCondition(ctx, expr.condition, expr.trueBranch, expr.falseBranch)

    fun <T : KSort> KContext.postRewriteIte(
        condition: KExpr<KBoolSort>,
        trueBranch: KExpr<T>,
        falseBranch: KExpr<T>
    ): KExpr<T> = simplifyIte(condition, trueBranch, falseBranch)

    /**
     * Simplify ite expression in two stages:
     * 1. Simplify condition only [SimplifierStagedIteCondition].
     * If condition is true/false only one branch simplification is required.
     * 2. Simplify ite branches [SimplifierStagedIteBranches].
     * */
    override fun <T : KSort> transform(expr: KIteExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.condition,
            a1 = expr.trueBranch,
            a2 = expr.falseBranch,
            preprocess = { preprocess(it) },
            simplifier = { c, t, f -> postRewriteIte(c, t, f) }
        )

    private fun <T : KSort> transform(expr: SimplifierStagedIteCondition<T>): KExpr<T> =
        simplifyExpr(expr, expr.condition) { simplifiedCondition ->
            simplifyIteNotCondition(
                simplifiedCondition,
                expr.trueBranch,
                expr.falseBranch
            ) { condition2, trueBranch2, falseBranch2 ->
                rewrite(
                    simplifyIteLight(
                        condition2,
                        trueBranch2,
                        falseBranch2
                    ) { condition3, trueBranch3, falseBranch3 ->
                        SimplifierStagedIteBranches(ctx, condition3, trueBranch3, falseBranch3)
                    }
                )
            }
        }

    private fun <T : KSort> transform(expr: SimplifierStagedIteBranches<T>): KExpr<T> =
        simplifyExpr(expr, expr.trueBranch, expr.falseBranch) { simplifiedTrueBranch, simplifiedFalseBranch ->
            postRewriteIte(expr.simplifiedCondition, simplifiedTrueBranch, simplifiedFalseBranch)
        }

    fun KContext.preprocess(expr: KImpliesExpr): KExpr<KBoolSort> =
        rewriteImplies(
            p = expr.p,
            q = expr.q,
            rewriteNot = { KNotExpr(this, it) },
            rewriteOr = { l, r -> KOrBinaryExpr(this, l, r) }
        )

    fun KContext.postRewriteImplies(p: KExpr<KBoolSort>, q: KExpr<KBoolSort>): KExpr<KBoolSort> =
        error("Always preprocessed")

    override fun transform(expr: KImpliesExpr): KExpr<KBoolSort> = simplifyExpr(
        expr = expr,
        a0 = expr.p,
        a1 = expr.q,
        preprocess = { preprocess(it) },
        simplifier = { l, r -> postRewriteImplies(l, r) }
    )

    fun KContext.preprocess(expr: KXorExpr): KExpr<KBoolSort> =
        rewriteXor(
            a = expr.a,
            b = expr.b,
            rewriteNot = { KNotExpr(this, it) },
            rewriteEq = { l, r -> KEqExpr(this, l, r) }
        )

    fun KContext.postRewriteXor(a: KExpr<KBoolSort>, b: KExpr<KBoolSort>): KExpr<KBoolSort> =
        error("Always preprocessed")

    override fun transform(expr: KXorExpr): KExpr<KBoolSort> = simplifyExpr(
        expr = expr,
        a0 = expr.a,
        a1 = expr.b,
        preprocess = { preprocess(it) },
        simplifier = { l, r -> postRewriteXor(l, r) }
    )

    fun simplifyEqBool(lhs: KExpr<KBoolSort>, rhs: KExpr<KBoolSort>): KExpr<KBoolSort> =
        ctx.simplifyEqBool(lhs, rhs, order = true)

    fun areDefinitelyDistinctBool(lhs: KExpr<KBoolSort>, rhs: KExpr<KBoolSort>): Boolean =
        lhs.isComplement(rhs)

    /**
     * Auxiliary expression to perform ite condition (1) simplification stage.
     * Check out [KIteExpr] simplification.
     * @see [SimplifierAuxExpression]
     * */
    private class SimplifierStagedIteCondition<T : KSort>(
        ctx: KContext,
        val condition: KExpr<KBoolSort>,
        val trueBranch: KExpr<T>,
        val falseBranch: KExpr<T>
    ) : KApp<T, KBoolSort>(ctx) {

        override val decl: KDecl<T>
            get() = ctx.mkIteDecl(trueBranch.sort)

        override val sort: T
            get() = trueBranch.sort

        override val args: List<KExpr<KBoolSort>>
            get() = listOf(condition)

        override fun accept(transformer: KTransformerBase): KExpr<T> {
            transformer as KBoolExprSimplifier
            return transformer.transform(this)
        }
    }

    /**
     * Auxiliary expression to perform ite branch (2) simplification stage.
     * Check out [KIteExpr] simplification.
     * @see [SimplifierAuxExpression]
     * */
    private class SimplifierStagedIteBranches<T : KSort>(
        ctx: KContext,
        val simplifiedCondition: KExpr<KBoolSort>,
        val trueBranch: KExpr<T>,
        val falseBranch: KExpr<T>
    ) : KApp<T, T>(ctx) {

        override val decl: KDecl<T>
            get() = ctx.mkIteDecl(trueBranch.sort)

        override val sort: T
            get() = trueBranch.sort

        override val args: List<KExpr<T>>
            get() = listOf(trueBranch, falseBranch)

        override fun accept(transformer: KTransformerBase): KExpr<T> {
            transformer as KBoolExprSimplifier
            return transformer.transform(this)
        }
    }

    private enum class SimplifierBooleanOperation {
        AND, OR
    }

    /**
     * Auxiliary expression to perform simplification of AND/OR expression
     * arguments one by one.
     * @see [SimplifierAuxExpression]
     * */
    private class SimplifierStagedBooleanOperationExpr(
        ctx: KContext,
        val operation: SimplifierBooleanOperation,
        val neutralElement: KExpr<KBoolSort>,
        val zeroElement: KExpr<KBoolSort>,
        override val args: List<KExpr<KBoolSort>>
    ) : KApp<KBoolSort, KBoolSort>(ctx) {
        override val sort: KBoolSort = ctx.boolSort
        override val decl: KDecl<KBoolSort>
            get() = when (operation) {
                SimplifierBooleanOperation.AND -> ctx.mkAndDecl()
                SimplifierBooleanOperation.OR -> ctx.mkOrDecl()
            }

        override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> {
            transformer as KBoolExprSimplifier
            return transformer.transform(this)
        }

        val simplifiedArgs = arrayListOf<KExpr<KBoolSort>>()

        private val argsIteratorStack = arrayListOf(args.iterator())
        private var currentArgument: KExpr<KBoolSort>? = null

        fun currentArgument(): KExpr<KBoolSort> = currentArgument ?: neutralElement

        fun hasUnprocessedArgument(): Boolean {
            if (currentArgument != null) return true

            moveIterator()
            return argsIteratorStack.isNotEmpty()
        }

        /**
         * Select next argument to process.
         * If argument can be flattened the next argument
         * is the first argument of flattened expression.
         * */
        fun processNextArgument() {
            currentArgument = null

            while (hasUnprocessedArgument()) {
                val argument = argsIteratorStack.last().next()
                if (!tryFlatExpr(argument)) {
                    currentArgument = argument
                    return
                }
            }
        }

        private fun moveIterator() {
            while (argsIteratorStack.isNotEmpty() && !argsIteratorStack.last().hasNext()) {
                argsIteratorStack.removeLast()
            }
        }

        private fun tryFlatExpr(expr: KExpr<KBoolSort>): Boolean {
            val flatArgs = when {
                expr is KAndExpr && operation == SimplifierBooleanOperation.AND -> expr.args
                expr is KOrExpr && operation == SimplifierBooleanOperation.OR -> expr.args
                else -> return false
            }

            argsIteratorStack.add(flatArgs.iterator())
            return true
        }
    }
}
