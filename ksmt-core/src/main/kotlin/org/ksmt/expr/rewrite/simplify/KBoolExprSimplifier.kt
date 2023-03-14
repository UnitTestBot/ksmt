package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KAndExpr
import org.ksmt.expr.KApp
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExpr
import org.ksmt.expr.KImpliesExpr
import org.ksmt.expr.KIteExpr
import org.ksmt.expr.KNotExpr
import org.ksmt.expr.KOrBinaryExpr
import org.ksmt.expr.KOrExpr
import org.ksmt.expr.KXorExpr
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort

interface KBoolExprSimplifier : KExprSimplifierBase {

    override fun transform(expr: KAndExpr): KExpr<KBoolSort> = simplifyExpr(
        expr = expr,
        preprocess = {
            SimplifierStagedBooleanOperationExpr(
                ctx,
                operation = SimplifierBooleanOperation.AND,
                neutralElement = trueExpr,
                zeroElement = falseExpr,
                args = expr.args
            )
        }
    )

    override fun transform(expr: KOrExpr): KExpr<KBoolSort> = simplifyExpr(
        expr = expr,
        preprocess = {
            SimplifierStagedBooleanOperationExpr(
                ctx,
                operation = SimplifierBooleanOperation.OR,
                neutralElement = falseExpr,
                zeroElement = trueExpr,
                args = expr.args
            )
        }
    )

    /**
     * Perform simplification of AND/OR expression arguments one by one.
     * */
    private fun transform(expr: SimplifierStagedBooleanOperationExpr) =
        if (!expr.hasUnprocessedArgument()) {
            when (expr.operation) {
                SimplifierBooleanOperation.AND -> ctx.simplifyAnd(expr.simplifiedArgs)
                SimplifierBooleanOperation.OR -> ctx.simplifyOr(expr.simplifiedArgs)
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
        retryExprTransformation(expr)

        return expr
    }

    override fun transform(expr: KNotExpr) = simplifyExpr(expr, expr.arg) { arg ->
        simplifyNot(arg)
    }

    /**
     * Simplify ite expression in two stages:
     * 1. Simplify condition only [SimplifierStagedIteCondition].
     * If condition is true/false only one branch simplification is required.
     * 2. Simplify ite branches [SimplifierStagedIteBranches].
     * */
    override fun <T : KSort> transform(expr: KIteExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            preprocess = {
                SimplifierStagedIteCondition(ctx, expr.condition, expr.trueBranch, expr.falseBranch)
            }
        )

    private fun <T : KSort> transform(expr: SimplifierStagedIteCondition<T>): KExpr<T> =
        simplifyExpr(expr, expr.condition) { simplifiedCondition ->
            var c = simplifiedCondition
            var t = expr.trueBranch
            var e = expr.falseBranch

            // (ite (not c) a b) ==> (ite c b a)
            if (c is KNotExpr) {
                c = c.arg
                val tmp = t
                t = e
                e = tmp
            }

            // (ite true t e) ==> t
            if (c == trueExpr) {
                return@simplifyExpr rewrite(t)
            }

            // (ite false t e) ==> e
            if (c == falseExpr) {
                return@simplifyExpr rewrite(e)
            }

            rewrite(SimplifierStagedIteBranches(ctx, c, t, e))
        }

    private fun <T : KSort> transform(expr: SimplifierStagedIteBranches<T>): KExpr<T> =
        simplifyExpr(expr, expr.trueBranch, expr.falseBranch) { simplifiedTrueBranch, simplifiedFalseBranch ->
            simplifyIte(expr.simplifiedCondition, simplifiedTrueBranch, simplifiedFalseBranch)
        }

    override fun transform(expr: KImpliesExpr): KExpr<KBoolSort> = simplifyExpr(
        expr = expr,
        preprocess = {
            val notP = KNotExpr(this, expr.p)
            KOrBinaryExpr(this, notP, expr.q)
        }
    )

    override fun transform(expr: KXorExpr): KExpr<KBoolSort> = simplifyExpr(
        expr = expr,
        preprocess = {
            val notA = KNotExpr(this, expr.a)
            KEqExpr(this, notA, expr.b)
        }
    )

    fun simplifyEqBool(lhs: KExpr<KBoolSort>, rhs: KExpr<KBoolSort>): KExpr<KBoolSort> =
        ctx.simplifyEqBool(lhs, rhs)

    fun areDefinitelyDistinctBool(lhs: KExpr<KBoolSort>, rhs: KExpr<KBoolSort>): Boolean =
        lhs.isComplement(rhs)

    private fun flatAnd(expr: KAndExpr) = flatExpr<KAndExpr>(expr) { it.args }
    private fun flatOr(expr: KOrExpr) = flatExpr<KOrExpr>(expr) { it.args }

    private inline fun <reified T> flatExpr(
        initial: KExpr<KBoolSort>,
        getArgs: (T) -> List<KExpr<KBoolSort>>,
    ): List<KExpr<KBoolSort>> {
        val flatten = arrayListOf<KExpr<KBoolSort>>()
        val unprocessed = arrayListOf<KExpr<KBoolSort>>()
        unprocessed += initial
        while (unprocessed.isNotEmpty()) {
            val e = unprocessed.removeLast()
            if (e !is T) {
                flatten += e
                continue
            }
            unprocessed += getArgs(e).asReversed()
        }
        return flatten
    }

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
