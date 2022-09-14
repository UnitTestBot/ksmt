package org.ksmt.solver.util

import org.ksmt.expr.KExpr
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.solver.util.KExprInternalizerBase.ExprInternalizationResult.Companion.argumentsInternalizationRequired
import org.ksmt.solver.util.KExprInternalizerBase.ExprInternalizationResult.Companion.notInitializedInternalizationResult
import org.ksmt.sort.KSort

abstract class KExprInternalizerBase<T : Any> : KTransformerBase {
    val exprStack = arrayListOf<KExpr<*>>()

    /**
     * Keeps result of last [KTransformerBase.transform] invocation.
     * */
    var lastExprInternalizationResult: ExprInternalizationResult = notInitializedInternalizationResult

    abstract fun findInternalizedExpr(expr: KExpr<*>): T?

    abstract fun saveInternalizedExpr(expr: KExpr<*>, internalized: T)

    fun <S : KSort> KExpr<S>.internalizeExpr(): T {
        exprStack.add(this)

        while (exprStack.isNotEmpty()) {
            lastExprInternalizationResult = notInitializedInternalizationResult
            val expr = exprStack.removeLast()

            val internalized = findInternalizedExpr(expr)
            if (internalized != null) continue

            /**
             * Internalize expression non-recursively.
             * 1. Ensure all expression arguments are internalized.
             * If not so, [lastExprInternalizationResult] is set to [argumentsInternalizationRequired]
             * 2. Internalize expression if all arguments are available and
             * set [lastExprInternalizationResult] to internalization result.
             * */
            expr.accept(this@KExprInternalizerBase)

            check(!lastExprInternalizationResult.notInitialized) {
                "Internalization result wasn't initialized during expr internalization"
            }

            if (!lastExprInternalizationResult.isArgumentsInternalizationRequired) {
                saveInternalizedExpr(expr, lastExprInternalizationResult.internalizedExpr())
            }
        }
        return findInternalizedExpr(this) ?: error("expr is not properly internalized: $this")
    }

    @Suppress("UNCHECKED_CAST")
    fun ExprInternalizationResult.internalizedExpr(): T =
        internalizedExprInternal as? T ?: error("expr is not internalized")

    inline fun <S : KExpr<*>> S.transform(operation: () -> T): S = also {
        lastExprInternalizationResult = ExprInternalizationResult(operation())
    }

    inline fun <reified A0 : T, S : KExpr<*>> S.transform(
        arg: KExpr<*>,
        operation: (A0) -> T
    ): S = also {
        val internalizedArg = findInternalizedExpr(arg)

        lastExprInternalizationResult = if (internalizedArg == null) {
            exprStack.add(this)
            exprStack.add(arg)
            argumentsInternalizationRequired
        } else {
            ExprInternalizationResult(operation(internalizedArg as A0))
        }
    }

    inline fun <reified A0 : T, reified A1 : T, S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        operation: (A0, A1) -> T
    ): S = also {
        val internalizedArg0 = findInternalizedExpr(arg0)
        val internalizedArg1 = findInternalizedExpr(arg1)

        lastExprInternalizationResult = if (internalizedArg0 == null || internalizedArg1 == null) {
            exprStack.add(this)
            internalizedArg0 ?: exprStack.add(arg0)
            internalizedArg1 ?: exprStack.add(arg1)
            argumentsInternalizationRequired
        } else {
            ExprInternalizationResult(
                operation(internalizedArg0 as A0, internalizedArg1 as A1)
            )
        }
    }

    inline fun <reified A0 : T, reified A1 : T, reified A2 : T, S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        arg2: KExpr<*>,
        operation: (A0, A1, A2) -> T
    ): S = also {
        val internalizedArg0 = findInternalizedExpr(arg0)
        val internalizedArg1 = findInternalizedExpr(arg1)
        val internalizedArg2 = findInternalizedExpr(arg2)

        val someArgumentIsNull = internalizedArg0 == null || internalizedArg1 == null || internalizedArg2 == null

        lastExprInternalizationResult = if (someArgumentIsNull) {
            exprStack.add(this)
            internalizedArg0 ?: exprStack.add(arg0)
            internalizedArg1 ?: exprStack.add(arg1)
            internalizedArg2 ?: exprStack.add(arg2)
            argumentsInternalizationRequired
        } else {
            ExprInternalizationResult(
                operation(internalizedArg0 as A0, internalizedArg1 as A1, internalizedArg2 as A2)
            )
        }
    }

    inline fun <reified A0 : T, reified A1 : T, reified A2 : T, reified A3 : T, S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        arg2: KExpr<*>,
        arg3: KExpr<*>,
        operation: (A0, A1, A2, A3) -> T
    ): S = also {
        val internalizedArg0 = findInternalizedExpr(arg0)
        val internalizedArg1 = findInternalizedExpr(arg1)
        val internalizedArg2 = findInternalizedExpr(arg2)
        val internalizedArg3 = findInternalizedExpr(arg3)

        val args = listOf(internalizedArg0, internalizedArg1, internalizedArg2, internalizedArg3)
        val someArgumentIsNull = args.any { it == null }

        if (someArgumentIsNull) {
            exprStack.add(this)

            internalizedArg0 ?: exprStack.add(arg0)
            internalizedArg1 ?: exprStack.add(arg1)
            internalizedArg2 ?: exprStack.add(arg2)
            internalizedArg3 ?: exprStack.add(arg3)

            lastExprInternalizationResult = argumentsInternalizationRequired
        } else {
            lastExprInternalizationResult = ExprInternalizationResult(
                operation(
                    internalizedArg0 as A0,
                    internalizedArg1 as A1,
                    internalizedArg2 as A2,
                    internalizedArg3 as A3
                )
            )
        }
    }

    inline fun <reified A : T, S : KExpr<*>> S.transformList(
        args: List<KExpr<*>>,
        operation: (Array<A>) -> T
    ): S = also {
        val internalizedArgs = mutableListOf<A>()
        var exprAdded = false
        var argsReady = true

        for (arg in args) {
            val internalized = findInternalizedExpr(arg)

            if (internalized != null) {
                internalizedArgs.add(internalized as A)
                continue
            }

            argsReady = false

            if (!exprAdded) {
                exprStack.add(this)
                exprAdded = true
            }

            exprStack.add(arg)
        }

        lastExprInternalizationResult = if (argsReady) {
            ExprInternalizationResult(operation(internalizedArgs.toTypedArray()))
        } else {
            argumentsInternalizationRequired
        }
    }

    @JvmInline
    value class ExprInternalizationResult(private val value: Any) {
        val isArgumentsInternalizationRequired: Boolean
            get() = value === argumentsInternalizationRequiredMarker

        val notInitialized: Boolean
            get() = value === notInitializedInternalizationResultMarker

        val internalizedExprInternal: Any
            get() = value

        companion object {
            private val argumentsInternalizationRequiredMarker = Any()
            private val notInitializedInternalizationResultMarker = Any()
            val argumentsInternalizationRequired = ExprInternalizationResult(argumentsInternalizationRequiredMarker)
            val notInitializedInternalizationResult =
                ExprInternalizationResult(notInitializedInternalizationResultMarker)
        }
    }
}
