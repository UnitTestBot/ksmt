package org.ksmt.solver.util

import org.ksmt.expr.KExpr
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KSort
import org.ksmt.utils.uncheckedCast

abstract class KExprInternalizerBase<T : Any> : KTransformerBase {
    @JvmField
    val exprStack = arrayListOf<KExpr<*>>()

    abstract fun findInternalizedExpr(expr: KExpr<*>): T?

    abstract fun saveInternalizedExpr(expr: KExpr<*>, internalized: T)

    fun <S : KSort> KExpr<S>.internalizeExpr(): T {
        exprStack.add(this)

        while (exprStack.isNotEmpty()) {
            val expr = exprStack.removeLast()

            val internalized = findInternalizedExpr(expr)
            if (internalized != null) continue

            /**
             * Internalize expression non-recursively.
             * 1. Ensure all expression arguments are internalized.
             * If not so, internalize arguments first
             * 2. Internalize expression if all arguments are available
             * */
            expr.accept(this@KExprInternalizerBase)
        }
        return findInternalizedExpr(this) ?: error("expr is not properly internalized: $this")
    }

    inline fun <S : KExpr<*>> S.transform(operation: () -> T): S = also {
        saveInternalizedExpr(this, operation())
    }

    inline fun <reified A0 : T, S : KExpr<*>> S.transform(
        arg: KExpr<*>,
        operation: (A0) -> T
    ): S = also {
        val internalizedArg = findInternalizedExpr(arg)

        if (internalizedArg == null) {
            exprStack.add(this)
            exprStack.add(arg)
        } else {
            val internalized = operation(internalizedArg as A0)
            saveInternalizedExpr(this, internalized)
        }
    }

    inline fun <reified A0 : T, reified A1 : T, S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        operation: (A0, A1) -> T
    ): S = also {
        val internalizedArg0 = findInternalizedExpr(arg0)
        val internalizedArg1 = findInternalizedExpr(arg1)

        if (internalizedArg0 == null || internalizedArg1 == null) {
            exprStack.add(this)
            internalizedArg0 ?: exprStack.add(arg0)
            internalizedArg1 ?: exprStack.add(arg1)
        } else {
            val internalized = operation(internalizedArg0 as A0, internalizedArg1 as A1)
            saveInternalizedExpr(this, internalized)
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

        if (someArgumentIsNull) {
            exprStack.add(this)
            internalizedArg0 ?: exprStack.add(arg0)
            internalizedArg1 ?: exprStack.add(arg1)
            internalizedArg2 ?: exprStack.add(arg2)
        } else {
            val internalized = operation(internalizedArg0 as A0, internalizedArg1 as A1, internalizedArg2 as A2)
            saveInternalizedExpr(this, internalized)
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

        val someArgumentIsNull =
            internalizedArg0 == null
                    || internalizedArg1 == null
                    || internalizedArg2 == null
                    || internalizedArg3 == null

        if (someArgumentIsNull) {
            exprStack.add(this)

            internalizedArg0 ?: exprStack.add(arg0)
            internalizedArg1 ?: exprStack.add(arg1)
            internalizedArg2 ?: exprStack.add(arg2)
            internalizedArg3 ?: exprStack.add(arg3)
        } else {
            val internalized = operation(
                internalizedArg0 as A0,
                internalizedArg1 as A1,
                internalizedArg2 as A2,
                internalizedArg3 as A3
            )
            saveInternalizedExpr(this, internalized)
        }
    }

    inline fun <reified A : T, S : KExpr<*>> S.transformList(
        args: List<KExpr<*>>,
        operation: (Array<A>) -> T
    ): S = also {
        val internalizedArgs = arrayOfNulls<A>(args.size)
        var hasNonInternalizedArgs = false

        for (i in args.indices) {
            val arg = args[i]
            val internalized = findInternalizedExpr(arg)

            if (internalized != null) {
                internalizedArgs[i] = internalized as A
                continue
            }

            if (!hasNonInternalizedArgs) {
                hasNonInternalizedArgs = true
                exprStack.add(this)
            }

            exprStack.add(arg)
        }

        if (!hasNonInternalizedArgs) {
            val internalized = operation(internalizedArgs.uncheckedCast())
            saveInternalizedExpr(this, internalized)
        }
    }
}
