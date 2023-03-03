package org.ksmt.solver.util

import org.ksmt.expr.KExpr
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.solver.util.KExprLongInternalizerBase.Companion.NOT_INTERNALIZED
import org.ksmt.sort.KSort

/**
 * Specialized version of [KExprInternalizerBase] for Int native expressions.
 * */
abstract class KExprIntInternalizerBase : KTransformerBase {
    @JvmField
    val exprStack = arrayListOf<KExpr<*>>()

    /**
     * Return internalized expression or
     * [NOT_INTERNALIZED] if expression was not internalized yet.
     * */
    abstract fun findInternalizedExpr(expr: KExpr<*>): Int

    abstract fun saveInternalizedExpr(expr: KExpr<*>, internalized: Int)

    fun <S : KSort> KExpr<S>.internalizeExpr(): Int {
        exprStack.add(this)

        while (exprStack.isNotEmpty()) {
            val expr = exprStack.removeLast()

            val internalized = findInternalizedExpr(expr)
            if (internalized != NOT_INTERNALIZED) continue

            expr.accept(this@KExprIntInternalizerBase)
        }

        return findInternalizedExpr(this).also {
            check(it != NOT_INTERNALIZED) { "expr is not properly internalized: $this" }
        }
    }

    inline fun <S : KExpr<*>> S.transform(operation: () -> Int): S = also {
        saveInternalizedExpr(this, operation())
    }

    inline fun <S : KExpr<*>> S.transform(
        arg: KExpr<*>,
        operation: (Int) -> Int
    ): S = also {
        val internalizedArg = findInternalizedExpr(arg)

        if (internalizedArg == NOT_INTERNALIZED) {
            exprStack.add(this)
            exprStack.add(arg)
        } else {
            val internalized = operation(internalizedArg)
            saveInternalizedExpr(this, internalized)
        }
    }

    inline fun <S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        operation: (Int, Int) -> Int
    ): S = also {
        val internalizedArg0 = findInternalizedExpr(arg0)
        val internalizedArg1 = findInternalizedExpr(arg1)

        if (internalizedArg0 == NOT_INTERNALIZED || internalizedArg1 == NOT_INTERNALIZED) {
            exprStack.add(this)

            if (internalizedArg0 == NOT_INTERNALIZED) {
                exprStack.add(arg0)
            }
            if (internalizedArg1 == NOT_INTERNALIZED) {
                exprStack.add(arg1)
            }
        } else {
            val internalized = operation(internalizedArg0, internalizedArg1)
            saveInternalizedExpr(this, internalized)
        }
    }

    inline fun <S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        arg2: KExpr<*>,
        operation: (Int, Int, Int) -> Int
    ): S = also {
        val internalizedArg0 = findInternalizedExpr(arg0)
        val internalizedArg1 = findInternalizedExpr(arg1)
        val internalizedArg2 = findInternalizedExpr(arg2)

        val someArgumentIsNotInternalzied =
            internalizedArg0 == NOT_INTERNALIZED
                    || internalizedArg1 == NOT_INTERNALIZED
                    || internalizedArg2 == NOT_INTERNALIZED

        if (someArgumentIsNotInternalzied) {
            exprStack.add(this)

            if (internalizedArg0 == NOT_INTERNALIZED) {
                exprStack.add(arg0)
            }
            if (internalizedArg1 == NOT_INTERNALIZED) {
                exprStack.add(arg1)
            }
            if (internalizedArg2 == NOT_INTERNALIZED) {
                exprStack.add(arg2)
            }
        } else {
            val internalized = operation(internalizedArg0, internalizedArg1, internalizedArg2)
            saveInternalizedExpr(this, internalized)
        }
    }

    inline fun <S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        arg2: KExpr<*>,
        arg3: KExpr<*>,
        operation: (Int, Int, Int, Int) -> Int
    ): S = also {
        val internalizedArg0 = findInternalizedExpr(arg0)
        val internalizedArg1 = findInternalizedExpr(arg1)
        val internalizedArg2 = findInternalizedExpr(arg2)
        val internalizedArg3 = findInternalizedExpr(arg3)

        val someArgumentIsNotInternalized =
            internalizedArg0 == NOT_INTERNALIZED
                    || internalizedArg1 == NOT_INTERNALIZED
                    || internalizedArg2 == NOT_INTERNALIZED
                    || internalizedArg3 == NOT_INTERNALIZED

        if (someArgumentIsNotInternalized) {
            exprStack.add(this)

            if (internalizedArg0 == NOT_INTERNALIZED) {
                exprStack.add(arg0)
            }
            if (internalizedArg1 == NOT_INTERNALIZED) {
                exprStack.add(arg1)
            }
            if (internalizedArg2 == NOT_INTERNALIZED) {
                exprStack.add(arg2)
            }
            if (internalizedArg3 == NOT_INTERNALIZED) {
                exprStack.add(arg3)
            }
        } else {

            val internalized = operation(
                internalizedArg0,
                internalizedArg1,
                internalizedArg2,
                internalizedArg3
            )
            saveInternalizedExpr(this, internalized)
        }
    }

    inline fun <S : KExpr<*>> S.transformList(
        args: List<KExpr<*>>,
        operation: (IntArray) -> Int
    ): S = also {
        val internalizedArgs = IntArray(args.size)
        var hasNonInternalizedArgs = false

        for (i in args.indices) {
            val arg = args[i]
            val internalized = findInternalizedExpr(arg)

            if (internalized != NOT_INTERNALIZED) {
                internalizedArgs[i] = internalized
                continue
            }

            if (!hasNonInternalizedArgs) {
                hasNonInternalizedArgs = true
                exprStack.add(this)
            }

            exprStack.add(arg)
        }

        if (!hasNonInternalizedArgs) {
            val internalized = operation(internalizedArgs)
            saveInternalizedExpr(this, internalized)
        }
    }

    companion object {
        const val NOT_INTERNALIZED = -1
    }
}
