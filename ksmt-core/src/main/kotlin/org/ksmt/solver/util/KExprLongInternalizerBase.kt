package org.ksmt.solver.util

import org.ksmt.expr.KExpr
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KSort

/**
 * Specialized version of [KExprInternalizerBase] for Long native expressions.
 * */
abstract class KExprLongInternalizerBase : KTransformerBase {
    @JvmField
    val exprStack = arrayListOf<KExpr<*>>()

    /**
     * Return internalized expression or
     * [NOT_INTERNALIZED] if expression was not internalized yet.
     * */
    abstract fun findInternalizedExpr(expr: KExpr<*>): Long

    abstract fun saveInternalizedExpr(expr: KExpr<*>, internalized: Long)

    fun <S : KSort> KExpr<S>.internalizeExpr(): Long = internalizationLoop(
        exprStack = exprStack,
        initialExpr = this,
        notInternalized = NOT_INTERNALIZED,
        findInternalized = { findInternalizedExpr(it) }
    )

    inline fun <S : KExpr<*>> S.transform(operation: () -> Long): S =
        transform(
            operation = { operation() },
            saveInternalized = { expr, internalized -> saveInternalizedExpr(expr, internalized) }
        )

    inline fun <S : KExpr<*>> S.transform(
        arg: KExpr<*>,
        operation: (Long) -> Long
    ): S = transform(
        exprStack = exprStack,
        arg = arg,
        operation = { operation(it) },
        notInternalized = NOT_INTERNALIZED,
        findInternalized = { findInternalizedExpr(it) },
        saveInternalized = { expr, internalized -> saveInternalizedExpr(expr, internalized) }
    )

    inline fun <S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        operation: (Long, Long) -> Long
    ): S = transform(
        exprStack = exprStack,
        arg0 = arg0,
        arg1 = arg1,
        operation = { a0, a1 -> operation(a0, a1) },
        notInternalized = NOT_INTERNALIZED,
        findInternalized = { findInternalizedExpr(it) },
        saveInternalized = { expr, internalized -> saveInternalizedExpr(expr, internalized) }
    )

    inline fun <S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        arg2: KExpr<*>,
        operation: (Long, Long, Long) -> Long
    ): S = transform(
        exprStack = exprStack,
        arg0 = arg0,
        arg1 = arg1,
        arg2 = arg2,
        operation = { a0, a1, a2 -> operation(a0, a1, a2) },
        notInternalized = NOT_INTERNALIZED,
        findInternalized = { findInternalizedExpr(it) },
        saveInternalized = { expr, internalized -> saveInternalizedExpr(expr, internalized) }
    )

    inline fun <S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        arg2: KExpr<*>,
        arg3: KExpr<*>,
        operation: (Long, Long, Long, Long) -> Long
    ): S = transform(
        exprStack = exprStack,
        arg0 = arg0,
        arg1 = arg1,
        arg2 = arg2,
        arg3 = arg3,
        operation = { a0, a1, a2, a3 -> operation(a0, a1, a2, a3) },
        notInternalized = NOT_INTERNALIZED,
        findInternalized = { findInternalizedExpr(it) },
        saveInternalized = { expr, internalized -> saveInternalizedExpr(expr, internalized) }
    )

    inline fun <S : KExpr<*>> S.transformList(
        args: List<KExpr<*>>,
        operation: (LongArray) -> Long
    ): S = transformList(
        exprStack = exprStack,
        args = args,
        operation = { transformedArgs -> operation(transformedArgs) },
        mkArray = { size -> LongArray(size) },
        setArray = { array, idx, value -> array[idx] = value },
        notInternalized = NOT_INTERNALIZED,
        findInternalized = { findInternalizedExpr(it) },
        saveInternalized = { expr, internalized -> saveInternalizedExpr(expr, internalized) }
    )

    companion object {
        const val NOT_INTERNALIZED = -1L
    }
}
