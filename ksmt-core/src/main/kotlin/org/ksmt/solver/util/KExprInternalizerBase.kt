package org.ksmt.solver.util

import org.ksmt.expr.KExpr
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KSort

abstract class KExprInternalizerBase<T : Any> : KTransformerBase {
    @JvmField
    val exprStack = arrayListOf<KExpr<*>>()

    abstract fun findInternalizedExpr(expr: KExpr<*>): T?

    abstract fun saveInternalizedExpr(expr: KExpr<*>, internalized: T)

    fun <S : KSort> KExpr<S>.internalizeExpr(): T =
        internalizationLoop(
            exprStack = exprStack,
            initialExpr = this,
            notInternalized = null,
            findInternalized = { findInternalizedExpr(it) }
        )!!

    inline fun <S : KExpr<*>> S.transform(operation: () -> T): S = transform(
        operation = { operation() },
        saveInternalized = { expr, internalized -> saveInternalizedExpr(expr, internalized) }
    )

    inline fun <reified A0 : T, S : KExpr<*>> S.transform(
        arg: KExpr<*>,
        operation: (A0) -> T
    ): S = transform(
        exprStack = exprStack,
        arg = arg,
        operation = { operation(it as A0) },
        notInternalized = null,
        findInternalized = { findInternalizedExpr(it) },
        saveInternalized = { expr, internalized -> saveInternalizedExpr(expr, internalized!!) }
    )

    inline fun <reified A0 : T, reified A1 : T, S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        operation: (A0, A1) -> T
    ): S = transform(
        exprStack = exprStack,
        arg0 = arg0,
        arg1 = arg1,
        operation = { a0, a1 -> operation(a0 as A0, a1 as A1) },
        notInternalized = null,
        findInternalized = { findInternalizedExpr(it) },
        saveInternalized = { expr, internalized -> saveInternalizedExpr(expr, internalized!!) }
    )

    inline fun <reified A0 : T, reified A1 : T, reified A2 : T, S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        arg2: KExpr<*>,
        operation: (A0, A1, A2) -> T
    ): S = transform(
        exprStack = exprStack,
        arg0 = arg0,
        arg1 = arg1,
        arg2 = arg2,
        operation = { a0, a1, a2 -> operation(a0 as A0, a1 as A1, a2 as A2) },
        notInternalized = null,
        findInternalized = { findInternalizedExpr(it) },
        saveInternalized = { expr, internalized -> saveInternalizedExpr(expr, internalized!!) }
    )

    inline fun <reified A0 : T, reified A1 : T, reified A2 : T, reified A3 : T, S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        arg2: KExpr<*>,
        arg3: KExpr<*>,
        operation: (A0, A1, A2, A3) -> T
    ): S = transform(
        exprStack = exprStack,
        arg0 = arg0,
        arg1 = arg1,
        arg2 = arg2,
        arg3 = arg3,
        operation = { a0, a1, a2, a3 -> operation(a0 as A0, a1 as A1, a2 as A2, a3 as A3) },
        notInternalized = null,
        findInternalized = { findInternalizedExpr(it) },
        saveInternalized = { expr, internalized -> saveInternalizedExpr(expr, internalized!!) }
    )

    inline fun <reified A : T, S : KExpr<*>> S.transformList(
        args: List<KExpr<*>>,
        operation: (Array<A>) -> T
    ): S = transformList(
        exprStack = exprStack,
        args = args,
        operation = { transformedArgs ->
            @Suppress("UNCHECKED_CAST")
            operation(transformedArgs as Array<A>)
        },
        mkArray = { size -> arrayOfNulls<A>(size) },
        setArray = { array, idx, value -> array[idx] = value as A },
        notInternalized = null,
        findInternalized = { findInternalizedExpr(it) },
        saveInternalized = { expr, internalized -> saveInternalizedExpr(expr, internalized!!) }
    )
}
