package org.ksmt.solver.util

import org.ksmt.expr.KExpr
import org.ksmt.expr.transformer.KTransformerBase

inline fun <T> KTransformerBase.internalizationLoop(
    exprStack: MutableList<KExpr<*>>,
    initialExpr: KExpr<*>,
    notInternalized: T,
    findInternalized: (KExpr<*>) -> T
): T {
    exprStack.add(initialExpr)

    while (exprStack.isNotEmpty()) {
        val expr = exprStack.removeLast()

        val internalized = findInternalized(expr)
        if (internalized != notInternalized) continue

        /**
         * Internalize expression non-recursively.
         * 1. Ensure all expression arguments are internalized.
         * If not so, internalize arguments first
         * 2. Internalize expression if all arguments are available
         * */
        expr.accept(this)
    }

    return findInternalized(initialExpr).also {
        check(it != notInternalized) { "expr is not properly internalized: $initialExpr" }
    }
}

inline fun <S : KExpr<*>, T> S.transform(
    operation: () -> T,
    saveInternalized: (KExpr<*>, T) -> Unit
): S = also {
    saveInternalized(this, operation())
}

@Suppress("LongParameterList")
inline fun <S : KExpr<*>, T> S.transform(
    exprStack: MutableList<KExpr<*>>,
    arg: KExpr<*>,
    operation: (T) -> T,
    notInternalized: T,
    findInternalized: (KExpr<*>) -> T,
    saveInternalized: (KExpr<*>, T) -> Unit
): S = also {
    val internalizedArg = findInternalized(arg)

    if (internalizedArg == notInternalized) {
        exprStack.add(this)
        exprStack.add(arg)
    } else {
        saveInternalized(this, operation(internalizedArg))
    }
}

@Suppress("LongParameterList")
inline fun <S : KExpr<*>, T> S.transform(
    exprStack: MutableList<KExpr<*>>,
    arg0: KExpr<*>,
    arg1: KExpr<*>,
    operation: (T, T) -> T,
    notInternalized: T,
    findInternalized: (KExpr<*>) -> T,
    saveInternalized: (KExpr<*>, T) -> Unit
): S = also {
    val internalizedArg0 = findInternalized(arg0)
    val internalizedArg1 = findInternalized(arg1)

    if (internalizedArg0 == notInternalized || internalizedArg1 == notInternalized) {
        exprStack.add(this)

        if (internalizedArg0 == notInternalized) {
            exprStack.add(arg0)
        }
        if (internalizedArg1 == notInternalized) {
            exprStack.add(arg1)
        }
    } else {
        val internalized = operation(internalizedArg0, internalizedArg1)
        saveInternalized(this, internalized)
    }
}

@Suppress("LongParameterList")
inline fun <S : KExpr<*>, T> S.transform(
    exprStack: MutableList<KExpr<*>>,
    arg0: KExpr<*>,
    arg1: KExpr<*>,
    arg2: KExpr<*>,
    operation: (T, T, T) -> T,
    notInternalized: T,
    findInternalized: (KExpr<*>) -> T,
    saveInternalized: (KExpr<*>, T) -> Unit
): S = also {
    val internalizedArg0 = findInternalized(arg0)
    val internalizedArg1 = findInternalized(arg1)
    val internalizedArg2 = findInternalized(arg2)

    val someArgumentIsNotInternalized =
        internalizedArg0 == notInternalized
                || internalizedArg1 == notInternalized
                || internalizedArg2 == notInternalized

    if (someArgumentIsNotInternalized) {
        exprStack.add(this)

        if (internalizedArg0 == notInternalized) {
            exprStack.add(arg0)
        }
        if (internalizedArg1 == notInternalized) {
            exprStack.add(arg1)
        }
        if (internalizedArg2 == notInternalized) {
            exprStack.add(arg2)
        }
    } else {
        val internalized = operation(internalizedArg0, internalizedArg1, internalizedArg2)
        saveInternalized(this, internalized)
    }
}

@Suppress("LongParameterList")
inline fun <S : KExpr<*>, T> S.transform(
    exprStack: MutableList<KExpr<*>>,
    arg0: KExpr<*>,
    arg1: KExpr<*>,
    arg2: KExpr<*>,
    arg3: KExpr<*>,
    operation: (T, T, T, T) -> T,
    notInternalized: T,
    findInternalized: (KExpr<*>) -> T,
    saveInternalized: (KExpr<*>, T) -> Unit
): S = also {
    val internalizedArg0 = findInternalized(arg0)
    val internalizedArg1 = findInternalized(arg1)
    val internalizedArg2 = findInternalized(arg2)
    val internalizedArg3 = findInternalized(arg3)

    val someArgumentIsNotInternalized =
        internalizedArg0 == notInternalized
                || internalizedArg1 == notInternalized
                || internalizedArg2 == notInternalized
                || internalizedArg3 == notInternalized

    if (someArgumentIsNotInternalized) {
        exprStack.add(this)

        if (internalizedArg0 == notInternalized) {
            exprStack.add(arg0)
        }
        if (internalizedArg1 == notInternalized) {
            exprStack.add(arg1)
        }
        if (internalizedArg2 == notInternalized) {
            exprStack.add(arg2)
        }
        if (internalizedArg3 == notInternalized) {
            exprStack.add(arg3)
        }
    } else {

        val internalized = operation(
            internalizedArg0,
            internalizedArg1,
            internalizedArg2,
            internalizedArg3
        )
        saveInternalized(this, internalized)
    }
}

@Suppress("LongParameterList")
inline fun <S : KExpr<*>, T, A> S.transformList(
    exprStack: MutableList<KExpr<*>>,
    args: List<KExpr<*>>,
    operation: (A) -> T,
    mkArray: (Int) -> A,
    setArray: (A, Int, T) -> Unit,
    notInternalized: T,
    findInternalized: (KExpr<*>) -> T,
    saveInternalized: (KExpr<*>, T) -> Unit
): S = also {
    val internalizedArgs = mkArray(args.size)
    var hasNonInternalizedArgs = false

    for (i in args.indices) {
        val arg = args[i]
        val internalized = findInternalized(arg)

        if (internalized != notInternalized) {
            setArray(internalizedArgs, i, internalized)
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
        saveInternalized(this, internalized)
    }
}
