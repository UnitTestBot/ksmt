package org.ksmt.solver.util

import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

abstract class KExprConverterBase<T : Any> {
    abstract fun findConvertedNative(expr: T): KExpr<*>?
    abstract fun saveConvertedNative(native: T, converted: KExpr<*>)
    abstract fun convertNativeExpr(expr: T): ExprConversionResult

    val exprStack = arrayListOf<T>()

    fun <S : KSort> T.convertFromNative(): KExpr<S> {
        exprStack.add(this)
        while (exprStack.isNotEmpty()) {
            val expr = exprStack.removeLast()

            if (findConvertedNative(expr) != null) continue

            val converted = convertNativeExpr(expr)

            if (!converted.argumentsConversionRequired) {
                saveConvertedNative(expr, converted.convertedExpr)
            }
        }
        @Suppress("UNCHECKED_CAST")
        return findConvertedNative(this) as? KExpr<S>
            ?: error("expr is not properly converted")
    }

    /**
     * Ensure all expression arguments are already converted.
     * If not so, [argumentsConversionRequired] is returned.
     * */
    inline fun ensureArgsConvertedAndConvert(
        expr: T,
        args: Array<T>,
        expectedSize: Int,
        converter: (List<KExpr<*>>) -> KExpr<*>
    ): ExprConversionResult {
        check(args.size == expectedSize) { "arguments size mismatch: expected $expectedSize, actual ${args.size}" }
        val convertedArgs = mutableListOf<KExpr<*>>()
        var exprAdded = false
        var argsReady = true
        for (arg in args) {
            val converted = findConvertedNative(arg)
            if (converted != null) {
                convertedArgs.add(converted)
                continue
            }
            argsReady = false
            if (!exprAdded) {
                exprStack.add(expr)
                exprAdded = true
            }
            exprStack.add(arg)
        }

        if (!argsReady) return argumentsConversionRequired

        val convertedExpr = converter(convertedArgs)
        return ExprConversionResult(convertedExpr)
    }

    inline fun <T : KSort> convert(op: () -> KExpr<T>) = ExprConversionResult(op())

    @Suppress("UNCHECKED_CAST")
    inline fun <S : KSort, A0 : KSort> T.convert(
        args: Array<T>,
        op: (KExpr<A0>) -> KExpr<S>
    ) = ensureArgsConvertedAndConvert(this, args, expectedSize = 1) { convertedArgs ->
        op(convertedArgs[0] as KExpr<A0>)
    }

    @Suppress("UNCHECKED_CAST")
    inline fun <S : KSort, A0 : KSort, A1 : KSort> T.convert(
        args: Array<T>,
        op: (KExpr<A0>, KExpr<A1>) -> KExpr<S>
    ) = ensureArgsConvertedAndConvert(this, args, expectedSize = 2) { convertedArgs ->
        op(convertedArgs[0] as KExpr<A0>, convertedArgs[1] as KExpr<A1>)
    }

    @Suppress("UNCHECKED_CAST", "MagicNumber")
    inline fun <S : KSort, A0 : KSort, A1 : KSort, A2 : KSort> T.convert(
        args: Array<T>,
        op: (KExpr<A0>, KExpr<A1>, KExpr<A2>) -> KExpr<S>
    ) = ensureArgsConvertedAndConvert(this, args, expectedSize = 3) { convertedArgs ->
        op(convertedArgs[0] as KExpr<A0>, convertedArgs[1] as KExpr<A1>, convertedArgs[2] as KExpr<A2>)
    }

    @Suppress("UNCHECKED_CAST")
    inline fun <S : KSort, A : KSort> T.convertList(
        args: Array<T>,
        op: (List<KExpr<A>>) -> KExpr<S>
    ) = ensureArgsConvertedAndConvert(this, args, expectedSize = args.size) { convertedArgs ->
        op(convertedArgs as List<KExpr<A>>)
    }

    @Suppress("UNCHECKED_CAST")
    inline fun <S : KSort> T.convertReduced(
        args: Array<T>,
        op: (KExpr<S>, KExpr<S>) -> KExpr<S>
    ) = ensureArgsConvertedAndConvert(this, args, expectedSize = args.size) { convertedArgs ->
        (convertedArgs as List<KExpr<S>>).reduce(op)
    }

    @JvmInline
    value class ExprConversionResult(private val expr: KExpr<*>?) {
        val argumentsConversionRequired: Boolean
            get() = expr == null

        val convertedExpr: KExpr<*>
            get() = expr ?: error("expr is not converted")
    }

    val argumentsConversionRequired = ExprConversionResult(null)
}
