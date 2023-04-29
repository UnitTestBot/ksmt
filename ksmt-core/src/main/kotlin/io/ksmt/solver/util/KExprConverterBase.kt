package io.ksmt.solver.util

import io.ksmt.expr.KExpr
import io.ksmt.solver.util.KExprConverterUtils.argumentsConversionRequired
import io.ksmt.sort.KSort

abstract class KExprConverterBase<T : Any> {
    abstract fun findConvertedNative(expr: T): KExpr<*>?

    abstract fun saveConvertedNative(native: T, converted: KExpr<*>)

    abstract fun convertNativeExpr(expr: T): ExprConversionResult

    val exprStack = arrayListOf<T>()

    fun <S : KSort> T.convertFromNative(): KExpr<S> = conversionLoop(
        stack = exprStack,
        native = this,
        stackPush = { stack, element -> stack.add(element) },
        stackPop = { stack -> stack.removeLast() },
        stackIsNotEmpty = { stack -> stack.isNotEmpty() },
        convertNative = { expr -> convertNativeExpr(expr) },
        findConverted = { expr -> findConvertedNative(expr) },
        saveConverted = { expr, converted -> saveConvertedNative(expr, converted) }
    )

    /**
     * Ensure all expression arguments are already converted.
     * Return converted arguments or null if not all arguments converted.
     * */
    fun ensureArgsConvertedAndConvert(
        expr: T,
        args: Array<T>,
        expectedSize: Int
    ): List<KExpr<*>>? = ensureArgsConvertedAndConvert(
        stack = exprStack,
        expr = expr,
        args = args,
        expectedSize = expectedSize,
        arraySize = { it.size },
        arrayGet = { array, idx -> array[idx] },
        stackPush = { stack, element -> stack.add(element) },
        findConverted = { findConvertedNative(it) }
    )

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
        val convertedArgs = ensureArgsConvertedAndConvert(expr, args, expectedSize)
        return if (convertedArgs == null) {
            argumentsConversionRequired
        } else {
            ExprConversionResult(converter(convertedArgs))
        }
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
        op(
            convertedArgs[0] as KExpr<A0>,
            convertedArgs[1] as KExpr<A1>,
            convertedArgs[2] as KExpr<A2>
        )
    }

    @Suppress("UNCHECKED_CAST", "MagicNumber")
    inline fun <S : KSort, A0 : KSort, A1 : KSort, A2 : KSort, A3 : KSort> T.convert(
        args: Array<T>,
        op: (KExpr<A0>, KExpr<A1>, KExpr<A2>, KExpr<A3>) -> KExpr<S>
    ) = ensureArgsConvertedAndConvert(this, args, expectedSize = 4) { convertedArgs ->
        op(
            convertedArgs[0] as KExpr<A0>,
            convertedArgs[1] as KExpr<A1>,
            convertedArgs[2] as KExpr<A2>,
            convertedArgs[3] as KExpr<A3>
        )
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
}
