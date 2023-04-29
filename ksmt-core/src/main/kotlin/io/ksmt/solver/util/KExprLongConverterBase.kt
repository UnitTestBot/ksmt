package io.ksmt.solver.util

import it.unimi.dsi.fastutil.longs.LongArrayList
import io.ksmt.expr.KExpr
import io.ksmt.solver.util.KExprConverterUtils.argumentsConversionRequired
import io.ksmt.sort.KSort

/**
 * Specialized version of [KExprConverterBase] for Long native expressions.
 * */
abstract class KExprLongConverterBase {
    abstract fun findConvertedNative(expr: Long): KExpr<*>?

    abstract fun saveConvertedNative(native: Long, converted: KExpr<*>)

    abstract fun convertNativeExpr(expr: Long): ExprConversionResult

    @JvmField
    val exprStack = LongArrayList()

    fun <S : KSort> convertFromNative(native: Long): KExpr<S> = conversionLoop(
        stack = exprStack,
        native = native,
        stackPush = { stack, element -> stack.add(element) },
        stackPop = { stack -> stack.removeLong(stack.lastIndex) },
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
        expr: Long,
        args: LongArray,
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
        expr: Long,
        args: LongArray,
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
    inline fun <S : KSort, A0 : KSort> Long.convert(
        args: LongArray,
        op: (KExpr<A0>) -> KExpr<S>
    ) = ensureArgsConvertedAndConvert(this, args, expectedSize = 1) { convertedArgs ->
        op(convertedArgs[0] as KExpr<A0>)
    }

    @Suppress("UNCHECKED_CAST")
    inline fun <S : KSort, A0 : KSort, A1 : KSort> Long.convert(
        args: LongArray,
        op: (KExpr<A0>, KExpr<A1>) -> KExpr<S>
    ) = ensureArgsConvertedAndConvert(this, args, expectedSize = 2) { convertedArgs ->
        op(convertedArgs[0] as KExpr<A0>, convertedArgs[1] as KExpr<A1>)
    }

    @Suppress("UNCHECKED_CAST", "MagicNumber")
    inline fun <S : KSort, A0 : KSort, A1 : KSort, A2 : KSort> Long.convert(
        args: LongArray,
        op: (KExpr<A0>, KExpr<A1>, KExpr<A2>) -> KExpr<S>
    ) = ensureArgsConvertedAndConvert(this, args, expectedSize = 3) { convertedArgs ->
        op(
            convertedArgs[0] as KExpr<A0>,
            convertedArgs[1] as KExpr<A1>,
            convertedArgs[2] as KExpr<A2>
        )
    }

    @Suppress("UNCHECKED_CAST", "MagicNumber")
    inline fun <S : KSort, A0 : KSort, A1 : KSort, A2 : KSort, A3 : KSort> Long.convert(
        args: LongArray,
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
    inline fun <S : KSort, A : KSort> Long.convertList(
        args: LongArray,
        op: (List<KExpr<A>>) -> KExpr<S>
    ) = ensureArgsConvertedAndConvert(this, args, expectedSize = args.size) { convertedArgs ->
        op(convertedArgs as List<KExpr<A>>)
    }

    @Suppress("UNCHECKED_CAST")
    inline fun <S : KSort> Long.convertReduced(
        args: LongArray,
        op: (KExpr<S>, KExpr<S>) -> KExpr<S>
    ) = ensureArgsConvertedAndConvert(this, args, expectedSize = args.size) { convertedArgs ->
        (convertedArgs as List<KExpr<S>>).reduce(op)
    }
}
