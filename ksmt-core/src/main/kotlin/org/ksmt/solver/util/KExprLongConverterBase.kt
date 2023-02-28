package org.ksmt.solver.util

import org.ksmt.expr.KExpr
import org.ksmt.solver.util.KExprConverterBase.Companion.ExprConversionResult
import org.ksmt.solver.util.KExprConverterBase.Companion.argumentsConversionRequired
import org.ksmt.sort.KSort

abstract class KExprLongConverterBase {
    abstract fun findConvertedNative(expr: Long): KExpr<*>?

    abstract fun saveConvertedNative(native: Long, converted: KExpr<*>)

    abstract fun convertNativeExpr(expr: Long): ExprConversionResult

    @JvmField
    val exprStack = arrayListOf<Long>()

    fun <S : KSort> convertFromNative(native: Long): KExpr<S> {
        exprStack.add(native)

        while (exprStack.isNotEmpty()) {
            val expr = exprStack.removeLast()

            if (findConvertedNative(expr) != null) continue

            val converted = convertNativeExpr(expr)

            if (!converted.isArgumentsConversionRequired) {
                saveConvertedNative(expr, converted.convertedExpr)
            }
        }

        @Suppress("UNCHECKED_CAST")
        return findConvertedNative(native) as? KExpr<S> ?: error("expr is not properly converted")
    }

    fun checkArgumentsSizeMatchExpected(args: LongArray, expectedSize: Int) {
        check(args.size == expectedSize) {
            "arguments size mismatch: expected $expectedSize, actual ${args.size}"
        }
    }

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
        checkArgumentsSizeMatchExpected(args, expectedSize)

        val convertedArgs = mutableListOf<KExpr<*>>()
        var hasNotConvertedArgs = false

        for (arg in args) {
            val converted = findConvertedNative(arg)

            if (converted != null) {
                convertedArgs.add(converted)
                continue
            }

            if (!hasNotConvertedArgs) {
                hasNotConvertedArgs = true
                exprStack.add(expr)
            }

            exprStack.add(arg)
        }

        if (hasNotConvertedArgs) return argumentsConversionRequired

        val convertedExpr = converter(convertedArgs)
        return ExprConversionResult(convertedExpr)
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
