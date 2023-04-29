package io.ksmt.solver.util

import io.ksmt.expr.KExpr
import io.ksmt.solver.util.KExprConverterUtils.checkArgumentsSizeMatchExpected
import io.ksmt.sort.KSort

@Suppress("LongParameterList")
inline fun <S : KSort, T, Stack> conversionLoop(
    stack: Stack,
    native: T,
    stackPush: (Stack, T) -> Unit,
    stackPop: (Stack) -> T,
    stackIsNotEmpty: (Stack) -> Boolean,
    convertNative: (T) -> ExprConversionResult,
    findConverted: (T) -> KExpr<*>?,
    saveConverted: (T, KExpr<*>) -> Unit
): KExpr<S> {
    stackPush(stack, native)

    while (stackIsNotEmpty(stack)) {
        val expr = stackPop(stack)

        if (findConverted(expr) != null) continue

        val converted = convertNative(expr)

        if (!converted.isArgumentsConversionRequired) {
            saveConverted(expr, converted.convertedExpr)
        }
    }

    @Suppress("UNCHECKED_CAST")
    return findConverted(native) as? KExpr<S> ?: error("expr is not properly converted")
}

/**
 * Ensure all expression arguments are already converted.
 * Return converted arguments or null if not all arguments converted.
 * */
@Suppress("LongParameterList")
inline fun <T, TArray, Stack> ensureArgsConvertedAndConvert(
    stack: Stack,
    expr: T,
    args: TArray,
    expectedSize: Int,
    stackPush: (Stack, T) -> Unit,
    arraySize: (TArray) -> Int,
    arrayGet: (TArray, Int) -> T,
    findConverted: (T) -> KExpr<*>?,
): List<KExpr<*>>? {
    val argsSize = arraySize(args)
    checkArgumentsSizeMatchExpected(argsSize, expectedSize)

    val convertedArgs = mutableListOf<KExpr<*>>()
    var hasNotConvertedArgs = false

    for (i in 0 until argsSize) {
        val arg = arrayGet(args, i)
        val converted = findConverted(arg)

        if (converted != null) {
            convertedArgs.add(converted)
            continue
        }

        if (!hasNotConvertedArgs) {
            hasNotConvertedArgs = true
            stackPush(stack, expr)
        }

        stackPush(stack, arg)
    }

    return if (hasNotConvertedArgs) null else convertedArgs
}

@JvmInline
value class ExprConversionResult(private val expr: KExpr<*>?) {
    val isArgumentsConversionRequired: Boolean
        get() = expr == null

    val convertedExpr: KExpr<*>
        get() = expr ?: error("expr is not converted")
}

object KExprConverterUtils {
    @JvmStatic
    val argumentsConversionRequired = ExprConversionResult(null)

    @JvmStatic
    fun checkArgumentsSizeMatchExpected(argumentsSize: Int, expectedSize: Int) {
        check(argumentsSize == expectedSize) {
            "arguments size mismatch: expected $expectedSize, actual $argumentsSize"
        }
    }
}
