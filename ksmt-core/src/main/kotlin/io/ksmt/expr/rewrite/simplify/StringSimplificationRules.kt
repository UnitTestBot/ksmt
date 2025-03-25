package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KStringConcatExpr
import io.ksmt.expr.KStringLiteralExpr
import io.ksmt.expr.KIntNumExpr
import io.ksmt.expr.KInt32NumExpr
import io.ksmt.expr.KInt64NumExpr
import io.ksmt.expr.KIntBigNumExpr
import io.ksmt.expr.KInterpretedValue
import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KStringSort
import io.ksmt.utils.StringUtils
import io.ksmt.utils.StringUtils.STRING_FROM_CODE_UPPER_BOUND

/*
* String concatenation simplification
* */

/**
 * Eval constants.
 * (concat const1 const2) ==> (const3) */
inline fun KContext.simplifyStringConcatBasic(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KStringSort>
): KExpr<KStringSort> =
    tryEvalStringLiteralOperation(arg0, arg1, { a1, a2 -> StringUtils.concatStrings(a1, a2) }) {
        cont(arg0, arg1)
    }

/**
 * ((concat a const1) const2) ==> (concat a (concat const1 const2))
 * ((concat const1 (concat const2 a)) => (concat (concat const1 const2) a)
 * ((concat (concat a const1) (concat const2 b)) ==> (concat a (concat (concat const1 const2) b))
 */
inline fun KContext.simplifyStringConcatNested(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    rewriteStringConcatExpr: KContext.(KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KStringSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KStringSort>
): KExpr<KStringSort> {
    // ((concat a const1) const2) ==> (concat a (concat const1 const2))
    if (arg0 is KStringLiteralExpr && arg1 is KStringConcatExpr) {
        val arg1Left = arg1.arg0
        if (arg1Left is KStringLiteralExpr) {
            return rewriteStringConcatExpr(StringUtils.concatStrings(arg0, arg1Left), arg1.arg1)
        }
    }
    // ((concat const1 (concat const2 a)) => (concat (concat const1 const2) a)
    if (arg1 is KStringLiteralExpr && arg0 is KStringConcatExpr) {
        val arg0Right = arg0.arg1
        if (arg0Right is KStringLiteralExpr) {
            return rewriteStringConcatExpr(arg0.arg0, StringUtils.concatStrings(arg0Right, arg1))
        }
    }
    // ((concat (concat a const1) (concat const2 b)) ==> (concat a (concat (concat const1 const2) b))
    if (arg0 is KStringConcatExpr && arg1 is KStringConcatExpr) {
        val arg0Right = arg0.arg1
        val arg1Left = arg1.arg0
        if (arg0Right is KStringLiteralExpr && arg1Left is KStringLiteralExpr) {
            return rewriteStringConcatExpr(
                arg0.arg0,
                mkStringConcat(StringUtils.concatStrings(arg0Right, arg1Left), arg1.arg1)
            )
        }
    }

    return cont(arg0, arg1)
}

/*
* String length expression simplification
* */

/**
 * Eval length of string constant. */
inline fun KContext.simplifyStringLenBasic(
    arg: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>) -> KExpr<KIntSort>
): KExpr<KIntSort> =
    tryEvalStringLiteralOperation(arg, {
        a -> StringUtils.getStringLen(a)
    }) {
        cont(arg)
    }

/*
* SuffixOf and PrefixOf expression simplifications
* */

/** Simplifies string suffix checking expressions
 * (str_suffix_of strConst1 strConst2) ==> boolConst */
inline fun KContext.simplifyStringSuffixOfBasic(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> =
    tryEvalStringLiteralOperation(arg0, arg1, { a0, a1 -> StringUtils.isStringSuffix(a0, a1) }) {
        cont(arg0, arg1)
    }

/** Simplifies string prefix checking expressions
 * (str_prefix_of strConst1 strConst2) ==> boolConst */
inline fun KContext.simplifyStringPrefixOfBasic(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> =
    tryEvalStringLiteralOperation(arg0, arg1, { a0, a1 -> StringUtils.isStringPrefix(a0, a1) }) {
        cont(arg0, arg1)
    }

/*
* String comparison expression simplifications
* */

/** Simplifies string "less than" comparison expressions
 * (str_lt strConst1 strConst2) ==> boolConst */
inline fun KContext.simplifyStringLtBasic(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> =
    tryEvalStringLiteralOperation(arg0, arg1, { a0, a1 -> StringUtils.stringLt(a0, a1) }) {
        cont(arg0, arg1)
    }

/** Simplifies string "less than or equal" comparison expressions
 * (str_le strConst1 strConst2) ==> boolConst */
inline fun KContext.simplifyStringLeBasic(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> =
    tryEvalStringLiteralOperation(arg0, arg1, { a0, a1 -> StringUtils.stringLe(a0, a1) }) {
        cont(arg0, arg1)
    }

/** Simplifies string "greater than" comparison expressions
 * (str_gt strConst1 strConst2) ==> boolConst */
inline fun KContext.simplifyStringGtBasic(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> =
    tryEvalStringLiteralOperation(arg0, arg1, { a0, a1 -> StringUtils.stringGt(a0, a1) }) {
        cont(arg0, arg1)
    }

/** Simplifies string "greater than or equal" comparison expressions
 * (str_ge strConst1 strConst2) ==> boolConst */
inline fun KContext.simplifyStringGeBasic(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> =
    tryEvalStringLiteralOperation(arg0, arg1, { a0, a1 -> StringUtils.stringGe(a0, a1) }) {
        cont(arg0, arg1)
    }

/*
* String contains expression simplifications
* */

/** Basic simplify string contains expression
 * (str_contains strConst1 strConst2) ==> boolConst */
inline fun KContext.simplifyStringContainsBasic(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> =
    tryEvalStringLiteralOperation(arg0, arg1, { a0, a1 -> StringUtils.stringContains(a0, a1) }) {
        cont(arg0, arg1)
    }

/*
* Substring expressions simplifications
* */

/** Eval constants. */
inline fun KContext.simplifyStringSingletonSubBasic(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KIntSort>,
    cont: (KExpr<KStringSort>, KExpr<KIntSort>) -> KExpr<KStringSort>
): KExpr<KStringSort> {
    return if (arg0 is KStringLiteralExpr && arg1 is KIntNumExpr) {
        val str = arg0.value
        val pos = when (arg1) {
            is KInt32NumExpr -> arg1.value.toLong()
            is KInt64NumExpr -> arg1.value
            is KIntBigNumExpr -> arg1.value.toLong()
            else -> return cont(arg0, arg1)
        }
        val result = if (pos <= Int.MAX_VALUE && pos >= 0 && pos < str.length) {
            str[pos.toInt()].toString()
        } else {
            ""
        }
        mkStringLiteral(result)
    } else {
        cont(arg0, arg1)
    }
}

/** Eval constants. */
inline fun KContext.simplifyStringSubBasic(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KIntSort>,
    arg2: KExpr<KIntSort>,
    cont: (KExpr<KStringSort>, KExpr<KIntSort>, KExpr<KIntSort>) -> KExpr<KStringSort>
): KExpr<KStringSort> {
    return if (arg0 is KStringLiteralExpr && arg1 is KIntNumExpr && arg2 is KIntNumExpr) {
        val str = arg0.value
        val startPos = arg1.toLongValue()
        val length = arg2.toLongValue()
        val result = if (startPos in 0..Int.MAX_VALUE && length in 0..Int.MAX_VALUE && startPos < str.length) {
            val endPos = minOf(startPos + length, str.length.toLong()).toInt()
            str.substring(startPos.toInt(), endPos)
        } else {
            ""
        }
        mkStringLiteral(result)
    } else {
        cont(arg0, arg1, arg2)
    }
}

/** Eval constants. */
@Suppress("NestedBlockDepth")
inline fun KContext.simplifyStringIndexOfBasic(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    arg2: KExpr<KIntSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>, KExpr<KIntSort>) -> KExpr<KIntSort>
): KExpr<KIntSort> = with(arg0.ctx) {

    if (arg0 is KStringLiteralExpr && arg1 is KStringLiteralExpr && arg2 is KIntNumExpr) {
        val str = arg0.value
        val search = arg1.value
        val startPosLong = arg2.toLongValue()
        val startPos: Int
        if (startPosLong <= Int.MAX_VALUE) {
            startPos = startPosLong.toInt()
        } else {
            return cont(arg0, arg1, arg2)
        }

        val result = if (startPos >= 0 && startPos <= str.length) {
            if (search.isEmpty()) {
                startPos
            } else {
                val index = str.indexOf(search, startPos)
                if (index >= 0) index else -1
            }
        } else {
            -1
        }
        return mkIntNum(result)
    } else {
        return cont(arg0, arg1, arg2)
    }
}

/*
* String replace expressions simplifications
* */

inline fun KContext.simplifyStringReplaceBasic(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    arg2: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KStringSort>
): KExpr<KStringSort> =
    tryEvalStringLiteralOperation(arg0, arg1, arg2, { a0, a1, a2 -> StringUtils.strintReplace(a0, a1, a2) }) {
        cont(arg0, arg1, arg2)
    }

inline fun KContext.simplifyStringReplaceAllBasic(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    arg2: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KStringSort>
): KExpr<KStringSort> =
    tryEvalStringLiteralOperation(arg0, arg1, arg2, { a0, a1, a2 -> StringUtils.strintReplaceAll(a0, a1, a2) }) {
        cont(arg0, arg1, arg2)
    }

/*
* String to lower/upper case expression simplifications
* */

/** Converting all letters of a string constant to lowercase. */
inline fun KContext.simplifyStringToLowerBasic(
    arg: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>) -> KExpr<KStringSort>
): KExpr<KStringSort> =
    tryEvalStringLiteralOperation(arg, { a -> StringUtils.stringToLowerCase(a) }) {
        cont(arg)
    }

/** Converting all letters of a string constant to uppercase. */
inline fun KContext.simplifyStringToUpperBasic(
    arg: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>) -> KExpr<KStringSort>
): KExpr<KStringSort> =
    tryEvalStringLiteralOperation(arg, { a -> StringUtils.stringToUpperCase(a) }) {
        cont(arg)
    }

/** Reverses a string constan.t */
inline fun KContext.simplifyStringReverseBasic(
    arg: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>) -> KExpr<KStringSort>
): KExpr<KStringSort> =
    tryEvalStringLiteralOperation(arg, { a -> StringUtils.stringReverse(a) }) {
        cont(arg)
    }

/*
* Mapping between strings and integers simplifications
* */

/** Eval constants: if string literal consist of one digit - return true, otherwise false. */
inline fun KContext.simplifyStringIsDigitBasic(
    arg: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> =
    tryEvalStringLiteralOperation(arg, { a -> StringUtils.stringIsDigit(a) }) {
        cont(arg)
    }

/** Eval constants: if string literal consist of one character - return its code, otherwise return -1. */
inline fun KContext.simplifyStringToCodeBasic(
    arg: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>) -> KExpr<KIntSort>
): KExpr<KIntSort> =
    tryEvalStringLiteralOperation(arg, { a -> StringUtils.stringToCode(a) }) {
        cont(arg)
    }

/** Eval constants: if int constant is in the range [0; STRING_FROM_CODE_UPPER_BOUND], then
 * return code point of constant, otherwise return empty string. */
inline fun KContext.simplifyStringFromCodeBasic(
    arg: KExpr<KIntSort>,
    cont: (KExpr<KIntSort>) -> KExpr<KStringSort>
): KExpr<KStringSort> {
    val value = when (arg) {
        is KInt32NumExpr -> arg.value.toLong()
        is KInt64NumExpr -> arg.value
        is KIntBigNumExpr -> arg.value.toLong()
        else -> return cont(arg)
    }

    return if (value in 0..STRING_FROM_CODE_UPPER_BOUND) {
        mkStringLiteral(value.toInt().toChar().toString())
    } else {
        mkStringLiteral("")
    }
}

/** Eval constants: if string literal consist of digits, then
 * return the positive integer denoted by literal;
 * otherwise, return -1. */
inline fun KContext.simplifyStringToIntBasic(
    arg: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>) -> KExpr<KIntSort>
): KExpr<KIntSort> =
    tryEvalStringLiteralOperation(arg, { a -> StringUtils.stringToInt(a) }) {
        cont(arg)
    }

/** Eval constants: if the integer is non-negative, return its string representation;
 * otherwise, return an empty string. */
inline fun KContext.simplifyStringFromIntBasic(
    arg: KExpr<KIntSort>,
    cont: (KExpr<KIntSort>) -> KExpr<KStringSort>
): KExpr<KStringSort> {
    val intValue = when (arg) {
        is KInt32NumExpr -> arg.value.toLong()
        is KInt64NumExpr -> arg.value
        is KIntBigNumExpr -> arg.value.toLong()
        else -> return cont(arg)
    }
    return mkStringLiteral(if (intValue >= 0) intValue.toString() else "")
}

inline fun <K : KSort> tryEvalStringLiteralOperation(
    arg: KExpr<KStringSort>,
    operation: (KStringLiteralExpr) -> KInterpretedValue<K>,
    cont: () -> KExpr<K>
): KExpr<K> = if (arg is KStringLiteralExpr) {
    operation(arg)
} else {
    cont()
}

inline fun <K: KSort> tryEvalStringLiteralOperation(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    operation: (KStringLiteralExpr, KStringLiteralExpr) -> KInterpretedValue<K>,
    cont: () -> KExpr<K>
): KExpr<K> = if (arg0 is KStringLiteralExpr && arg1 is KStringLiteralExpr) {
    operation(arg0, arg1)
} else {
    cont()
}

inline fun <K: KSort> tryEvalStringLiteralOperation(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    arg2: KExpr<KStringSort>,
    operation: (KStringLiteralExpr, KStringLiteralExpr, KStringLiteralExpr) -> KInterpretedValue<K>,
    cont: () -> KExpr<K>
): KExpr<K> = if (arg0 is KStringLiteralExpr && arg1 is KStringLiteralExpr && arg2 is KStringLiteralExpr) {
    operation(arg0, arg1, arg2)
} else {
    cont()
}

fun KIntNumExpr.toLongValue(): Long = when (this) {
    is KInt32NumExpr -> value.toLong()
    is KInt64NumExpr -> value
    is KIntBigNumExpr -> value.toLong()
    else -> throw IllegalArgumentException("Unsupported KIntNumExpr type: ${this::class}")
}
