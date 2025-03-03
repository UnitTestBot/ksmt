package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.*
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
inline fun KContext.simplifyStringBasicConcat(
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
inline fun KContext.simplifyStringNestedConcat(
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
inline fun KContext.simplifyStringLenExpr(
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
inline fun KContext.simplifyStringBasicSuffixOfExpr(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> =
    tryEvalStringLiteralOperation(arg0, arg1, { a0, a1 -> StringUtils.isStringSuffix(a0, a1) }) {
        cont(arg0, arg1)
    }

/** Simplifies string prefix checking expressions
 * (str_prefix_of strConst1 strConst2) ==> boolConst */
inline fun KContext.simplifyStringBasicPrefixOfExpr(
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
inline fun KContext.simplifyStringBasicLtExpr(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> =
    tryEvalStringLiteralOperation(arg0, arg1, { a0, a1 -> StringUtils.stringLt(a0, a1) }) {
        cont(arg0, arg1)
    }

/** Simplifies string "less than or equal" comparison expressions
 * (str_le strConst1 strConst2) ==> boolConst */
inline fun KContext.simplifyStringBasicLeExpr(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> =
    tryEvalStringLiteralOperation(arg0, arg1, { a0, a1 -> StringUtils.stringLe(a0, a1) }) {
        cont(arg0, arg1)
    }

/** Simplifies string "greater than" comparison expressions
 * (str_gt strConst1 strConst2) ==> boolConst */
inline fun KContext.simplifyStringBasicGtExpr(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> =
    tryEvalStringLiteralOperation(arg0, arg1, { a0, a1 -> StringUtils.stringGt(a0, a1) }) {
        cont(arg0, arg1)
    }

/** Simplifies string "greater than or equal" comparison expressions
 * (str_ge strConst1 strConst2) ==> boolConst */
inline fun KContext.simplifyStringBasicGeExpr(
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
inline fun KContext.simplifyStringBasicContainsExpr(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> =
    tryEvalStringLiteralOperation(arg0, arg1, { a0, a1 -> StringUtils.stringContains(a0, a1) }) {
        cont(arg0, arg1)
    }

/*
* String to lower/upper case expression simplifications
* */

/** Converting all letters of a string constant to lowercase. */
inline fun KContext.simplifyStringBasicToLowerExpr(
    arg: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>) -> KExpr<KStringSort>
): KExpr<KStringSort> =
    tryEvalStringLiteralOperation(arg, { a -> StringUtils.stringToLowerCase(a) }) {
        cont(arg)
    }

/** Converting all letters of a string constant to uppercase. */
inline fun KContext.simplifyStringBasicToUpperExpr(
    arg: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>) -> KExpr<KStringSort>
): KExpr<KStringSort> =
    tryEvalStringLiteralOperation(arg, { a -> StringUtils.stringToUpperCase(a) }) {
        cont(arg)
    }

/** Reverses a string constan.t */
inline fun KContext.simplifyStringBasicReverseExpr(
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
inline fun KContext.simplifyStringIsDigitExprBasic(
    arg: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> =
    tryEvalStringLiteralOperation(arg, { a -> StringUtils.stringIsDigit(a) }) {
        cont(arg)
    }

/** Eval constants: if string literal consist of one character - return its code, otherwise return -1. */
inline fun KContext.simplifyStringToCodeExprBasic(
    arg: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>) -> KExpr<KIntSort>
): KExpr<KIntSort> =
    tryEvalStringLiteralOperation(arg, { a -> StringUtils.stringToCode(a) }) {
        cont(arg)
    }

/** Eval constants: if int constant is in the range [0; STRING_FROM_CODE_UPPER_BOUND], then
 * return code point of constant, otherwise return empty string. */
inline fun KContext.simplifyStringFromCodeExprBasic(
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
inline fun KContext.simplifyStringToIntExprBasic(
    arg: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>) -> KExpr<KIntSort>
): KExpr<KIntSort> =
    tryEvalStringLiteralOperation(arg, { a -> StringUtils.stringToInt(a) }) {
        cont(arg)
    }

/** Eval constants: if the integer is non-negative, return its string representation;
 * otherwise, return an empty string. */
inline fun KContext.simplifyStringFromIntExprBasic(
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
