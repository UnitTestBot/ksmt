package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRegexSort
import io.ksmt.sort.KStringSort

fun KContext.simplifyStringConcat(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>
): KExpr<KStringSort> = simplifyStringBasicConcat(arg0, arg1) {arg2, arg3 ->
    simplifyStringNestedConcat(arg2, arg3, KContext::simplifyStringConcat, ::mkStringConcatNoSimplify)
}

fun KContext.simplifyStringLen(
    arg: KExpr<KStringSort>
): KExpr<KIntSort> = simplifyStringLenExpr(arg, ::mkStringLenNoSimplify)

fun KContext.simplifyStringToRegex(
    arg: KExpr<KStringSort>
): KExpr<KRegexSort> = mkStringToRegexNoSimplify(arg) // Temporarily

fun KContext.simplifyStringInRegex(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KRegexSort>
): KExpr<KBoolSort> = mkStringInRegexNoSimplify(arg0, arg1) // Temporarily

fun KContext.simplifyStringSuffixOf(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>
): KExpr<KBoolSort> = simplifyStringBasicSuffixOfExpr(arg0, arg1, ::mkStringSuffixOfNoSimplify)

fun KContext.simplifyStringPrefixOf(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>
): KExpr<KBoolSort> = simplifyStringBasicPrefixOfExpr(arg0, arg1, ::mkStringPrefixOfNoSimplify)

fun KContext.simplifyStringLt(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>
): KExpr<KBoolSort> = mkStringLtNoSimplify(arg0, arg1) // Temporarily

fun KContext.simplifyStringLe(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>
): KExpr<KBoolSort> = mkStringLeNoSimplify(arg0, arg1) // Temporarily

fun KContext.simplifyStringGt(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>
): KExpr<KBoolSort> = mkStringGtNoSimplify(arg0, arg1) // Temporarily

fun KContext.simplifyStringGe(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>
): KExpr<KBoolSort> = mkStringGeNoSimplify(arg0, arg1) // Temporarily

fun KContext.simplifyStringContains(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>
): KExpr<KBoolSort> = mkStringContainsNoSimplify(arg0, arg1) // Temporarily

fun KContext.simplifyStringSingletonSub(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KIntSort>
): KExpr<KStringSort> = mkStringSingletonSubNoSimplify(arg0, arg1) // Temporarily

fun KContext.simplifyStringSub(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KIntSort>,
    arg2: KExpr<KIntSort>
): KExpr<KStringSort> = mkStringSubNoSimplify(arg0, arg1, arg2) // Temporarily

fun KContext.simplifyStringIndexOf(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    arg2: KExpr<KIntSort>
): KExpr<KIntSort> = mkStringIndexOfNoSimplify(arg0, arg1, arg2) // Temporarily

fun KContext.simplifyStringIndexOfRegex(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KRegexSort>,
    arg2: KExpr<KIntSort>
): KExpr<KIntSort> = mkStringIndexOfRegexNoSimplify(arg0, arg1, arg2) // Temporarily

fun KContext.simplifyStringReplace(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    arg2: KExpr<KStringSort>
): KExpr<KStringSort> = mkStringReplaceNoSimplify(arg0, arg1, arg2) // Temporarily

fun KContext.simplifyStringReplaceAll(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    arg2: KExpr<KStringSort>
): KExpr<KStringSort> = mkStringReplaceAllNoSimplify(arg0, arg1, arg2) // Temporarily

fun KContext.simplifyStringReplaceWithRegex(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KRegexSort>,
    arg2: KExpr<KStringSort>
): KExpr<KStringSort> = mkStringReplaceWithRegexNoSimplify(arg0, arg1, arg2) // Temporarily

fun KContext.simplifyStringReplaceAllWithRegex(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KRegexSort>,
    arg2: KExpr<KStringSort>
): KExpr<KStringSort> = mkStringReplaceAllWithRegexNoSimplify(arg0, arg1, arg2) // Temporarily

fun KContext.simplifyStringToLower(
    arg: KExpr<KStringSort>
): KExpr<KStringSort> = mkStringToLowerNoSimplify(arg) // Temporarily

fun KContext.simplifyStringToUpper(
    arg: KExpr<KStringSort>
): KExpr<KStringSort> = mkStringToUpperNoSimplify(arg) // Temporarily

fun KContext.simplifyStringReverse(
    arg: KExpr<KStringSort>
): KExpr<KStringSort> = mkStringReverseNoSimplify(arg) // Temporarily

fun KContext.simplifyStringIsDigit(
    arg: KExpr<KStringSort>
): KExpr<KBoolSort> = mkStringIsDigitNoSimplify(arg) // Temporarily

fun KContext.simplifyStringToCode(
    arg: KExpr<KStringSort>
): KExpr<KIntSort> = mkStringToCodeNoSimplify(arg) // Temporarily

fun KContext.simplifyStringFromCode(
    arg: KExpr<KIntSort>
): KExpr<KStringSort> = mkStringFromCodeNoSimplify(arg) // Temporarily

fun KContext.simplifyStringToInt(
    arg: KExpr<KStringSort>
): KExpr<KIntSort> = mkStringToIntNoSimplify(arg) // Temporarily

fun KContext.simplifyStringFromInt(
    arg: KExpr<KIntSort>
): KExpr<KStringSort> = mkStringFromIntNoSimplify(arg) // Temporarily
