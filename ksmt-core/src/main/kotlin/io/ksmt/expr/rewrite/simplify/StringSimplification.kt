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
): KExpr<KBoolSort> = simplifyStringBasicLtExpr(arg0, arg1, ::mkStringLtNoSimplify)

fun KContext.simplifyStringLe(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>
): KExpr<KBoolSort> = simplifyStringBasicLeExpr(arg0, arg1, ::mkStringLeNoSimplify)

fun KContext.simplifyStringGt(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>
): KExpr<KBoolSort> = simplifyStringBasicGtExpr(arg0, arg1, ::mkStringGtNoSimplify)

fun KContext.simplifyStringGe(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>
): KExpr<KBoolSort> = simplifyStringBasicGeExpr(arg0, arg1, ::mkStringGeNoSimplify)

fun KContext.simplifyStringContains(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>
): KExpr<KBoolSort> = simplifyStringBasicContainsExpr(arg0, arg1, ::mkStringContainsNoSimplify)

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
): KExpr<KStringSort> = simplifyStringBasicToLowerExpr(arg, ::mkStringToLowerNoSimplify)

fun KContext.simplifyStringToUpper(
    arg: KExpr<KStringSort>
): KExpr<KStringSort> = simplifyStringBasicToUpperExpr(arg, ::mkStringToUpperNoSimplify)

fun KContext.simplifyStringReverse(
    arg: KExpr<KStringSort>
): KExpr<KStringSort> = simplifyStringBasicReverseExpr(arg, ::mkStringReverseNoSimplify)

fun KContext.simplifyStringIsDigit(
    arg: KExpr<KStringSort>
): KExpr<KBoolSort> = simplifyStringIsDigitExprBasic(arg, ::mkStringIsDigitNoSimplify)

fun KContext.simplifyStringToCode(
    arg: KExpr<KStringSort>
): KExpr<KIntSort> = simplifyStringToCodeExprBasic(arg, ::mkStringToCodeNoSimplify)

fun KContext.simplifyStringFromCode(
    arg: KExpr<KIntSort>
): KExpr<KStringSort> = simplifyStringFromCodeExprBasic(arg, ::mkStringFromCodeNoSimplify)

fun KContext.simplifyStringToInt(
    arg: KExpr<KStringSort>
): KExpr<KIntSort> = simplifyStringToIntExprBasic(arg, ::mkStringToIntNoSimplify)

fun KContext.simplifyStringFromInt(
    arg: KExpr<KIntSort>
): KExpr<KStringSort> = simplifyStringFromIntExprBasic(arg, ::mkStringFromIntNoSimplify)
