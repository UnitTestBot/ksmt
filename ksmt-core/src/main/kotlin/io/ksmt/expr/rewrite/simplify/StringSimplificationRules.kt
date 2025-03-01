package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.expr.KStringConcatExpr
import io.ksmt.expr.KStringLiteralExpr
import io.ksmt.sort.KStringSort
import io.ksmt.utils.StringUtils

inline fun KContext.simplifyStringBasicConcat(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KStringSort>
): KExpr<KStringSort> =
    tryEvalStringLiteralOperation(arg0, arg1, { a1, a2 -> StringUtils.concatStrings(a1, a2) }) {
        cont(arg0, arg1)
    }

inline fun KContext.simplifyStringNestedConcat(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    rewriteStringConcatExpr: KContext.(KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KStringSort>,
    cont: (KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<KStringSort>
): KExpr<KStringSort> {
    if (arg0 is KStringLiteralExpr && arg1 is KStringConcatExpr) {
        val arg1Left = arg1.arg0
        if (arg1Left is KStringLiteralExpr) {
            return rewriteStringConcatExpr(StringUtils.concatStrings(arg0, arg1Left), arg1.arg1)
        }
    }

    if (arg1 is KStringLiteralExpr && arg0 is KStringConcatExpr) {
        val arg0Right = arg0.arg1
        if (arg0Right is KStringLiteralExpr) {
            return rewriteStringConcatExpr(arg0.arg0, StringUtils.concatStrings(arg0Right, arg1))
        }
    }

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

inline fun tryEvalStringLiteralOperation(
    lhs: KExpr<KStringSort>,
    rhs: KExpr<KStringSort>,
    operation: (KStringLiteralExpr, KStringLiteralExpr) -> KStringLiteralExpr,
    cont: () -> KExpr<KStringSort>
): KExpr<KStringSort> = if (lhs is KStringLiteralExpr && rhs is KStringLiteralExpr) {
    operation(lhs, rhs)
} else {
    cont()
}
