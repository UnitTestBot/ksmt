package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.expr.KInterpretedValue
import io.ksmt.expr.KStringConcatExpr
import io.ksmt.expr.KStringLiteralExpr
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KStringSort
import io.ksmt.utils.StringUtils

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

inline fun KContext.simplifyStringLenExpr(
    arg: KExpr<KStringSort>,
    cont: (KExpr<KStringSort>) -> KExpr<KIntSort>
): KExpr<KIntSort> =
    tryEvalStringLiteralOperation(arg, { literalArg ->
        mkIntNum(literalArg.value.length)
    }) {
        cont(arg)
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

inline fun tryEvalStringLiteralOperation(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>,
    operation: (KStringLiteralExpr, KStringLiteralExpr) -> KStringLiteralExpr,
    cont: () -> KExpr<KStringSort>
): KExpr<KStringSort> = if (arg0 is KStringLiteralExpr && arg1 is KStringLiteralExpr) {
    operation(arg0, arg1)
} else {
    cont()
}
