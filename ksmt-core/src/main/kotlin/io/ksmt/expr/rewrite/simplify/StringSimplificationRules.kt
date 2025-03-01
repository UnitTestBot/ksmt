package io.ksmt.expr.rewrite.simplify

import io.ksmt.expr.KExpr
import io.ksmt.expr.KStringLiteralExpr
import io.ksmt.sort.KStringSort

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
