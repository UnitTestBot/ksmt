package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KAndBinaryExpr
import io.ksmt.expr.KAndNaryExpr
import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort

internal fun KContext.mkAndAuxExpr(args: List<KExpr<KBoolSort>>): KExpr<KBoolSort> =
    if (args.size == 2) {
        KAndBinaryExpr(this, args.first(), args.last())
    } else {
        KAndNaryExpr(this, args)
    }
