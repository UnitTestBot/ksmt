package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.expr.KAndBinaryExpr
import org.ksmt.expr.KAndNaryExpr
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBoolSort

internal fun KContext.mkAndAuxExpr(args: List<KExpr<KBoolSort>>): KExpr<KBoolSort> =
    if (args.size == 2) {
        KAndBinaryExpr(this, args.first(), args.last())
    } else {
        KAndNaryExpr(this, args)
    }
