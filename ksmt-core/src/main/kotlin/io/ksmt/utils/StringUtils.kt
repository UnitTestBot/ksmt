package io.ksmt.utils

import io.ksmt.expr.KStringLiteralExpr

object StringUtils {

    @JvmStatic
    fun concatStrings(lhs: KStringLiteralExpr, rhs: KStringLiteralExpr): KStringLiteralExpr = with (lhs.ctx) {
        mkStringLiteral(lhs.value + rhs.value)
    }
}
