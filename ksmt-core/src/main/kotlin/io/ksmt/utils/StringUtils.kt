package io.ksmt.utils

import io.ksmt.expr.KInterpretedValue
import io.ksmt.expr.KStringLiteralExpr
import io.ksmt.sort.KBoolSort

object StringUtils {

    @JvmStatic
    fun concatStrings(lhs: KStringLiteralExpr, rhs: KStringLiteralExpr): KStringLiteralExpr = with (lhs.ctx) {
        mkStringLiteral(lhs.value + rhs.value)
    }

    @JvmStatic
    fun isStringSuffix(arg0: KStringLiteralExpr, arg1: KStringLiteralExpr): KInterpretedValue<KBoolSort> = with (arg0.ctx) {
        mkBool(arg1.value.endsWith(arg0.value)).uncheckedCast()
    }

    @JvmStatic
    fun isStringPrefix(arg0: KStringLiteralExpr, arg1: KStringLiteralExpr): KInterpretedValue<KBoolSort> = with (arg0.ctx) {
        mkBool(arg1.value.startsWith(arg0.value)).uncheckedCast()
    }
}
