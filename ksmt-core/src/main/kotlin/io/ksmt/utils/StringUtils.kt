package io.ksmt.utils

import io.ksmt.expr.KInt32NumExpr
import io.ksmt.expr.KInt64NumExpr
import io.ksmt.expr.KIntBigNumExpr
import io.ksmt.expr.KIntNumExpr
import io.ksmt.expr.KInterpretedValue
import io.ksmt.expr.KStringLiteralExpr
import io.ksmt.sort.KBoolSort

object StringUtils {

    @JvmStatic
    fun getStringLen(arg: KStringLiteralExpr): KIntNumExpr = with (arg.ctx) {
        mkIntNum(arg.value.length)
    }

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

    @JvmStatic
    fun stringLt(arg0: KStringLiteralExpr, arg1: KStringLiteralExpr): KInterpretedValue<KBoolSort> = with (arg0.ctx) {
        mkBool(arg0.value < arg1.value).uncheckedCast()
    }

    @JvmStatic
    fun stringLe(arg0: KStringLiteralExpr, arg1: KStringLiteralExpr): KInterpretedValue<KBoolSort> = with (arg0.ctx) {
        mkBool(arg0.value <= arg1.value).uncheckedCast()
    }

    @JvmStatic
    fun stringGt(arg0: KStringLiteralExpr, arg1: KStringLiteralExpr): KInterpretedValue<KBoolSort> = with (arg0.ctx) {
        mkBool(arg0.value > arg1.value).uncheckedCast()
    }

    @JvmStatic
    fun stringGe(arg0: KStringLiteralExpr, arg1: KStringLiteralExpr): KInterpretedValue<KBoolSort> = with (arg0.ctx) {
        mkBool(arg0.value >= arg1.value).uncheckedCast()
    }

    @JvmStatic
    fun stringContains(arg0: KStringLiteralExpr, arg1: KStringLiteralExpr): KInterpretedValue<KBoolSort> = with (arg0.ctx) {
        mkBool(arg0.value.contains(arg1.value)).uncheckedCast()
    }

    @JvmStatic
    fun stringToLowerCase(arg: KStringLiteralExpr): KStringLiteralExpr = with (arg.ctx) {
        mkStringLiteral(arg.value.lowercase())
    }

    @JvmStatic
    fun stringToUpperCase(arg: KStringLiteralExpr): KStringLiteralExpr = with (arg.ctx) {
        mkStringLiteral(arg.value.uppercase())
    }

    @JvmStatic
    fun stringReverse(arg: KStringLiteralExpr): KStringLiteralExpr = with (arg.ctx) {
        mkStringLiteral(arg.value.reversed())
    }

}
