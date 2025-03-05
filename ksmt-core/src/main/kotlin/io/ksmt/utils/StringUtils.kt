package io.ksmt.utils

import io.ksmt.expr.KInt32NumExpr
import io.ksmt.expr.KInt64NumExpr
import io.ksmt.expr.KIntBigNumExpr
import io.ksmt.expr.KIntNumExpr
import io.ksmt.expr.KInterpretedValue
import io.ksmt.expr.KStringLiteralExpr
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KStringSort

object StringUtils {

    const val STRING_FROM_CODE_UPPER_BOUND: Int = 196607

    @JvmStatic
    fun getStringLen(arg: KStringLiteralExpr): KIntNumExpr = with(arg.ctx) {
        mkIntNum(arg.value.length)
    }

    @JvmStatic
    fun concatStrings(
        lhs: KStringLiteralExpr,
        rhs: KStringLiteralExpr
    ): KStringLiteralExpr = with(lhs.ctx) {
        mkStringLiteral(lhs.value + rhs.value)
    }

    @JvmStatic
    fun isStringSuffix(
        arg0: KStringLiteralExpr,
        arg1: KStringLiteralExpr
    ): KInterpretedValue<KBoolSort> = with(arg0.ctx) {
        mkBool(arg1.value.endsWith(arg0.value)).uncheckedCast()
    }

    @JvmStatic
    fun isStringPrefix(
        arg0: KStringLiteralExpr,
        arg1: KStringLiteralExpr
    ): KInterpretedValue<KBoolSort> = with(arg0.ctx) {
        mkBool(arg1.value.startsWith(arg0.value)).uncheckedCast()
    }

    @JvmStatic
    fun stringLt(
        arg0: KStringLiteralExpr,
        arg1: KStringLiteralExpr
    ): KInterpretedValue<KBoolSort> = with(arg0.ctx) {
        mkBool(arg0.value < arg1.value).uncheckedCast()
    }

    @JvmStatic
    fun stringLe(
        arg0: KStringLiteralExpr,
        arg1: KStringLiteralExpr
    ): KInterpretedValue<KBoolSort> = with(arg0.ctx) {
        mkBool(arg0.value <= arg1.value).uncheckedCast()
    }

    @JvmStatic
    fun stringGt(
        arg0: KStringLiteralExpr,
        arg1: KStringLiteralExpr
    ): KInterpretedValue<KBoolSort> = with(arg0.ctx) {
        mkBool(arg0.value > arg1.value).uncheckedCast()
    }

    @JvmStatic
    fun stringGe(
        arg0: KStringLiteralExpr,
        arg1: KStringLiteralExpr
    ): KInterpretedValue<KBoolSort> = with(arg0.ctx) {
        mkBool(arg0.value >= arg1.value).uncheckedCast()
    }

    @JvmStatic
    fun stringContains(
        arg0: KStringLiteralExpr,
        arg1: KStringLiteralExpr
    ): KInterpretedValue<KBoolSort> = with(arg0.ctx) {
        mkBool(arg0.value.contains(arg1.value)).uncheckedCast()
    }

    @JvmStatic
    fun stringToLowerCase(
        arg: KStringLiteralExpr
    ): KStringLiteralExpr = with(arg.ctx) {
        mkStringLiteral(arg.value.lowercase())
    }

    @JvmStatic
    fun stringToUpperCase(
        arg: KStringLiteralExpr
    ): KStringLiteralExpr = with(arg.ctx) {
        mkStringLiteral(arg.value.uppercase())
    }

    @JvmStatic
    fun stringReverse(
        arg: KStringLiteralExpr
    ): KStringLiteralExpr = with(arg.ctx) {
        mkStringLiteral(arg.value.reversed())
    }

    @JvmStatic
    fun stringIsDigit(
        arg: KStringLiteralExpr
    ): KInterpretedValue<KBoolSort> = with(arg.ctx) {
        mkBool(arg.value.length == 1 && arg.value[0].isDigit()).uncheckedCast()
    }
    
    @JvmStatic
    fun stringToCode(
        arg: KStringLiteralExpr
    ): KIntNumExpr = with(arg.ctx) {
        mkIntNum(arg.value.singleOrNull()?.code ?: -1)
    }

    @JvmStatic
    fun stringToInt(
        arg: KStringLiteralExpr
    ): KIntNumExpr = with(arg.ctx) {
        return if (arg.value.isNotEmpty() && arg.value.all { it.isDigit() }) {
            mkIntNum(arg.value.toLongOrNull() ?: -1)
        } else {
            mkIntNum(-1)
        }
    }

    @JvmStatic
    fun strintReplace(
        arg0: KStringLiteralExpr,
        arg1: KStringLiteralExpr,
        arg2: KStringLiteralExpr
    ): KStringLiteralExpr = with(arg0.ctx) {
        val str = arg0.value
        val search = arg1.value
        val replace = arg2.value
        return mkStringLiteral(if (search.isEmpty()) (replace + str) else str.replaceFirst(search, replace))
    }

    @JvmStatic
    fun strintReplaceAll(
        arg0: KStringLiteralExpr,
        arg1: KStringLiteralExpr,
        arg2: KStringLiteralExpr
    ): KStringLiteralExpr = with(arg0.ctx) {
        val str = arg0.value
        val search = arg1.value
        val replace = arg2.value
        return mkStringLiteral(if (search.isEmpty()) str else str.replace(search, replace))
    }
}
