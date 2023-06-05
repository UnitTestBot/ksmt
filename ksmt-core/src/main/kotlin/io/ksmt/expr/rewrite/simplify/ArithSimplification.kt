package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.sort.KArithSort
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort

fun <T : KArithSort> KContext.simplifyArithUnaryMinus(arg: KExpr<T>): KExpr<T> =
    simplifyArithUnaryMinusLight(arg, ::mkArithUnaryMinusNoSimplify)

fun <T : KArithSort> KContext.simplifyArithAdd(args: List<KExpr<T>>): KExpr<T> =
    simplifyArithAddLight(args, ::mkArithAddNoSimplify)

fun <T : KArithSort> KContext.simplifyArithSub(args: List<KExpr<T>>): KExpr<T> =
    rewriteArithSub(args, KContext::simplifyArithAdd, KContext::simplifyArithUnaryMinus)

fun <T : KArithSort> KContext.simplifyArithMul(args: List<KExpr<T>>): KExpr<T> =
    simplifyArithMulLight(args, ::mkArithMulNoSimplify)

fun <T : KArithSort> KContext.simplifyArithDiv(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyArithDivLight(lhs, rhs, KContext::simplifyArithUnaryMinus, ::mkArithDivNoSimplify)

fun <T : KArithSort> KContext.simplifyArithPower(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    simplifyArithPowerLight(lhs, rhs, ::mkArithPowerNoSimplify)

fun <T : KArithSort> KContext.simplifyArithLe(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    simplifyArithLeLight(lhs, rhs, ::mkArithLeNoSimplify)

fun <T : KArithSort> KContext.simplifyArithLt(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    simplifyArithLtLight(lhs, rhs, ::mkArithLtNoSimplify)

fun <T : KArithSort> KContext.simplifyArithGe(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    rewriteArithGe(lhs, rhs, KContext::simplifyArithLe)

fun <T : KArithSort> KContext.simplifyArithGt(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    rewriteArithGt(lhs, rhs, KContext::simplifyArithLt)

fun KContext.simplifyIntMod(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KExpr<KIntSort> =
    simplifyIntModLight(lhs, rhs, ::mkIntModNoSimplify)

fun KContext.simplifyIntRem(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KExpr<KIntSort> =
    simplifyIntRemLight(lhs, rhs, ::mkIntRemNoSimplify)

fun KContext.simplifyIntToReal(arg: KExpr<KIntSort>): KExpr<KRealSort> =
    simplifyIntToRealLight(arg, ::mkIntToRealNoSimplify)

fun KContext.simplifyRealIsInt(arg: KExpr<KRealSort>): KExpr<KBoolSort> =
    simplifyRealIsIntLight(arg, ::mkRealIsIntNoSimplify)

fun KContext.simplifyRealToInt(arg: KExpr<KRealSort>): KExpr<KIntSort> =
    simplifyRealToIntLight(arg, ::mkRealToIntNoSimplify)

