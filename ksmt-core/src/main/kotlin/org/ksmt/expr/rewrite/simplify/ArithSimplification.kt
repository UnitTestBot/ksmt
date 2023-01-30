package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort

fun <T : KArithSort> KContext.simplifyArithUnaryMinus(arg: KExpr<T>): KExpr<T> = mkArithUnaryMinusNoSimplify(arg)

fun <T : KArithSort> KContext.simplifyArithAdd(args: List<KExpr<T>>): KExpr<T> = mkArithAddNoSimplify(args)

fun <T : KArithSort> KContext.simplifyArithSub(args: List<KExpr<T>>): KExpr<T> = mkArithSubNoSimplify(args)

fun <T : KArithSort> KContext.simplifyArithMul(args: List<KExpr<T>>): KExpr<T> = mkArithMulNoSimplify(args)

fun <T : KArithSort> KContext.simplifyArithDiv(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> = mkArithDivNoSimplify(lhs, rhs)

fun <T : KArithSort> KContext.simplifyArithPower(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
    mkArithPowerNoSimplify(lhs, rhs)


fun <T : KArithSort> KContext.simplifyArithGe(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    mkArithGeNoSimplify(lhs, rhs)

fun <T : KArithSort> KContext.simplifyArithGt(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    mkArithGtNoSimplify(lhs, rhs)

fun <T : KArithSort> KContext.simplifyArithLe(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    mkArithLeNoSimplify(lhs, rhs)

fun <T : KArithSort> KContext.simplifyArithLt(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
    mkArithLtNoSimplify(lhs, rhs)


fun KContext.simplifyIntMod(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KExpr<KIntSort> = mkIntModNoSimplify(lhs, rhs)

fun KContext.simplifyIntRem(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KExpr<KIntSort> = mkIntRemNoSimplify(lhs, rhs)

fun KContext.simplifyIntToReal(arg: KExpr<KIntSort>): KExpr<KRealSort> = mkIntToRealNoSimplify(arg)

fun KContext.simplifyRealIsInt(arg: KExpr<KRealSort>): KExpr<KBoolSort> = mkRealIsIntNoSimplify(arg)

fun KContext.simplifyRealToInt(arg: KExpr<KRealSort>): KExpr<KIntSort> = mkRealToIntNoSimplify(arg)

