package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort

fun KContext.simplifyNot(arg: KExpr<KBoolSort>): KExpr<KBoolSort> = mkNotNoSimplify(arg)

fun KContext.simplifyAnd(args: List<KExpr<KBoolSort>>): KExpr<KBoolSort> = mkAndNoSimplify(args)

fun KContext.simplifyOr(args: List<KExpr<KBoolSort>>): KExpr<KBoolSort> = mkOrNoSimplify(args)

fun KContext.simplifyImplies(p: KExpr<KBoolSort>, q: KExpr<KBoolSort>): KExpr<KBoolSort> = mkImpliesNoSimplify(p, q)

fun KContext.simplifyXor(a: KExpr<KBoolSort>, b: KExpr<KBoolSort>): KExpr<KBoolSort> = mkXorNoSimplify(a, b)


fun <T : KSort> KContext.simplifyEq(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> = mkEqNoSimplify(lhs, rhs)

fun <T : KSort> KContext.simplifyDistinct(args: List<KExpr<T>>): KExpr<KBoolSort> = mkDistinctNoSimplify(args)

fun <T : KSort> KContext.simplifyIte(
    condition: KExpr<KBoolSort>,
    trueBranch: KExpr<T>,
    falseBranch: KExpr<T>
): KExpr<T> = mkIteNoSimplify(condition, trueBranch, falseBranch)
