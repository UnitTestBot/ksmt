package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.sort.KRegexSort
import io.ksmt.sort.KStringSort

fun KContext.simplifyRegexConcat(
    arg0: KExpr<KRegexSort>,
    arg1: KExpr<KRegexSort>
): KExpr<KRegexSort> = mkRegexConcatNoSimplify(arg0, arg1) // Temporarily

fun KContext.simplifyRegexUnion(
    arg0: KExpr<KRegexSort>,
    arg1: KExpr<KRegexSort>
): KExpr<KRegexSort> = mkRegexUnionNoSimplify(arg0, arg1) // Temporarily

fun KContext.simplifyRegexIntersection(
    arg0: KExpr<KRegexSort>,
    arg1: KExpr<KRegexSort>
): KExpr<KRegexSort> = mkRegexIntersectionNoSimplify(arg0, arg1) // Temporarily

fun KContext.simplifyRegexStar(
    arg: KExpr<KRegexSort>
): KExpr<KRegexSort> = mkRegexStarNoSimplify(arg) // Temporarily

fun KContext.simplifyRegexCross(
    arg: KExpr<KRegexSort>
): KExpr<KRegexSort> = mkRegexCrossNoSimplify(arg) // Temporarily

fun KContext.simplifyRegexDifference(
    arg0: KExpr<KRegexSort>,
    arg1: KExpr<KRegexSort>
): KExpr<KRegexSort> = mkRegexDifferenceNoSimplify(arg0, arg1) // Temporarily

fun KContext.simplifyRegexComplement(
    arg: KExpr<KRegexSort>
): KExpr<KRegexSort> = mkRegexComplementNoSimplify(arg) // Temporarily

fun KContext.simplifyRegexOption(
    arg: KExpr<KRegexSort>
): KExpr<KRegexSort> = mkRegexOptionNoSimplify(arg) // Temporarily

fun KContext.simplifyRegexRange(
    arg0: KExpr<KStringSort>,
    arg1: KExpr<KStringSort>
): KExpr<KRegexSort> = mkRegexRangeNoSimplify(arg0, arg1) // Temporarily

fun KContext.simplifyRegexPower(
    power: Int,
    arg: KExpr<KRegexSort>,
): KExpr<KRegexSort> = mkRegexPowerNoSimplify(power, arg) // Temporarily

fun KContext.simplifyRegexLoop(
    from: Int,
    to: Int,
    arg: KExpr<KRegexSort>,
): KExpr<KRegexSort> = mkRegexLoopNoSimplify(from, to, arg) // Temporarily
