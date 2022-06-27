package com.microsoft.z3

fun IntNum.intOrNull(): Int? {
    val result = Native.IntPtr()
    if (!Native.getNumeralInt(context.nCtx(), nativeObject, result)) return null
    return result.value
}

fun IntNum.longOrNull(): Long? {
    val result = Native.LongPtr()
    if (!Native.getNumeralInt64(context.nCtx(), nativeObject, result)) return null
    return result.value
}

fun Context.mkExistsQuantifier(
    boundConstants: Array<Expr>,
    body: Expr,
    weight: Int,
    patterns: Array<Pattern>,
    noPatterns: Array<Expr>,
    quantifierId: Symbol?,
    skolemId: Symbol?
): Quantifier = mkExists(boundConstants, body, weight, patterns, noPatterns, quantifierId, skolemId)

fun Context.mkForallQuantifier(
    boundConstants: Array<Expr>,
    body: Expr,
    weight: Int,
    patterns: Array<Pattern>,
    noPatterns: Array<Expr>,
    quantifierId: Symbol?,
    skolemId: Symbol?
): Quantifier = mkForall(boundConstants, body, weight, patterns, noPatterns, quantifierId, skolemId)
