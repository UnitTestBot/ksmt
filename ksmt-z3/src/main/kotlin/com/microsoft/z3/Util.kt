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

@Suppress("LongParameterList")
fun Context.mkExistsQuantifier(
    boundConstants: Array<Expr<*>>,
    body: Expr<BoolSort>,
    weight: Int,
    patterns: Array<Pattern>,
    noPatterns: Array<Expr<*>>,
    quantifierId: Symbol?,
    skolemId: Symbol?
): Quantifier = mkExists(boundConstants, body, weight, patterns, noPatterns, quantifierId, skolemId)

@Suppress("LongParameterList")
fun Context.mkForallQuantifier(
    boundConstants: Array<Expr<*>>,
    body: Expr<BoolSort>,
    weight: Int,
    patterns: Array<Pattern>,
    noPatterns: Array<Expr<*>>,
    quantifierId: Symbol?,
    skolemId: Symbol?
): Quantifier = mkForall(boundConstants, body, weight, patterns, noPatterns, quantifierId, skolemId)

val Expr<*>.ctx: Context
    get() = context

fun Context.mkBvNumeral(bits: BooleanArray): Expr<*> =
    Expr.create(this, Native.mkBvNumeral(nCtx(), bits.size, bits))

val Quantifier.isLambda: Boolean
    get() = Native.isLambda(context.nCtx(), nativeObject)
