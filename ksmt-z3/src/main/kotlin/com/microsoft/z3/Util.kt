package com.microsoft.z3

fun intOrNull(ctx: Long, expr: Long): Int? {
    val result = Native.IntPtr()
    if (!Native.getNumeralInt(ctx, expr, result)) return null
    return result.value
}

fun longOrNull(ctx: Long, expr: Long): Long? {
    val result = Native.LongPtr()
    if (!Native.getNumeralInt64(ctx, expr, result)) return null
    return result.value
}

fun fpSignificandUInt64OrNull(ctx: Long, expr: Long): Long? {
    val result = Native.LongPtr()
    if (!Native.fpaGetNumeralSignificandUint64(ctx, expr, result)) return null
    return result.value
}

fun fpExponentInt64OrNull(ctx: Long, expr: Long, biased: Boolean): Long? {
    val result = Native.LongPtr()
    if (!Native.fpaGetNumeralExponentInt64(ctx, expr, result, biased)) return null
    return result.value
}

fun fpSignOrNull(ctx: Long, expr: Long): Boolean? {
    val result = Native.IntPtr()
    if (!Native.fpaGetNumeralSign(ctx, expr, result)) return null
    return result.value != 0
}

@Suppress("LongParameterList")
fun mkQuantifier(
    ctx: Long,
    isUniversal: Boolean,
    boundConsts: LongArray,
    body: Long,
    weight: Int,
    patterns: LongArray
): Long = Native.mkQuantifierConst(
    ctx, isUniversal, weight, boundConsts.size, boundConsts, patterns.size, patterns, body
)

fun getAppArgs(ctx: Long, expr: Long): LongArray {
    val size = Native.getAppNumArgs(ctx, expr)
    return LongArray(size) { idx -> Native.getAppArg(ctx, expr, idx) }
}
