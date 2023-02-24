package com.microsoft.z3

import com.microsoft.z3.enumerations.Z3_error_code

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

/**
 * We have no way to obtain array sort domain size.
 * To overcome this we iterate over domain until index is out of bounds.
 * */
fun getArraySortDomain(ctx: Long, sort: Long): List<Long> {
    val domain = arrayListOf<Long>()
    while (true) {
        val domainSortI = Native.INTERNALgetArraySortDomainN(ctx, sort, domain.size)
        when (val errorCode = Z3_error_code.fromInt(Native.INTERNALgetErrorCode(ctx))) {
            Z3_error_code.Z3_OK -> {
                domain.add(domainSortI)
                continue
            }

            /**
             * Z3 set [Z3_error_code.Z3_INVALID_ARG] error code when sort domain index is out of bounds.
             * */
            Z3_error_code.Z3_INVALID_ARG -> break
            else -> throw Z3Exception(Native.INTERNALgetErrorMsg(ctx, errorCode.toInt()))
        }
    }
    return domain
}
