package com.microsoft.z3

import com.microsoft.z3.Native.IntPtr
import com.microsoft.z3.Native.LongPtr
import com.microsoft.z3.enumerations.Z3_error_code
import com.microsoft.z3.enumerations.Z3_lbool

fun intOrNull(ctx: Long, expr: Long): Int? {
    val result = IntPtr()
    if (!Native.getNumeralInt(ctx, expr, result)) return null
    return result.value
}

fun longOrNull(ctx: Long, expr: Long): Long? {
    val result = LongPtr()
    if (!Native.getNumeralInt64(ctx, expr, result)) return null
    return result.value
}

fun fpSignificandUInt64OrNull(ctx: Long, expr: Long): Long? {
    val result = LongPtr()
    if (!Native.fpaGetNumeralSignificandUint64(ctx, expr, result)) return null
    return result.value
}

fun fpExponentInt64OrNull(ctx: Long, expr: Long, biased: Boolean): Long? {
    val result = LongPtr()
    if (!Native.fpaGetNumeralExponentInt64(ctx, expr, result, biased)) return null
    return result.value
}

fun fpSignOrNull(ctx: Long, expr: Long): Boolean? {
    val result = IntPtr()
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
@Suppress("LoopWithTooManyJumpStatements")
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

fun Solver.solverAssert(expr: Long) {
    Native.solverAssert(context.nCtx(), nativeObject, expr)
}

fun Solver.solverAssertAndTrack(expr: Long, track: Long) {
    Native.solverAssertAndTrack(context.nCtx(), nativeObject, expr, track)
}

fun Solver.solverCheckAssumptions(assumptions: LongArray): Status {
    val checkResult = Native.solverCheckAssumptions(context.nCtx(), nativeObject, assumptions.size, assumptions)
    return when (Z3_lbool.fromInt(checkResult)) {
        Z3_lbool.Z3_L_TRUE -> Status.SATISFIABLE
        Z3_lbool.Z3_L_FALSE -> Status.UNSATISFIABLE
        else -> Status.UNKNOWN
    }
}

private fun astVectorToLongArray(ctx: Long, vector: Long): LongArray {
    Native.astVectorIncRef(ctx, vector)
    return try {
        val size = Native.astVectorSize(ctx, vector)
        LongArray(size) {
            Native.astVectorGet(ctx, vector, it)
        }
    } finally {
        Native.astVectorDecRef(ctx, vector)
    }
}

fun Solver.solverGetUnsatCore(): LongArray {
    val nativeUnsatCoreVector = Native.solverGetUnsatCore(context.nCtx(), nativeObject)
    return astVectorToLongArray(context.nCtx(), nativeUnsatCoreVector)
}

fun Model.getNativeConstDecls(): LongArray = LongArray(numConsts) {
    Native.modelGetConstDecl(context.nCtx(), nativeObject, it)
}

fun Model.getNativeFuncDecls(): LongArray = LongArray(numFuncs) {
    Native.modelGetFuncDecl(context.nCtx(), nativeObject, it)
}

fun Model.getNativeSorts(): LongArray = LongArray(numSorts) {
    Native.modelGetSort(context.nCtx(), nativeObject, it)
}

fun Model.getSortUniverse(sort: Long): LongArray {
    val nativeSortUniverseVector = Native.modelGetSortUniverse(context.nCtx(), nativeObject, sort)
    return astVectorToLongArray(context.nCtx(), nativeSortUniverseVector)
}

fun Model.evalNative(expr: Long, complete: Boolean): Long {
    val result = LongPtr()
    if (!Native.modelEval(context.nCtx(), nativeObject, expr, complete, result)) {
        throw ModelEvaluationFailedException()
    }
    return result.value
}

fun Model.getConstInterp(decl: Long): Long? {
    val interp = Native.modelGetConstInterp(context.nCtx(), nativeObject, decl)
    return interp.takeIf { it != 0L }
}

fun Model.getFuncInterp(decl: Long): Z3NativeFuncInterp? {
    val interp = Native.modelGetFuncInterp(context.nCtx(), nativeObject, decl)
    if (interp == 0L) return null
    return retrieveNativeFuncInterp(context.nCtx(), interp)
}

class Z3NativeFuncInterpEntry(
    val args: LongArray,
    val value: Long
)

class Z3NativeFuncInterp(
    val entries: Array<Z3NativeFuncInterpEntry>,
    val elseExpr: Long
)

private fun retrieveNativeFuncInterp(ctx: Long, nativeFuncInterp: Long): Z3NativeFuncInterp {
    Native.funcInterpIncRef(ctx, nativeFuncInterp)
    return try {
        val numEntries = Native.funcInterpGetNumEntries(ctx, nativeFuncInterp)
        val entries = Array(numEntries) {
            val nativeEntry = Native.funcInterpGetEntry(ctx, nativeFuncInterp, it)
            retrieveNativeFuncInterpEntry(ctx, nativeEntry)
        }
        val elseExpr = Native.funcInterpGetElse(ctx, nativeFuncInterp)
        Z3NativeFuncInterp(entries, elseExpr)
    } finally {
        Native.funcInterpDecRef(ctx, nativeFuncInterp)
    }
}

private fun retrieveNativeFuncInterpEntry(ctx: Long, nativeEntry: Long): Z3NativeFuncInterpEntry {
    Native.funcEntryIncRef(ctx, nativeEntry)
    return try {
        val argsSize = Native.funcEntryGetNumArgs(ctx, nativeEntry)
        val args = LongArray(argsSize) {
            Native.funcEntryGetArg(ctx, nativeEntry, it)
        }
        val value = Native.funcEntryGetValue(ctx, nativeEntry)
        Z3NativeFuncInterpEntry(args, value)
    } finally {
        Native.funcEntryDecRef(ctx, nativeEntry)
    }
}
