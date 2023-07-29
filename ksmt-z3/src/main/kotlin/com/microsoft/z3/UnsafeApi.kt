package com.microsoft.z3

import it.unimi.dsi.fastutil.longs.LongSet

fun incRefUnsafe(ctx: Long, ast: Long) {
    // Invoke incRef directly without status check
    Native.INTERNALincRef(ctx, ast)
}

fun decRefUnsafe(ctx: Long, ast: Long) {
    // Invoke decRef directly without status check
    Native.INTERNALdecRef(ctx, ast)
}

fun LongSet.decRefUnsafeAll(ctx: Long) = longIterator().forEachRemaining {
    decRefUnsafe(ctx, it)
}
