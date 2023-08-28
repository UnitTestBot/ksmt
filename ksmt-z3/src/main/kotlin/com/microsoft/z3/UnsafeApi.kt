package com.microsoft.z3

fun incRefUnsafe(ctx: Long, ast: Long) {
    // Invoke incRef directly without status check
    Native.INTERNALincRef(ctx, ast)
}

fun decRefUnsafe(ctx: Long, ast: Long) {
    // Invoke decRef directly without status check
    Native.INTERNALdecRef(ctx, ast)
}
