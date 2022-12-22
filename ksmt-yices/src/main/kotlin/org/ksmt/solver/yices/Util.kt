package org.ksmt.solver.yices

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBvSort

typealias YicesTerm = Int
typealias YicesSort = Int

fun KContext.mkBv(bits: Array<Boolean>, sizeBits: UInt): KExpr<KBvSort> {
    val value = bits.map { if (it) 1 else 0 }.reversed().joinToString(separator = "")

    return mkBv(value, sizeBits)
}