package org.ksmt.solver.yices

import com.sri.yices.BigRational
import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KRealSort

typealias YicesTerm = Int
typealias YicesSort = Int

fun KContext.mkBv(bits: BooleanArray, sizeBits: UInt): KExpr<KBvSort> {
    val value = bits.map { if (it) 1 else 0 }.reversed().joinToString(separator = "")

    return mkBv(value, sizeBits)
}

fun KContext.mkRealNum(value: BigRational): KExpr<KRealSort> =
    mkRealNum(mkIntNum(value.numerator), mkIntNum(value.denominator))
