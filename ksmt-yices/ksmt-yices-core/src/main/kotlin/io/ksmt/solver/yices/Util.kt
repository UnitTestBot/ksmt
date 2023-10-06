package io.ksmt.solver.yices

import com.sri.yices.BigRational
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.sort.KArithSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import java.math.BigInteger

typealias YicesTerm = Int
typealias YicesSort = Int
typealias YicesSortArray = IntArray
typealias YicesTermArray = IntArray

private fun BooleanArray.extractByte(start: Int): Byte {
    val end = Integer.min(start + Byte.SIZE_BITS, size)

    return copyOfRange(start, end).mapIndexed { index, b ->
        val bit = if (b) 1 else 0

        bit * (1 shl index)
    }.sum().toByte()
}

fun KContext.mkBv(bits: BooleanArray, sizeBits: UInt): KExpr<KBvSort> {
    val length = (bits.size + Byte.SIZE_BITS - 1) / Byte.SIZE_BITS
    val byteArray = ByteArray(length) {
        bits.extractByte((length - it - 1) * Byte.SIZE_BITS)
    }

    return mkBv(BigInteger(1, byteArray), sizeBits)
}

fun KContext.mkRealNum(value: BigRational): KExpr<KRealSort> =
    mkRealNum(mkIntNum(value.numerator), mkIntNum(value.denominator))

fun KContext.mkIntNum(value: BigRational): KExpr<KIntSort> =
    if (value.isInteger) {
        mkIntNum(value.numerator)
    } else {
        mkRealToInt(mkRealNum(value))
    }

fun KContext.mkArithNum(value: BigRational): KExpr<out KArithSort> =
    if (value.isInteger) mkIntNum(value) else mkRealNum(value)
