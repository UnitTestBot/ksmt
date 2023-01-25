package org.ksmt

import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.expr.KFpRoundingModeExpr
import org.ksmt.expr.KFpValue
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort
import org.ksmt.utils.BvUtils
import org.ksmt.utils.FpUtils.mkFpMaxValue
import org.ksmt.utils.uncheckedCast
import kotlin.random.Random
import kotlin.random.nextInt

abstract class ExpressionEvalTest {

    fun <S : KBvSort> KContext.randomBvValues(sort: S) = sequence<KBitVecValue<S>> {
        // special values
        with(BvUtils) {
            yield(bvMaxValueSigned(sort.sizeBits).uncheckedCast())
            yield(bvMaxValueUnsigned(sort.sizeBits).uncheckedCast())
            yield(bvMinValueSigned(sort.sizeBits).uncheckedCast())
            yield(bvZero(sort.sizeBits).uncheckedCast())
            yield(bvOne(sort.sizeBits).uncheckedCast())
        }

        // small positive values
        repeat(5) {
            val value = random.nextInt(1..20)
            yield(mkBv(value, sort))
        }

        // small negative values
        repeat(5) {
            val value = random.nextInt(1..20)
            yield(mkBv(-value, sort))
        }

        // random values
        repeat(30) {
            val binaryValue = String(CharArray(sort.sizeBits.toInt()) {
                if (random.nextBoolean()) '1' else '0'
            })
            yield(mkBv(binaryValue, sort.sizeBits).uncheckedCast())
        }
    }

    fun <S : KBvSort> KContext.randomBvNonZeroValues(sort: S): Sequence<KBitVecValue<S>> {
        val zero = mkBv(0, sort)
        return randomBvValues(sort).filter { it != zero }
    }

    fun KContext.roundingModeValues(): Sequence<KFpRoundingModeExpr> =
        KFpRoundingMode.values().asSequence().map { mkFpRoundingModeExpr(it) }

    fun <S : KFpSort> KContext.randomFpValues(sort: S) = sequence<KFpValue<S>> {
        // special values
        yield(mkFpZero(sort = sort, signBit = true))
        yield(mkFpZero(sort = sort, signBit = false))
        yield(mkFpInf(sort = sort, signBit = true))
        yield(mkFpInf(sort = sort, signBit = false))
        yield(mkFpNan(sort))
        yield(mkFpMaxValue(sort = sort, signBit = false))
        yield(mkFpMaxValue(sort = sort, signBit = true))

        // small positive values
        repeat(5) {
            val value = random.nextDouble()
            yield(mkFp(value, sort))
        }

        // small negative values
        repeat(5) {
            val value = random.nextDouble()
            yield(mkFp(-value, sort))
        }

        // random values
        val exponentBvSort = mkBvSort(sort.exponentBits)
        val significandBvSort = mkBvSort(sort.significandBits - 1u)
        randomBvValues(exponentBvSort).shuffled(random).forEach { exponent ->
            randomBvValues(significandBvSort).shuffled(random).forEach { significand ->
                val sign = random.nextBoolean()
                val value = mkFpBiased(significand, exponent, sign, sort)
                yield(value)
            }
        }
    }

    companion object {
        val random = Random(42)
    }
}
