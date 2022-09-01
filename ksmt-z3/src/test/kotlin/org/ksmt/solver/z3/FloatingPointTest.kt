package org.ksmt.solver.z3

import java.lang.Float.intBitsToFloat
import kotlin.math.sign
import kotlin.random.Random
import kotlin.random.nextUInt
import kotlin.test.assertTrue
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.sort.KFpSort
import org.ksmt.utils.booleanSignBit
import org.ksmt.utils.extractExponent
import org.ksmt.utils.extractSignificand


class FloatingPointTest {
    private var context = KContext()
    private var solver = KZ3Solver(context)

    @BeforeEach
    fun createNewEnvironment() {
        context = KContext()
        solver = KZ3Solver(context)
    }

    @AfterEach
    fun clearResources() {
        solver.close()
    }

    private fun <S : KFpSort> KContext.symbolicValuesCheck(symbolicValues: List<KExpr<S>>, sort: S) {
        val symbolicConsts = symbolicValues.indices.map { sort.mkConst("const_${it}") }
        val pairs = symbolicValues.zip(symbolicConsts)

        pairs.forEach { (value, const) ->
            solver.assert(value eq const)
        }

        solver.check()
        val model = solver.model()

        pairs.forEach { (value, const) ->
            assertTrue(
                "Values for $const are different: ${System.lineSeparator()}" +
                        "expected: $value ${System.lineSeparator()}" +
                        "found:    ${model.eval(const)}"
            ) { model.eval(const) === value }
        }
    }

    private fun <S : KFpSort> KContext.createSymbolicValues(
        it: Float,
        sort: S,
        mkSpecificSort: (Float) -> KExpr<S>
    ): List<KExpr<S>> = listOf(
        mkSpecificSort(it),
        mkFp(it, sort),
        mkFp(
            it.extractSignificand(sort),
            it.extractExponent(sort, isBiased = false),
            signBit = it.booleanSignBit,
            sort
        ),
        mkFp(
            it.extractSignificand(sort).toLong(),
            it.extractExponent(sort, isBiased = false).toLong(),
            signBit = it.booleanSignBit,
            sort
        )
    )

    private fun <S : KFpSort> KContext.createSymbolicValues(
        it: Double,
        sort: S,
        mkSpecificSort: (Double) -> KExpr<S>
    ): List<KExpr<S>> = listOf(
        mkSpecificSort(it),
        mkFp(it, sort),
        mkFp(
            it.extractSignificand(sort),
            it.extractExponent(sort, isBiased = false),
            signBit = it.booleanSignBit,
            sort
        ),
        mkFp(
            it.extractSignificand(sort),
            it.extractExponent(sort, isBiased = false),
            signBit = it.booleanSignBit,
            sort
        )
    )

    @Test
    fun testCreateFp16(): Unit = with(context) {
        val values = (0..10000)
            .map {
                val sign = Random.nextInt(from = 0, until = 2)
                val exponent = Random.nextInt(from = 128, until = 142)
                val significand = Random.nextInt(from = 0, until = 1024)
                intBitsToFloat(((sign shl 31) or (exponent shl 23) or (significand shl 13)))
            }.distinct()

        val sort = mkFp16Sort()

        val symbolicValues = values.map {
            createSymbolicValues(it, sort, context::mkFp16).distinct().single()
        }

        symbolicValuesCheck(symbolicValues, sort)
    }

    @Test
    fun testCreateFp32(): Unit = with(context) {
        val values = (0..10000).map { Random.nextFloat() }

        val sort = mkFp32Sort()

        val symbolicValues = values.map {
            createSymbolicValues(it, sort, context::mkFp32).distinct().single()
        }

        symbolicValuesCheck(symbolicValues, sort)
    }

    @Test
    fun testCreateFp64(): Unit = with(context) {
        val values = (0..1000).map {
            Random.nextFloat().toDouble()
        }

        val sort = mkFp64Sort()

        val symbolicValues = values.map {
            createSymbolicValues(it, sort, context::mkFp64).distinct().single()
        }

        symbolicValuesCheck(symbolicValues, sort)
    }

    @Test
    fun testCreateFp128(): Unit = with(context) {
        val values = (0..1000)
            .map {
                Random.nextLong() to Random.nextLong(
                    from = 0b000000000000000.toLong(),
                    until = 0b011111111111111.toLong()
                ) * sign(Random.nextInt().toDouble()).toLong()
            }

        val randomDoubles = (0..1000).map { Random.nextDouble() }
        val randomFloats = (0..1000).map { Random.nextFloat() }

        val signBit = Random.nextBoolean()

        val sort = mkFp128Sort()

        val symbolicValues = values.map {
            listOf(
                mkFp128(it.first, it.second, signBit),
                mkFp(it.first, it.second, signBit, sort)
            ).distinct().single()
        }.toMutableList()

        symbolicValues += randomDoubles.map { mkFp(it, sort) }
        symbolicValues += randomFloats.map { mkFp(it, sort) }

        symbolicValuesCheck(symbolicValues, sort)
    }

    @Test
    @Disabled("Not supported yet, doesn't work with particular sorts, for example, FP 32 132")
    fun testCreateFpCustomSize(): Unit =
        repeat(10) {
            createNewEnvironment()
            with(context) {
                // mkFpSort(3u, 127u) listOf(mkFp(-1054027720, sort)) and 2u don't work
                val sort = mkFpSort(
                    Random.nextInt(from = 2, until = 64).toUInt(),
                    Random.nextUInt(from = 10u, until = 150u)
                )
                // val sort = mkFpSort(4u, 92u)

                println("${it + 1} run, sort: $sort")

                val values = (0..100)
                    .map {
                        val (significand, exponent) = Random.nextLong() to Random.nextLong()
                        val finalSignificand = significand and ((1L shl sort.significandBits.toInt() - 1) - 1)
                        val finalExponent = exponent and ((1L shl sort.exponentBits.toInt()) - 1)

                        finalSignificand to finalExponent
                    }
                // TODO this combination doesn't work
                // 7 run, sort: FP (eBits: 6) (sBits: 109)
                // val values = listOf((-8634236606667726792L to 33L))

                // TODO here we should apply masks to avoid exponents that are not in [min..max] range
                 val randomDoubles = (0..1000).map { Random.nextDouble() }
                 val randomFloats = (0..1000).map { Random.nextFloat() }

                val signBit = Random.nextBoolean()

                // TODO not supported yet
                val symbolicValues = values.mapTo(mutableListOf()) { (significand, exponent) ->
                    mkFp(significand, exponent, signBit, sort)
                }
                 symbolicValues += randomDoubles.mapTo(mutableListOf()) { value -> mkFp(value, sort) }
                 symbolicValues += randomFloats.map { value -> mkFp(value, sort) }

                symbolicValuesCheck(symbolicValues, sort)
            }
        }
}
