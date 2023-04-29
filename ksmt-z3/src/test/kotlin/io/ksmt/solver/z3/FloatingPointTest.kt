package io.ksmt.solver.z3

import com.microsoft.z3.Context
import com.microsoft.z3.Expr
import com.microsoft.z3.FPSort
import com.microsoft.z3.Status
import java.lang.Float.intBitsToFloat
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sign
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.random.nextUInt
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import io.ksmt.KContext
import io.ksmt.decl.KConstDecl
import io.ksmt.expr.KApp
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFp16Value
import io.ksmt.expr.KFp32Value
import io.ksmt.expr.KFp64Value
import io.ksmt.expr.KFpRoundingMode
import io.ksmt.expr.KFpRoundingModeExpr
import io.ksmt.expr.KTrue
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KFp16Sort
import io.ksmt.sort.KFp32Sort
import io.ksmt.sort.KFp64Sort
import io.ksmt.sort.KFpSort
import io.ksmt.utils.booleanSignBit
import io.ksmt.utils.extractExponent
import io.ksmt.utils.extractSignificand
import io.ksmt.utils.getHalfPrecisionExponent
import io.ksmt.utils.getValue
import io.ksmt.utils.halfPrecisionSignificand
import io.ksmt.utils.mkConst


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
        ),
        mkFpBiased(
            it.extractSignificand(sort),
            it.extractExponent(sort, isBiased = true),
            signBit = it.booleanSignBit,
            sort
        ),
        mkFpBiased(
            it.extractSignificand(sort).toLong(),
            it.extractExponent(sort, isBiased = true).toLong(),
            signBit = it.booleanSignBit,
            sort
        ),
        ((mkSpecificSort(it) as KApp<S, *>).decl as KConstDecl<S>).apply()
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
        mkFpBiased(
            it.extractSignificand(sort),
            it.extractExponent(sort, isBiased = true),
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
    fun testFp16Normalization(): Unit = with(context) {
        val sign = 1
        val exponent1 = "10001111".toInt(radix = 2)
        val exponent2 = "10101111".toInt(radix = 2)
        val significand1 = "11001100110000000000000".toInt(radix = 2)
        val significand2 = "11001100110001111111111".toInt(radix = 2)

        val value1 = intBitsToFloat(((sign shl 31) or (exponent1 shl 23) or significand1))
        val value2 = intBitsToFloat(((sign shl 31) or (exponent2 shl 23) or significand2))

        assertEquals(value1.getHalfPrecisionExponent(true), value2.getHalfPrecisionExponent(true))
        assertEquals(value1.halfPrecisionSignificand, value2.halfPrecisionSignificand)

        val symbolicValues1 = createSymbolicValues(value1, fp16Sort, context::mkFp16)
        val symbolicValues2 = createSymbolicValues(value2, fp16Sort, context::mkFp16)

        val distinctSymbolicValues = (symbolicValues1 + symbolicValues2).distinct()
        assertEquals(1, distinctSymbolicValues.size)
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

    @Test
    fun testFpAbsExpr(): Unit = with(context) {
        val negativeNumber = Random.nextDouble(from = Long.MIN_VALUE.toDouble(), until = 0.0)
        val positiveNumber = Random.nextDouble(from = 0.0, Long.MAX_VALUE.toDouble())

        val negativeFp = negativeNumber.toFp() as KFp64Value
        val positiveFp = positiveNumber.toFp() as KFp64Value

        val negativeVariable = mkFp64Sort().mkConst("negativeValue")
        val positiveVariable = mkFp64Sort().mkConst("positiveValue")

        solver.assert(mkFpAbsExpr(negativeFp) eq negativeVariable)
        solver.assert(mkFpAbsExpr(positiveFp) eq positiveVariable)

        solver.check()
        with(solver.model()) {
            assertEquals(abs(negativeNumber), (eval(negativeVariable) as KFp64Value).value, DELTA)
            assertEquals(positiveNumber, (eval(positiveVariable) as KFp64Value).value, DELTA)
        }
    }

    @Test
    fun testFpNegation(): Unit = with(context) {
        val number = Random.nextDouble()
        val numberFp = number.toFp() as KFp64Value

        val variable = mkFp64Sort().mkConst("variable")

        solver.assert(mkFpNegationExpr(numberFp) eq variable)

        solver.check()
        with(solver.model()) {
            assertEquals(-1 * number, (eval(variable) as KFp64Value).value, DELTA)
        }
    }

    private fun testBinaryArithOperation(
        symbolicOperation: (KFpRoundingModeExpr, KFp64Value, KFp64Value) -> KExpr<KFp64Sort>,
        concreteOperation: (Double, Double) -> Double
    ): Unit = with(context) {
        val fst = Random.nextDouble()
        val snd = Random.nextDouble()

        val fstFp = fst.toFp() as KFp64Value
        val sndFp = snd.toFp() as KFp64Value

        val fstVariable = mkFp64Sort().mkConst("fstVariable")
        val sndVariable = mkFp64Sort().mkConst("sndVariable")

        val roundingMode = KFpRoundingMode.RoundNearestTiesToEven

        solver.assert(symbolicOperation(mkFpRoundingModeExpr(roundingMode), fstFp, sndFp) eq fstVariable)
        solver.assert(symbolicOperation(mkFpRoundingModeExpr(roundingMode), sndFp, fstFp) eq sndVariable)

        solver.check()
        with(solver.model()) {
            assertEquals(concreteOperation(fst, snd), (eval(fstVariable) as KFp64Value).value, DELTA)
            assertEquals(concreteOperation(snd, fst), (eval(sndVariable) as KFp64Value).value, DELTA)
        }
    }

    @Test
    fun testFpAddExpr() = testBinaryArithOperation(context::mkFpAddExpr) { a, b -> a + b }

    @Test
    fun testFpSubExpr() = testBinaryArithOperation(context::mkFpSubExpr) { a, b -> a - b }

    @Test
    fun testFpMulExpr() = testBinaryArithOperation(context::mkFpMulExpr) { a, b -> a * b }

    @Test
    fun testFpDivExpr() = testBinaryArithOperation(context::mkFpDivExpr) { a, b -> a / b }

    @Test
    fun testFpFusedMulAddExpr(): Unit = with(context) {
        val fst = Random.nextDouble()
        val snd = Random.nextDouble()
        val third = Random.nextDouble()

        val fstFp = fst.toFp() as KFp64Value
        val sndFp = snd.toFp() as KFp64Value
        val thirdFp = third.toFp() as KFp64Value

        val fstVariable = mkFp64Sort().mkConst("fstVariable")
        val sndVariable = mkFp64Sort().mkConst("sndVariable")
        val thirdVariable = mkFp64Sort().mkConst("thirdVariable")

        val roundingMode = KFpRoundingMode.RoundNearestTiesToEven

        solver.assert(mkFpFusedMulAddExpr(mkFpRoundingModeExpr(roundingMode), fstFp, sndFp, thirdFp) eq fstVariable)
        solver.assert(mkFpFusedMulAddExpr(mkFpRoundingModeExpr(roundingMode), sndFp, thirdFp, fstFp) eq sndVariable)
        solver.assert(mkFpFusedMulAddExpr(mkFpRoundingModeExpr(roundingMode), thirdFp, fstFp, sndFp) eq thirdVariable)

        solver.check()
        with(solver.model()) {
            assertEquals((fst * snd) + third, (eval(fstVariable) as KFp64Value).value, DELTA)
            assertEquals((snd * third) + fst, (eval(sndVariable) as KFp64Value).value, DELTA)
            assertEquals((third * fst) + snd, (eval(thirdVariable) as KFp64Value).value, DELTA)
        }
    }

    @Test
    fun testFpSqrtExpr(): Unit = with(context) {
        val value = Random.nextDouble()
        val valueFp = value.toFp() as KFp64Value
        val valueVariable = mkFp64Sort().mkConst("fstVariable")

        val roundingMode = KFpRoundingMode.RoundNearestTiesToEven

        solver.assert(mkFpSqrtExpr(mkFpRoundingModeExpr(roundingMode), valueFp) eq valueVariable)

        solver.check()
        with(solver.model()) {
            assertEquals(sqrt(value), (eval(valueVariable) as KFp64Value).value, DELTA)
        }
    }

    @Test
    fun testFpRoundToIntegral(): Unit = with(context) {
        val value = Random.nextDouble()

        val roundingModes = KFpRoundingMode.values()

        val variables = roundingModes.indices.map { mkFp64Sort().mkConst("variable$it") }

        variables.forEachIndexed { index, it ->
            val rm = mkFpRoundingModeExpr(roundingModes[index])
            solver.assert(mkFpRoundToIntegralExpr(rm, value.toFp() as KFp64Value) eq it)
        }

        solver.check()
        with(solver.model()) {
            variables.map { (eval(it) as KFp64Value).value }
        }
    }

    private fun testMinMax(
        symbolicOperation: (KFp64Value, KFp64Value) -> KExpr<KFp64Sort>,
        concreteOperation: (Double, Double) -> Double
    ): Unit = with(context) {
        val fst = Random.nextDouble()
        val snd = Random.nextDouble()

        val fstFp = fst.toFp() as KFp64Value
        val sndFp = snd.toFp() as KFp64Value

        val fstVariable = mkFp64Sort().mkConst("fstVariable")
        val sndVariable = mkFp64Sort().mkConst("sndVariable")

        solver.assert(symbolicOperation(fstFp, sndFp) eq fstVariable)
        solver.assert(symbolicOperation(sndFp, fstFp) eq sndVariable)

        solver.check()
        with(solver.model()) {
            assertEquals(concreteOperation(fst, snd), (eval(fstVariable) as KFp64Value).value, DELTA)
            assertEquals(concreteOperation(snd, fst), (eval(sndVariable) as KFp64Value).value, DELTA)
        }
    }

    @Test
    fun testMinValue(): Unit = testMinMax(context::mkFpMinExpr) { a: Double, b: Double -> min(a, b) }

    @Test
    fun testMaxValue(): Unit = testMinMax(context::mkFpMaxExpr) { a: Double, b: Double -> max(a, b) }

    private inline fun <reified S : KFpSort, reified C : Number> testConstant(
        sort: S,
        symbolicOperation: () -> KExpr<S>,
        concreteOperation: () -> C,
        valueGetter: (KExpr<S>) -> C,
        assertEquals: (C, C) -> Unit
    ): Unit = with(context) {
        val constVar by sort

        val symbolicValue = symbolicOperation()

        val decl = (symbolicValue as KApp<S, *>).decl as KConstDecl<S>
        val declValue = decl.apply()

        solver.assert(symbolicValue eq constVar)

        solver.check()
        with(solver.model()) {
            assertEquals(concreteOperation(), valueGetter(eval(constVar)))
            assertEquals(symbolicValue, declValue)
        }
    }

    private fun testDoubleConstant(
        symbolicOperation: () -> KExpr<KFp64Sort>,
        concreteOperation: () -> Double
    ) = testConstant(
        context.mkFp64Sort(),
        symbolicOperation,
        concreteOperation,
        { (it as KFp64Value).value },
        { l, r -> assertEquals(l, r, DELTA) }
    )

    private fun testFloatConstant(
        symbolicOperation: () -> KExpr<KFp32Sort>,
        concreteOperation: () -> Float
    ) = testConstant(
        context.mkFp32Sort(),
        symbolicOperation,
        concreteOperation,
        { (it as KFp32Value).value },
        { l, r -> assertEquals(l, r, absoluteTolerance = 1e-12f) }
    )

    private fun testHalfConstant(
        symbolicOperation: () -> KExpr<KFp16Sort>,
        concreteOperation: () -> Float
    ) = testConstant(
        context.mkFp16Sort(),
        symbolicOperation,
        concreteOperation,
        { (it as KFp16Value).value },
        { l, r -> assertEquals(l, r, absoluteTolerance = 1e-12f) }
    )

    @Test
    fun testDoubleNaNValue() = testDoubleConstant(
        symbolicOperation = { context.mkFpNaN(context.mkFp64Sort()) },
        concreteOperation = { Double.NaN }
    )

    @Test
    fun testDoublePosInfValue() = testDoubleConstant(
        symbolicOperation = { context.mkFpInf(signBit = false, context.mkFp64Sort()) },
        concreteOperation = { Double.POSITIVE_INFINITY }
    )

    @Test
    fun testDoubleNegInfValue() = testDoubleConstant(
        symbolicOperation = { context.mkFpInf(signBit = true, context.mkFp64Sort()) },
        concreteOperation = { Double.NEGATIVE_INFINITY }
    )

    @Test
    fun testDoublePosZeroValue() = testDoubleConstant(
        symbolicOperation = { context.mkFpZero(signBit = false, context.mkFp64Sort()) },
        concreteOperation = { +0.0 }
    )

    @Test
    fun testDoubleNegZeroValue() = testDoubleConstant(
        symbolicOperation = { context.mkFpZero(signBit = true, context.mkFp64Sort()) },
        concreteOperation = { -0.0 }
    )

    @Test
    fun testFloatNaNValue() = testFloatConstant(
        symbolicOperation = { context.mkFpNaN(context.mkFp32Sort()) },
        concreteOperation = { Float.NaN }
    )

    @Test
    fun testFloatPosInfValue() = testFloatConstant(
        symbolicOperation = { context.mkFpInf(signBit = false, context.mkFp32Sort()) },
        concreteOperation = { Float.POSITIVE_INFINITY }
    )

    @Test
    fun testFloatNegInfValue() = testFloatConstant(
        symbolicOperation = { context.mkFpInf(signBit = true, context.mkFp32Sort()) },
        concreteOperation = { Float.NEGATIVE_INFINITY }
    )

    @Test
    fun testFloatPosZeroValue() = testFloatConstant(
        symbolicOperation = { context.mkFpZero(signBit = false, context.mkFp32Sort()) },
        concreteOperation = { +0.0f }
    )

    @Test
    fun testFloatNegZeroValue() = testFloatConstant(
        symbolicOperation = { context.mkFpZero(signBit = true, context.mkFp32Sort()) },
        concreteOperation = { -0.0f }
    )

    @Test
    fun testHalfNaNValue() = testHalfConstant(
        symbolicOperation = { context.mkFpNaN(context.mkFp16Sort()) },
        concreteOperation = { Float.NaN }
    )

    @Test
    fun testHalfPosInfValue() = testHalfConstant(
        symbolicOperation = { context.mkFpInf(signBit = false, context.mkFp16Sort()) },
        concreteOperation = { Float.POSITIVE_INFINITY }
    )

    @Test
    fun testHalfNegInfValue() = testHalfConstant(
        symbolicOperation = { context.mkFpInf(signBit = true, context.mkFp16Sort()) },
        concreteOperation = { Float.NEGATIVE_INFINITY }
    )

    @Test
    fun testHalfPosZeroValue() = testHalfConstant(
        symbolicOperation = { context.mkFpZero(signBit = false, context.mkFp16Sort()) },
        concreteOperation = { +0.0f }
    )

    @Test
    fun testHalfNegZeroValue() = testHalfConstant(
        symbolicOperation = { context.mkFpZero(signBit = true, context.mkFp16Sort()) },
        concreteOperation = { -0.0f }
    )

    @Test
    fun testMatchSolverInternalNaN() = compareWithSolverInternal(::compareNaNWithSolverInternal)

    @Test
    fun testMatchSolverInternalPosInf() = compareWithSolverInternal(::comparePosInfWithSolverInternal)

    @Test
    fun testMatchSolverInternalNegInf() = compareWithSolverInternal(::compareNegInfWithSolverInternal)

    @Test
    fun testMatchSolverInternalPosZero() = compareWithSolverInternal(::comparePosZeroWithSolverInternal)

    @Test
    fun testMatchSolverInternalNegZero() = compareWithSolverInternal(::compareNegZeroWithSolverInternal)

    private fun testCompare(
        symbolicOperation: (KExpr<KFp64Sort>, KExpr<KFp64Sort>) -> KExpr<KBoolSort>,
        concreteOperation: (Double, Double) -> Boolean
    ): Unit = with(context) {
        val fst = Random.nextDouble()
        val snd = Random.nextDouble()

        val fstFp = fst.toFp() as KFp64Value
        val sndFp = snd.toFp() as KFp64Value

        val fstVariable = mkBoolSort().mkConst("fstVariable")
        val sndVariable = mkBoolSort().mkConst("sndVariable")

        solver.assert(symbolicOperation(fstFp, sndFp) eq fstVariable)
        solver.assert(symbolicOperation(sndFp, fstFp) eq sndVariable)

        solver.check()
        with(solver.model()) {
            assertEquals(concreteOperation(fst, snd), eval(fstVariable) is KTrue)
            assertEquals(concreteOperation(snd, fst), eval(sndVariable) is KTrue)
        }
    }

    private fun testPredicate(
        symbolicPredicate: (KExpr<KFp64Sort>) -> KExpr<KBoolSort>,
        concreteOperation: (Double) -> Boolean
    ): Unit = with(context) {
        val values = listOf(Double.NaN, +0.0, -0.0, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 1.0.pow(-126))
        val fpValues = values.map { it.toFp() as KFp64Value }
        val variables = values.indices.map { mkBoolSort().mkConst("variable$it") }

        val valuesWithVariables = fpValues.zip(variables)
        valuesWithVariables.forEach { (value, variable) ->
            solver.assert(symbolicPredicate(value) eq variable)
        }

        solver.check()
        with(solver.model()) {
            valuesWithVariables.forEach { (fp, variable) ->
                assertEquals(concreteOperation(fp.value), eval(variable) is KTrue)
            }
        }
    }

    private fun compareWithSolverInternal(check: (KFpSort) -> Unit) {
        val sorts = with(context) {
            val fpCustom = mkFpSort(5u, 10u)
            listOf(fp16Sort, fp32Sort, fp64Sort, mkFp128Sort(), fpCustom)
        }
        sorts.forEach(check)
    }

    private fun <S : KFpSort> compareNaNWithSolverInternal(sort: S) = compareWithSolverInternal(
        sort = sort,
        operation = context::mkFpNaN,
        solverInternal = { mkFPNaN(it) }
    )

    private fun <S : KFpSort> comparePosInfWithSolverInternal(sort: S) = compareWithSolverInternal(
        sort = sort,
        operation = { context.mkFpInf(sort = sort, signBit = false) },
        solverInternal = { mkFPInf(it, false) }
    )

    private fun <S : KFpSort> compareNegInfWithSolverInternal(sort: S) = compareWithSolverInternal(
        sort = sort,
        operation = { context.mkFpInf(sort = sort, signBit = true) },
        solverInternal = { mkFPInf(it, true) }
    )

    private fun <S : KFpSort> comparePosZeroWithSolverInternal(sort: S) = compareWithSolverInternal(
        sort = sort,
        operation = { context.mkFpZero(sort = sort, signBit = false) },
        solverInternal = { mkFPZero(it, false) }
    )

    private fun <S : KFpSort> compareNegZeroWithSolverInternal(sort: S) = compareWithSolverInternal(
        sort = sort,
        operation = { context.mkFpZero(sort = sort, signBit = true) },
        solverInternal = { mkFPZero(it, true) }
    )

    private fun <S : KFpSort> compareWithSolverInternal(
        sort: S,
        operation: (S) -> KExpr<S>,
        solverInternal: Context.(FPSort) -> Expr<*>
    ) = KZ3Context(context).use { solverInternalCtx ->
        val internalizer = KZ3ExprInternalizer(context, solverInternalCtx)

        val solverInternalSort = with(internalizer) { sort.internalizeSort() }
        val solverInternalExpr = solverInternal(
            solverInternalCtx.nativeContext,
            solverInternalCtx.nativeContext.wrapAST(solverInternalSort) as FPSort
        )

        val ksmtExpr = operation(sort)
        val ksmtInternalizedExpr = with(internalizer) { ksmtExpr.internalizeExpr() }

        val check = solverInternalCtx.nativeContext.mkEq(
            solverInternalExpr,
            solverInternalCtx.nativeContext.wrapAST(ksmtInternalizedExpr) as Expr<*>
        )

        val checker = solverInternalCtx.nativeContext.mkSolver()
        checker.add(check)
        val status = checker.check()

        val message = """
            Representation mismatch for $sort
            Expected: $solverInternalExpr
            Actual: $ksmtInternalizedExpr
            Actual KSMT: $ksmtExpr
        """.trimIndent()

        assertEquals(Status.SATISFIABLE, status, message)
    }

    @Test
    fun testLessOrEqualExpr() = testCompare(context::mkFpLessOrEqualExpr) { a: Double, b: Double -> a <= b }

    @Test
    fun testLessExpr() = testCompare(context::mkFpLessExpr) { a: Double, b: Double -> a < b }

    @Test
    fun testGreaterExpr() = testCompare(context::mkFpGreaterExpr) { a: Double, b: Double -> a > b }

    @Test
    fun testGreaterOrEqualExpr() = testCompare(context::mkFpGreaterOrEqualExpr) { a: Double, b: Double -> a >= b }

    @Test
    fun testEqualExpr() = testCompare(context::mkFpEqualExpr) { a: Double, b: Double -> abs(a - b) <= DELTA }

    @Test
    fun testIsZero() = testPredicate(context::mkFpIsZeroExpr) { value: Double -> value == +0.0 || value == -0.0 }

    @Test
    fun testIsInfinite() = testPredicate(context::mkFpIsInfiniteExpr) { value: Double -> value.isInfinite() }

    companion object {
        const val DELTA = 1e-15
    }
}
