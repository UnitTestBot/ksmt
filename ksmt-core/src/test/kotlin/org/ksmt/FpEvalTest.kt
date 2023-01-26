package org.ksmt

import org.junit.jupiter.api.Assumptions
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import org.ksmt.expr.KExpr
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.utils.FpUtils.isInfinity
import org.ksmt.utils.FpUtils.isNan
import org.ksmt.utils.FpUtils.isNegative
import org.ksmt.utils.FpUtils.isZero
import org.ksmt.utils.uncheckedCast
import kotlin.test.assertEquals

@Execution(ExecutionMode.CONCURRENT)
class FpEvalTest : ExpressionEvalTest() {

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpAbs(exponent: Int, significand: Int) = testOperation(exponent, significand, KContext::mkFpAbsExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpNegation(exponent: Int, significand: Int) =
        testOperation(exponent, significand, KContext::mkFpNegationExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpAdd(exponent: Int, significand: Int) = testOperation(exponent, significand, KContext::mkFpAddExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpSub(exponent: Int, significand: Int) = testOperation(exponent, significand, KContext::mkFpSubExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpMul(exponent: Int, significand: Int) = testOperation(exponent, significand, KContext::mkFpMulExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpDiv(exponent: Int, significand: Int) = testOperation(exponent, significand, KContext::mkFpDivExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpSqrt(exponent: Int, significand: Int) = testOperation(exponent, significand, KContext::mkFpSqrtExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpRoundToIntegral(exponent: Int, significand: Int) =
        testOperation(exponent, significand, KContext::mkFpRoundToIntegralExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpMin(exponent: Int, significand: Int) = runTest(exponent, significand) { sort: KFpSort, checker ->
        randomFpValuesNoNegZero(sort).take(30).forEach { a ->
            randomFpValuesNoNegZero(sort).take(30).forEach { b ->
                val expr = mkFpMinExpr(a, b)
                checker.check(expr) { "$a, $b" }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpMax(exponent: Int, significand: Int) = runTest(exponent, significand) { sort: KFpSort, checker ->
        randomFpValuesNoNegZero(sort).take(30).forEach { a ->
            randomFpValuesNoNegZero(sort).take(30).forEach { b ->
                val expr = mkFpMaxExpr(a, b)
                checker.check(expr) { "$a, $b" }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpLessOrEqual(exponent: Int, significand: Int) =
        testOperation(exponent, significand, KContext::mkFpLessOrEqualExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpLess(exponent: Int, significand: Int) = testOperation(exponent, significand, KContext::mkFpLessExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpGreaterOrEqual(exponent: Int, significand: Int) =
        testOperation(exponent, significand, KContext::mkFpGreaterOrEqualExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpGreater(exponent: Int, significand: Int) = testOperation(exponent, significand, KContext::mkFpGreaterExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpEqual(exponent: Int, significand: Int) = testOperation(exponent, significand, KContext::mkFpEqualExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpIsNormal(exponent: Int, significand: Int) =
        testOperation(exponent, significand, KContext::mkFpIsNormalExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpIsSubnormal(exponent: Int, significand: Int) =
        testOperation(exponent, significand, KContext::mkFpIsSubnormalExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpIsZero(exponent: Int, significand: Int) = testOperation(exponent, significand, KContext::mkFpIsZeroExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpIsInfinite(exponent: Int, significand: Int) =
        testOperation(exponent, significand, KContext::mkFpIsInfiniteExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpIsNaN(exponent: Int, significand: Int) = testOperation(exponent, significand, KContext::mkFpIsNaNExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpIsNegative(exponent: Int, significand: Int) =
        testOperation(exponent, significand, KContext::mkFpIsNegativeExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpIsPositive(exponent: Int, significand: Int) =
        testOperation(exponent, significand, KContext::mkFpIsPositiveExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpToIEEEBv(exponent: Int, significand: Int) =
        testOperationNoNan(exponent, significand, KContext::mkFpToIEEEBvExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpToReal(exponent: Int, significand: Int) {
        Assumptions.assumeTrue(exponent <= Int.SIZE_BITS) {
            "Exponent may contain values witch are too large to be represented as Real"
        }
        runTest(exponent, significand) { sort: KFpSort, checker ->
            randomFpValues(sort).filterNot { it.isNan() || it.isInfinity() }.take(100).forEach { value ->
                val expr = mkFpToRealExpr(value)
                checker.check(expr) { "$value" }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpFromReal(exponent: Int, significand: Int) = runTest(exponent, significand) { sort: KFpSort, checker ->
        roundingModeValues().forEach { rm ->
            randomRealValues().take(100).forEach { value ->
                val expr = mkRealToFpExpr(sort, rm, value)
                checker.check(expr) { "$rm, $value" }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testBvToFp(exponent: Int, significand: Int) = runTest(exponent, significand) { sort: KFpSort, checker ->
        val bvSorts = listOf(bv1Sort, bv8Sort, bv16Sort, bv32Sort, bv64Sort, mkBvSort(37u))
        roundingModeValues().forEach { rm ->
            bvSorts.forEach { bvSort ->
                randomBvValues(bvSort).forEach { value ->
                    val signed = mkBvToFpExpr(sort, rm, value, signed = true)
                    val unsigned = mkBvToFpExpr(sort, rm, value, signed = false)

                    checker.check(signed) { "Signed: $rm, $value" }
                    checker.check(unsigned) { "Unsigned: $rm, $value" }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpToBv(exponent: Int, significand: Int) = runTest(exponent, significand) { sort: KFpSort, checker ->
        val bvSorts = listOf(bv1Sort, bv8Sort, bv16Sort, bv32Sort, bv64Sort, mkBvSort(37u))
        roundingModeValues().forEach { rm ->
            randomFpValues(sort).take(100).forEach { value ->
                bvSorts.forEach { toBv ->
                    val signed = mkFpToBvExpr(rm, value, toBv.sizeBits.toInt(), isSigned = true)
                    val unsigned = mkFpToBvExpr(rm, value, toBv.sizeBits.toInt(), isSigned = false)

                    checker.check(signed) { "Signed: $rm, $value, $toBv" }
                    checker.check(unsigned) { "Unsigned: $rm, $value, $toBv" }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpFromBv(exponent: Int, significand: Int) = runTest(exponent, significand) { sort: KFpSort, checker ->
        val signValues = listOf(mkBv(0, bv1Sort), mkBv(1, bv1Sort))
        val exponentSort = mkBvSort(sort.exponentBits)
        val significandSort = mkBvSort(sort.significandBits - 1u)
        randomBvValues(exponentSort).forEach { exponent ->
            randomBvValues(significandSort).forEach { significand ->
                signValues.forEach { sign ->
                    val expr = mkFpFromBvExpr<KFpSort>(sign, exponent, significand)
                    checker.check(expr) { "$sign, $exponent, $significand" }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpToFp(exponent: Int, significand: Int) = runTest(exponent, significand) { sort: KFpSort, checker ->
        roundingModeValues().forEach { rm ->
            randomFpValues(sort).take(100).forEach { value ->
                fpSortsToTest.forEach { toSort ->
                    val expr = mkFpToFpExpr(mkFpSort(toSort.exponentBits, toSort.significandBits), rm, value)
                    checker.check(expr) { "$rm, $value, $toSort" }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpRem(exponent: Int, significand: Int) {
        val isFp32 = exponent.toUInt() == KFp32Sort.exponentBits && significand.toUInt() == KFp32Sort.significandBits
        val isFp64 = exponent.toUInt() == KFp64Sort.exponentBits && significand.toUInt() == KFp64Sort.significandBits
        Assumptions.assumeTrue(isFp32 || isFp64) {
            "Fp rem eval is implemented only for Fp32 and Fp64"
        }
        testOperation(exponent, significand, KContext::mkFpRemExpr)
    }

    @Disabled // We have no eval fo FMA
    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpFusedMulAdd(exponent: Int, significand: Int) = runTest(exponent, significand) { sort: KFpSort, checker ->
        roundingModeValues().forEach { rm ->
            randomFpValues(sort).forEach { a ->
                randomFpValues(sort).forEach { b ->
                    randomFpValues(sort).forEach { c ->
                        val expr = mkFpFusedMulAddExpr(rm, a, b, c)
                        checker.check(expr) { "$rm, $a, $b, $c" }
                    }
                }
            }
        }
    }

    @Test
    fun testFp16Creation(): Unit = with(KContext()) {
        val sort = fp16Sort
        for (exponent in 0 until (1 shl sort.exponentBits.toInt())) {
            val exponentBv = mkBv(exponent, sort.exponentBits)
            for (significand in 0 until (1 shl (sort.significandBits - 1u).toInt())) {
                val significandBv = mkBv(significand, sort.significandBits - 1u)
                val value = mkFpBiased(
                    sort = sort,
                    signBit = false,
                    biasedExponent = exponentBv,
                    significand = significandBv
                )
                assertEquals(exponentBv, value.biasedExponent)
                if (!value.isNan()) {
                    assertEquals(significandBv, value.significand)
                }
            }
        }
    }

    @JvmName("testOperationUnary")
    private fun <S : KFpSort, T : KSort> testOperation(
        exponent: Int,
        significand: Int,
        operation: KContext.(KExpr<S>) -> KExpr<T>
    ) = runTest(exponent, significand) { sort: S, checker ->
        randomFpValues(sort).take(100).forEach { value ->
            val expr = operation(value)
            checker.check(expr) { "$value" }
        }
    }

    @JvmName("testOperationUnaryNoNan")
    private fun <S : KFpSort, T : KSort> testOperationNoNan(
        exponent: Int,
        significand: Int,
        operation: KContext.(KExpr<S>) -> KExpr<T>
    ) = runTest(exponent, significand) { sort: S, checker ->
        randomFpValues(sort).filterNot { it.isNan() }.take(100).forEach { value ->
            val expr = operation(value)
            checker.check(expr) { "$value" }
        }
    }

    @JvmName("testOperationBinary")
    private fun <S : KFpSort, T : KSort> testOperation(
        exponent: Int,
        significand: Int,
        operation: KContext.(KExpr<S>, KExpr<S>) -> KExpr<T>
    ) = runTest(exponent, significand) { sort: S, checker ->
        randomFpValues(sort).take(60).forEach { a ->
            randomFpValues(sort).take(60).forEach { b ->
                val expr = operation(a, b)
                checker.check(expr) { "$a, $b" }
            }
        }
    }

    @JvmName("testOperationUnaryRm")
    private fun <S : KFpSort, T : KSort> testOperation(
        exponent: Int,
        significand: Int,
        operation: KContext.(KExpr<KFpRoundingModeSort>, KExpr<S>) -> KExpr<T>
    ) = runTest(exponent, significand) { sort: S, checker ->
        roundingModeValues().forEach { rm ->
            randomFpValues(sort).take(100).forEach { value ->
                val expr = operation(rm, value)
                checker.check(expr) { "$rm, $value" }
            }
        }
    }

    @JvmName("testOperationBinaryRm")
    private fun <S : KFpSort, T : KSort> testOperation(
        exponent: Int,
        significand: Int,
        operation: KContext.(KExpr<KFpRoundingModeSort>, KExpr<S>, KExpr<S>) -> KExpr<T>
    ) = runTest(exponent, significand) { sort: S, checker ->
        roundingModeValues().forEach { rm ->
            randomFpValues(sort).take(60).forEach { a ->
                randomFpValues(sort).take(60).forEach { b ->
                    val expr = operation(rm, a, b)
                    checker.check(expr) { "$rm, $a, $b" }
                }
            }
        }
    }

    private fun <S : KFpSort> runTest(
        exponent: Int,
        significand: Int,
        test: KContext.(S, TestRunner) -> Unit
    ) = runTest(
        mkSort = { mkFpSort(exponent.toUInt(), significand.toUInt()).uncheckedCast() },
        test = test.uncheckedCast()
    )

    private fun <S : KFpSort> KContext.randomFpValuesNoNegZero(sort: S) =
        randomFpValues(sort).filterNot { it.isZero() && it.isNegative() }

    companion object {
        private val fpSortsToTest by lazy {
            val context = KContext()
            val smallCustomFp = context.mkFpSort(5u, 5u)
            val middleCustomFp = context.mkFpSort(17u, 17u)
            val largeCustomFp = context.mkFpSort(50u, 50u)

            listOf(
                context.fp16Sort,
                context.fp32Sort,
                context.fp64Sort,
                context.mkFp128Sort(),
                smallCustomFp,
                middleCustomFp,
                largeCustomFp
            )
        }

        private val fpSizesToTest by lazy {
            fpSortsToTest.map { it.exponentBits.toInt() to it.significandBits.toInt() }
        }

        @JvmStatic
        fun fpSizes() = fpSizesToTest.map { Arguments.of(it.first, it.second) }
    }
}
