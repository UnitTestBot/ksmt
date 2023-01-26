package org.ksmt

import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.expr.KInterpretedConstant
import org.ksmt.expr.rewrite.simplify.KExprSimplifier
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.utils.FpUtils.isNan
import org.ksmt.utils.FpUtils.isNegative
import org.ksmt.utils.FpUtils.isZero
import org.ksmt.utils.uncheckedCast
import kotlin.test.assertEquals
import kotlin.test.assertTrue

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
    fun testFpRem(exponent: Int, significand: Int) = testOperation(exponent, significand, KContext::mkFpRemExpr)

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
    fun testFpToReal(exponent: Int, significand: Int) = testOperation(exponent, significand, KContext::mkFpToRealExpr)

    @ParameterizedTest
    @MethodSource("fpSizes")
    fun testFpToIEEEBv(exponent: Int, significand: Int) =
        testOperationNoNan(exponent, significand, KContext::mkFpToIEEEBvExpr)

//    @ParameterizedTest
//    @MethodSource("fpSizes")
//    fun testFpToBv(exponent: Int, significand: Int) = testOperation(exponent, significand, KContext::mkFpToBvExpr)

//    @ParameterizedTest
//    @MethodSource("fpSizes")
//    fun testFpFromBv(exponent: Int, significand: Int) = testOperation(exponent, significand, KContext::mkFpFromBvExpr)

//    @ParameterizedTest
//    @MethodSource("fpSizes")
//    fun testFpToFp(exponent: Int, significand: Int) = testOperation(exponent, significand, KContext::mkFpToFpExpr)

//    @ParameterizedTest
//    @MethodSource("fpSizes")
//    fun testFpFusedMulAdd(exponent: Int, significand: Int) =
//        testOperation(exponent, significand, KContext::mkFpFusedMulAddExpr)


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
        randomFpValues(sort).forEach { value ->
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
        randomFpValues(sort).filterNot { it.isNan() }.forEach { value ->
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
            randomFpValues(sort).forEach { value ->
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
    ) {
        val ctx = KContext()
        val sort: S = ctx.mkFpSort(exponent.toUInt(), significand.toUInt()).uncheckedCast()
        KZ3Solver(ctx).use { solver ->
            val checker = TestRunner(ctx, solver)
            ctx.test(sort, checker)
        }
    }

    private class TestRunner(
        private val ctx: KContext,
        private val solver: KSolver<*>
    ) {

        fun <T : KSort> check(expr: KExpr<T>, printArgs: () -> String) {
            val expectedValue = solverValue(expr)
            val actualValue = evalFpExpr(expr)

            assertEquals(expectedValue, actualValue, printArgs())

            val decl = (expectedValue as KApp<*, *>).decl
            val declValue = decl.apply(emptyList())
            assertEquals(expectedValue, declValue)
        }

        private fun <T : KSort> solverValue(expr: KExpr<T>): KExpr<T> =
            withSolverScope { solver ->
                with(ctx) {
                    val valueVar = mkFreshConst("v", expr.sort)
                    solver.assert(valueVar eq expr)
                    assertEquals(KSolverStatus.SAT, solver.check())
                    val value = solver.model().eval(valueVar)
                    assertTrue(value is KInterpretedConstant)
                    value
                }
            }


        private fun <T : KSort> evalFpExpr(expr: KExpr<T>): KExpr<T> {
            val evaluator = KExprSimplifier(ctx)
            return evaluator.apply(expr)
        }

        private fun <T> withSolverScope(block: (KSolver<*>) -> T): T = try {
            solver.push()
            block(solver)
        } finally {
            solver.pop()
        }
    }

    fun <S : KFpSort> KContext.randomFpValuesNoNegZero(sort: S) =
        randomFpValues(sort).filterNot { it.isZero() && it.isNegative() }

    companion object {
        private val fpSizesToTest by lazy {
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
            ).map { it.exponentBits.toInt() to it.significandBits.toInt() }
        }

        @JvmStatic
        fun fpSizes() = fpSizesToTest.map { Arguments.of(it.first, it.second) }
    }
}
