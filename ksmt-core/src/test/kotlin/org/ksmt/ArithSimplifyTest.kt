package org.ksmt

import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.ksmt.expr.KExpr
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KSort
import org.ksmt.utils.uncheckedCast
import kotlin.test.Test

@Execution(ExecutionMode.CONCURRENT)
class ArithSimplifyTest : ExpressionSimplifyTest() {

    @Test
    fun testUnaryMinus() {
        testOperation(isInt = true, KContext::mkArithUnaryMinus, KContext::mkArithUnaryMinusNoSimplify) {
            listOf(mkArithUnaryMinus(mkConst("y", it)))
        }

        testOperation(isInt = false, KContext::mkArithUnaryMinus, KContext::mkArithUnaryMinusNoSimplify) {
            listOf(mkArithUnaryMinus(mkConst("y", it)))
        }
    }

    @Test
    fun testAdd() {
        testOperation(isInt = true, KContext::mkArithAdd, KContext::mkArithAddNoSimplify) {
            listOf(mkIntNum(0), mkIntNum(1)).uncheckedCast()
        }

        testOperation(isInt = false, KContext::mkArithAdd, KContext::mkArithAddNoSimplify) {
            listOf(mkRealNum(0), mkRealNum(1)).uncheckedCast()
        }
    }

    @Test
    fun testSub() {
        testOperation(isInt = true, KContext::mkArithSub, KContext::mkArithSubNoSimplify) {
            listOf(mkIntNum(0), mkIntNum(1)).uncheckedCast()
        }

        testOperation(isInt = false, KContext::mkArithSub, KContext::mkArithSubNoSimplify) {
            listOf(mkRealNum(0), mkRealNum(1)).uncheckedCast()
        }
    }

    @Test
    fun testMul() {
        testOperation(isInt = true, KContext::mkArithMul, KContext::mkArithMulNoSimplify) {
            listOf(mkIntNum(0), mkIntNum(1), mkIntNum(-1)).uncheckedCast()
        }

        testOperation(isInt = false, KContext::mkArithMul, KContext::mkArithMulNoSimplify) {
            listOf(mkRealNum(0), mkRealNum(1), mkRealNum(-1)).uncheckedCast()
        }
    }

    @Test
    fun testDiv() {
        testOperation(isInt = true, KContext::mkArithDiv, KContext::mkArithDivNoSimplify) {
            listOf(mkIntNum(0), mkIntNum(1), mkIntNum(-1)).uncheckedCast()
        }

        testOperation(isInt = false, KContext::mkArithDiv, KContext::mkArithDivNoSimplify) {
            listOf(mkRealNum(0), mkRealNum(1), mkRealNum(-1)).uncheckedCast()
        }
    }

    @Test
    fun testIntMod() {
        testOperation(isInt = true, KContext::mkIntMod, KContext::mkIntModNoSimplify) {
            listOf(mkIntNum(0), mkIntNum(1), mkIntNum(-1)).uncheckedCast()
        }
    }

    @Test
    fun testIntRem() {
        testOperation(isInt = true, KContext::mkIntRem, KContext::mkIntRemNoSimplify) {
            listOf(mkIntNum(0), mkIntNum(1), mkIntNum(-1)).uncheckedCast()
        }
    }

    @JvmName("testUnary")
    private fun <S : KArithSort, T : KSort> testOperation(
        isInt: Boolean,
        operation: KContext.(KExpr<S>) -> KExpr<T>,
        operationNoSimplify: KContext.(KExpr<S>) -> KExpr<T>,
        mkSpecialValues: KContext.(S) -> List<KExpr<S>> = { emptyList() }
    ) = runTest(isInt) { sort: S, checker ->
        val x = mkConst("x", sort)
        val args = listOf(x) + mkSpecialValues(sort)
        args.forEach { value ->
            val unsimplifiedExpr = operationNoSimplify(value)
            val simplifiedExpr = operation(value)
            checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) {
                "$value"
            }
        }
    }

    @JvmName("testBinaryNested")
    private fun <S : KArithSort> testOperation(
        isInt: Boolean,
        operation: KContext.(KExpr<S>, KExpr<S>) -> KExpr<S>,
        operationNoSimplify: KContext.(KExpr<S>, KExpr<S>) -> KExpr<S>,
        mkSpecialValues: KContext.(S) -> List<KExpr<S>> = { emptyList() }
    ) = runTest(isInt) { sort: S, checker ->
        val a = mkConst("a", sort)
        val b = mkConst("b", sort)
        val args = listOf(a, b) + mkSpecialValues(sort)
        val nestedOperation = args.flatMap { x -> args.map { y -> operationNoSimplify(x, y) } }

        args.forEach { lhs ->
            (args + nestedOperation).forEach { rhs ->
                val unsimplifiedExpr = operationNoSimplify(lhs, rhs)
                val simplifiedExpr = operation(lhs, rhs)
                checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) {
                    "$lhs, $rhs"
                }
            }
        }
    }

    @JvmName("testBinaryList")
    private fun <S : KArithSort> testOperation(
        isInt: Boolean,
        operation: KContext.(List<KExpr<S>>) -> KExpr<S>,
        operationNoSimplify: KContext.(List<KExpr<S>>) -> KExpr<S>,
        mkSpecialValues: KContext.(S) -> List<KExpr<S>> = { emptyList() }
    ) = testOperation(
        isInt,
        { a, b -> operation(listOf(a, b)) },
        { a, b -> operationNoSimplify(listOf(a, b)) },
        mkSpecialValues
    )

    private fun <T : KArithSort> runTest(isInt: Boolean, test: KContext.(T, TestRunner) -> Unit) = runTest(
        mkSort = { (if (isInt) intSort else realSort).uncheckedCast() },
        test = test
    )
}
