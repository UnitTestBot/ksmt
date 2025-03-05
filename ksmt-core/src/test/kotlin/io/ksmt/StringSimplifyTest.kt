package io.ksmt

import io.ksmt.expr.KExpr
import io.ksmt.sort.KSort
import io.ksmt.sort.KStringSort
import io.ksmt.utils.StringUtils.sequentialStringsForComparisons
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode

@Execution(ExecutionMode.CONCURRENT)
class StringSimplifyTest: ExpressionSimplifyTest() {

    @Test
    fun testStringLen() = testOperation(KContext::mkStringLen, KContext::mkStringLenNoSimplify)

    @Test
    fun testStringLt() = testOperation(KContext::mkStringLt, KContext::mkStringLtNoSimplify) {
        sequentialStringsForComparisons()
    }

    @Test
    fun testStringLe() = testOperation(KContext::mkStringLe, KContext::mkStringLeNoSimplify) {
        sequentialStringsForComparisons()
    }

    @Test
    fun testStringGt() = testOperation(KContext::mkStringGt, KContext::mkStringGtNoSimplify) {
        sequentialStringsForComparisons()
    }

    @Test
    fun testStringGe() = testOperation(KContext::mkStringGe, KContext::mkStringGeNoSimplify) {
        sequentialStringsForComparisons()
    }

    @JvmName("testUnary")
    private fun <T : KSort> testOperation(
        operation: KContext.(KExpr<KStringSort>) -> KExpr<T>,
        operationNoSimplify: KContext.(KExpr<KStringSort>) -> KExpr<T>,
        mkSpecialValues: KContext.(KStringSort) -> List<KExpr<KStringSort>> = { emptyList() },
    ) = runTest { sort: KStringSort, checker ->
        val x = mkConst("x", sort)
        val specialValues = mkSpecialValues(sort)

        (listOf(x) + specialValues).forEach { value ->
            val unsimplifiedExpr = operationNoSimplify(value)
            val simplifiedExpr = operation(value)
            checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) { "$value" }
        }
    }

    @JvmName("testBinaryNotNested")
    private fun <T : KSort> testOperation(
        operation: KContext.(KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<T>,
        operationNoSimplify: KContext.(KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<T>,
        mkSpecialValues: KContext.(KStringSort) -> List<KExpr<KStringSort>> = { emptyList() }
    ) = runTest { sort: KStringSort, checker ->
        val a = mkConst("a", sort)
        val b = mkConst("b", sort)
        val specialValues = mkSpecialValues(sort)

        (listOf(a) + specialValues).forEach { lhs ->
            (listOf(b) + specialValues).forEach { rhs ->
                val unsimplifiedExpr = operationNoSimplify(lhs, rhs)
                val simplifiedExpr = operation(lhs, rhs)
                checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) {
                    "$lhs, $rhs"
                }
            }
        }
    }

    private fun runTest(test: KContext.(KStringSort, TestRunner) -> Unit) = runTest(
        mkSort = { stringSort },
        test = test
    )
}
