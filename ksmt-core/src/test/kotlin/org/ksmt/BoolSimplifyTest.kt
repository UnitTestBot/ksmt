package org.ksmt

import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBoolSort
import org.ksmt.utils.mkConst
import kotlin.test.Test

@Execution(ExecutionMode.CONCURRENT)
class BoolSimplifyTest : ExpressionSimplifyTest() {

    @Test
    fun testNot() = testOperation(KContext::mkNot, KContext::mkNotNoSimplify)

    @Test
    fun testAnd() = testListOperation(KContext::mkAnd, KContext::mkAndNoSimplify)

    @Test
    fun testOr() = testListOperation(KContext::mkOr, KContext::mkOrNoSimplify)

    @Test
    fun testImplies() = testOperation(KContext::mkImplies, KContext::mkImpliesNoSimplify)

    @Test
    fun testXor() = testOperation(KContext::mkXor, KContext::mkXorNoSimplify)

    @Test
    fun testEq() = testOperation(KContext::mkEq, KContext::mkEqNoSimplify)

    @Test
    fun testDistinct() = testListOperation(KContext::mkDistinct, KContext::mkDistinctNoSimplify)

    @Test
    fun testIte() = runTest { _, checker ->
        val args = listOf(
            boolSort.mkConst("c"),
            boolSort.mkConst("t"),
            boolSort.mkConst("e"),
            mkNot(boolSort.mkConst("c")),
            trueExpr,
            falseExpr
        )
        val nestedIte = args.flatMap { c ->
            args.map { t ->
                mkIteNoSimplify(c, t, boolSort.mkConst("d"))
            }
        }

        args.forEach { c ->
            (args + nestedIte).forEach { t ->
                args.forEach { e ->
                    val unsimplifiedExpr = mkIteNoSimplify(c, t, e)
                    val simplifiedExpr = mkIte(c, t, e)
                    checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) {
                        "$c, $t, $e"
                    }
                }
            }
        }
    }

    @JvmName("testUnary")
    private fun testOperation(
        operation: KContext.(KExpr<KBoolSort>) -> KExpr<KBoolSort>,
        operationNoSimplify: KContext.(KExpr<KBoolSort>) -> KExpr<KBoolSort>,
    ) = runTest { _, checker ->
        val x = mkConst("x", boolSort)
        val args = listOf(x, operationNoSimplify(x)) + listOf(trueExpr, falseExpr)
        args.forEach { value ->
            val unsimplifiedExpr = operationNoSimplify(value)
            val simplifiedExpr = operation(value)
            checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) {
                "$value"
            }
        }
    }

    @JvmName("testBinaryNested")
    private fun testOperation(
        operation: KContext.(KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort>,
        operationNoSimplify: KContext.(KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort>,
    ) = runTest { _, checker ->
        val a = mkConst("a", boolSort)
        val b = mkConst("b", boolSort)
        val args = listOf(a, b, trueExpr, falseExpr)
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
    private fun testListOperation(
        operation: KContext.(List<KExpr<KBoolSort>>) -> KExpr<KBoolSort>,
        operationNoSimplify: KContext.(List<KExpr<KBoolSort>>) -> KExpr<KBoolSort>,
    ) = testOperation(
        { a, b -> operation(listOf(a, b)) },
        { a, b -> operationNoSimplify(listOf(a, b)) },
    )

    private fun runTest(test: KContext.(KBoolSort, TestRunner) -> Unit) = runTest(
        mkSort = { boolSort },
        test = test
    )
}
