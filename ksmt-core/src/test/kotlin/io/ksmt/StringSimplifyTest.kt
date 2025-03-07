package io.ksmt

import io.ksmt.expr.KExpr
import io.ksmt.sort.KSort
import io.ksmt.sort.KStringSort
import io.ksmt.utils.StringUtils.sequentialEngAlphabetChars
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode

@Execution(ExecutionMode.CONCURRENT)
class StringSimplifyTest: ExpressionSimplifyTest() {

    @Test
    fun testStringLen() = testOperation(KContext::mkStringLen, KContext::mkStringLenNoSimplify)

    @Test
    fun testSuffixOf() = testOperation(KContext::mkStringSuffixOf, KContext::mkStringSuffixOfNoSimplify) {
        listOf(
            "prefixsuffix".expr,
            "prefix".expr,
            "suffix".expr
        )
    }

    @Test
    fun testPrefixOf() = testOperation(KContext::mkStringPrefixOf, KContext::mkStringPrefixOfNoSimplify) {
        listOf(
            "prefixsuffix".expr,
            "prefix".expr,
            "suffix".expr
        )
    }

    @Test
    fun testStringLt() = testOperation(KContext::mkStringLt, KContext::mkStringLtNoSimplify) {
        sequentialEngAlphabetChars()
    }

    @Test
    fun testStringLe() = testOperation(KContext::mkStringLe, KContext::mkStringLeNoSimplify) {
        sequentialEngAlphabetChars()
    }

    @Test
    fun testStringGt() = testOperation(KContext::mkStringGt, KContext::mkStringGtNoSimplify) {
        sequentialEngAlphabetChars()
    }

    @Test
    fun testStringGe() = testOperation(KContext::mkStringGe, KContext::mkStringGeNoSimplify) {
        sequentialEngAlphabetChars()
    }

    @Test
    fun testStringContais() = testOperation(KContext::mkStringContains, KContext::mkStringContainsNoSimplify) {
        listOf(
            "containsSomeString".expr,
            "Some".expr,
            "None".expr
        )
    }

    @Test
    fun testStringReplace() = testOperation(KContext::mkStringReplace, KContext::mkStringReplaceNoSimplify) {
        listOf(
            "containsSomeSomeString".expr,
            "Some".expr,
            "None".expr
        )
    }

    @Test
    fun testStringToCode() = testOperation(KContext::mkStringToCode, KContext::mkStringToCodeNoSimplify) {
        sequentialEngAlphabetChars()
    }

    @Test
    fun testStringToInt() = testOperation(KContext::mkStringToInt, KContext::mkStringToIntNoSimplify) {
        listOf("1".expr, "123".expr, "01".expr)
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

    @JvmName("testTernaryNotNested")
    private fun <T : KSort> testOperation(
        operation: KContext.(KExpr<KStringSort>, KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<T>,
        operationNoSimplify: KContext.(KExpr<KStringSort>, KExpr<KStringSort>, KExpr<KStringSort>) -> KExpr<T>,
        mkSpecialValues: KContext.(KStringSort) -> List<KExpr<KStringSort>> = { emptyList() }
    ) = runTest { sort: KStringSort, checker ->
        val a = mkConst("a", sort)
        val b = mkConst("b", sort)
        val c = mkConst("c", sort)
        val specialValues = mkSpecialValues(sort)

        (listOf(a) + specialValues).forEach { arg0 ->
            (listOf(b) + specialValues).forEach { arg1 ->
                (listOf(c) + specialValues).forEach { arg2 -> 
                    val unsimplifiedExpr = operationNoSimplify(arg0, arg1, arg2)
                    val simplifiedExpr = operation(arg0, arg1, arg2)
                    checker.check(unsimplifiedExpr = unsimplifiedExpr, simplifiedExpr = simplifiedExpr) {
                        "$arg0, $arg1, $arg2"
                    }
                }
            }
        }
    }

    private fun runTest(test: KContext.(KStringSort, TestRunner) -> Unit) = runTest(
        mkSort = { stringSort },
        test = test
    )
}
