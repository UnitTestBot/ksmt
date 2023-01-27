package org.ksmt.test

import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.expr.rewrite.simplify.KExprSimplifier
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.sort.KSort
import java.nio.file.Path

class SimplifierBenchmarksBasedTest : BenchmarksBasedTest() {

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("simplifierTestData")
    fun testSimplifier(name: String, samplePath: Path) =
        testConverter(name, samplePath) { assertions ->
            val simplifier = KExprSimplifier(ctx)
            val simplified = assertions.map { simplifier.apply(it) }
            simplified.forEach { ContextConsistencyChecker(ctx).apply(it) }
            simplified
        }

    companion object {
        @JvmStatic
        fun simplifierTestData() = testData
    }

    class ContextConsistencyChecker(ctx: KContext) : KNonRecursiveTransformer(ctx) {
        override fun <T : KSort, A : KSort> transformApp(expr: KApp<T, A>): KExpr<T> = with(ctx) {
            check(expr === expr.decl.apply(expr.args)) {
                "Context is inconsistent"
            }
            return super.transformApp(expr)
        }
    }
}
