package org.ksmt.test

import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import org.ksmt.expr.rewrite.simplify.KExprSimplifier
import java.nio.file.Path

class SimplifierBenchmarksBasedTest : BenchmarksBasedTest() {

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("simplifierTestData")
    fun testSimplifier(name: String, samplePath: Path) =
        testConverter(name, samplePath) { assertions ->
            val simplifier = KExprSimplifier(ctx)
            assertions.map { simplifier.apply(it) }
        }

    companion object {
        @JvmStatic
        fun simplifierTestData() = testData()
    }
}
