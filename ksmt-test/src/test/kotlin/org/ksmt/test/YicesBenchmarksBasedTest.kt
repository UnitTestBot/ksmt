package org.ksmt.test

import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import java.nio.file.Path

class YicesBenchmarksBasedTest : BenchmarksBasedTest() {

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("yicesTestData")
    fun testConverter(name: String, samplePath: Path) =
        testConverter(name, samplePath) { assertions ->
            internalizeAndConvertYices(assertions)
        }

    companion object {
        @JvmStatic
        fun yicesTestData() = testData().skipUnsupportedTheories()

        private fun List<BenchmarkTestArguments>.skipUnsupportedTheories() =
            filterNot { "FP" in it.name }
    }
}
