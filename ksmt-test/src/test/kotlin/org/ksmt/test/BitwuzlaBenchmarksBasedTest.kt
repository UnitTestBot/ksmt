package org.ksmt.test

import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import org.ksmt.solver.bitwuzla.KBitwuzlaSolver
import java.nio.file.Path


class BitwuzlaBenchmarksBasedTest : BenchmarksBasedTest() {

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("bitwuzlaTestData")
    fun testConverter(name: String, samplePath: Path) =
        testConverter(name, samplePath) { assertions ->
            internalizeAndConvertBitwuzla(assertions)
        }

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("bitwuzlaTestData")
    fun testModelConversion(name: String, samplePath: Path) =
        testModelConversion(name, samplePath, KBitwuzlaSolver::class)

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("bitwuzlaTestData")
    fun testSolver(name: String, samplePath: Path) =
        testSolver(name, samplePath, KBitwuzlaSolver::class)


    companion object {
        @JvmStatic
        fun bitwuzlaTestData() = testData()
            .skipUnsupportedTheories()
            .ensureNotEmpty()

        private fun List<BenchmarkTestArguments>.skipUnsupportedTheories() =
            filter { "QF_" in it.name }
                .filterNot { "LIA" in it.name || "LRA" in it.name }
                .filterNot { "NIA" in it.name || "NRA" in it.name }
    }
}
