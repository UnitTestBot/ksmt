package io.ksmt.test.benchmarks

import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import io.ksmt.solver.bitwuzla.KBitwuzlaSolver
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
    @MethodSource("bitwuzlaModelConversionTestData")
    fun testModelConversion(name: String, samplePath: Path) =
        testModelConversion(name, samplePath, KBitwuzlaSolver::class)

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("bitwuzlaTestData")
    fun testSolver(name: String, samplePath: Path) =
        testSolver(name, samplePath, KBitwuzlaSolver::class)


    companion object {
        @JvmStatic
        fun bitwuzlaTestData() = testData(::skipUnsupportedTheories)

        @JvmStatic
        fun bitwuzlaModelConversionTestData() =
            bitwuzlaTestData()
                // Bitwuzla doesn't support models for quantified formulas
                .filter { it.name.startsWith("QF_") }
                .ensureNotEmpty()

        private fun skipUnsupportedTheories(name: String): Boolean {
            if ("LIA" in name || "LRA" in name || "LIRA" in name) return false
            if ("NIA" in name || "NRA" in name || "NIRA" in name) return false
            return true
        }
    }
}
