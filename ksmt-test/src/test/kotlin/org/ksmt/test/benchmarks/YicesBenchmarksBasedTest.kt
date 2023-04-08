package org.ksmt.test.benchmarks

import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import org.ksmt.solver.yices.KYicesSolver
import java.nio.file.Path

class YicesBenchmarksBasedTest : BenchmarksBasedTest() {

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("yicesTestData")
    fun testConverter(name: String, samplePath: Path) =
        testConverter(name, samplePath) { assertions ->
            internalizeAndConvertYices(assertions)
        }

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("yicesTestData")
    fun testModelConversion(name: String, samplePath: Path) =
        testModelConversion(name, samplePath, KYicesSolver::class)

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("yicesTestData")
    fun testSolver(name: String, samplePath: Path) =
        testSolver(name, samplePath, KYicesSolver::class)

    companion object {
        @JvmStatic
        fun yicesTestData() = testData
            .skipUnsupportedTheories()
            .skipBadTestCases()
            .ensureNotEmpty()

        private fun List<BenchmarkTestArguments>.skipUnsupportedTheories() =
            filterNot { "QF" !in it.name || "FP" in it.name || "N" in it.name }

        private fun List<BenchmarkTestArguments>.skipBadTestCases(): List<BenchmarkTestArguments> =
            /**
             * Yices bug: returns an incorrect model
             */
            filterNot { it.name == "QF_UFBV_QF_UFBV_bv8_bv_eq_sdp_v4_ab_cti_max.smt2" }
                // Yices bug: incorrect bv-add after bv-and on 64 bit bv (63 and 65 are correct).
                .filterNot { it.name == "QF_BV_countbits064.smt2" }
                // Yices bug: same as in previous sample
                .filterNot { it.name == "QF_BV_nextpoweroftwo064.smt2" }
    }
}
