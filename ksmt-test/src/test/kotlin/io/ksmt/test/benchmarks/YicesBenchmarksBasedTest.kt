package io.ksmt.test.benchmarks

import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import io.ksmt.solver.yices.KYicesSolver
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
        fun yicesTestData() = testData { skipUnsupportedTheories(it) && skipBadTestCases(it) }

        private fun skipUnsupportedTheories(name: String): Boolean =
            "QF" in name && "FP" !in name && "N" !in name

        private fun skipBadTestCases(name: String): Boolean {
            // Yices bug: returns an incorrect model
            if (name == "QF_UFBV_QF_UFBV_bv8_bv_eq_sdp_v4_ab_cti_max.smt2") return false

            // Yices bug: incorrect bv-add after bv-and on 64 bit bv (63 and 65 are correct).
            if (name == "QF_BV_countbits064.smt2") return false

            // Yices bug: same as in previous sample
            if (name == "QF_BV_nextpoweroftwo064.smt2") return false

            return true
        }
    }
}
