package org.ksmt.test

import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import org.ksmt.solver.cvc5.KCvc5Solver
import java.nio.file.Path

class Cvc5BenchmarksBasedTest : BenchmarksBasedTest() {

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("cvc5TestData")
    fun testSolver(name: String, samplePath: Path) = testSolver(name, samplePath, KCvc5Solver::class)

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("cvc5TestData")
    fun testConverter(name: String, samplePath: Path) = testConverter(name, samplePath) { assertions ->
            internalizeAndConvertCvc5(assertions)
        }

    companion object {
        @JvmStatic
        fun cvc5TestData() = testData
    }
}
