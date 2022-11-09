package org.ksmt.test

import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import org.ksmt.KContext
import org.ksmt.solver.z3.KZ3SMTLibParser
import org.ksmt.solver.z3.KZ3Solver
import java.nio.file.Path
import java.util.concurrent.ConcurrentHashMap
import kotlin.io.path.Path
import kotlin.io.path.writeLines
import kotlin.system.measureNanoTime

class Z3ConverterBenchmark {

    //    @Disabled
    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("z3TestData")
    fun measureConversionTime(name: String, samplePath: Path) {
        try {
            with(KContext()) {
                val assertions = KZ3SMTLibParser(this).parse(samplePath)
                val conversionTime = KZ3Solver(this).use { solver ->
                    measureNanoTime {
                        assertions.forEach { solver.assert(it) }
                    }
                }
                timingData[name] = conversionTime
            }
        } catch (t: Throwable) {
            System.err.println(t.toString())
        }
    }

    companion object {
        private val timingData = ConcurrentHashMap<String, Long>()

        @JvmStatic
        fun z3TestData() = BenchmarksBasedTest.testData()

        @AfterAll
        @JvmStatic
        fun saveData() {
            val samples = timingData.size
            val wholeTime = timingData.values.sumOf { it.toBigInteger() }
            val averageTime = wholeTime / samples.toBigInteger()
            println("Samples: $samples")
            println("Whole time: $wholeTime")
            println("Average time: $averageTime")

            val header = "Sample name,Time ns"
            val detailedDataCsv = timingData.map { "${it.key},${it.value}" }

            Path("detailed_timing_data.csv")
                .writeLines(listOf(header) + detailedDataCsv)
        }
    }
}
