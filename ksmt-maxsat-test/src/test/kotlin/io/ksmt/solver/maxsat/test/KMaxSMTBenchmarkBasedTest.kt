package io.ksmt.solver.maxsat.test

import org.junit.jupiter.params.provider.Arguments
import java.nio.file.Path
import java.nio.file.Paths

interface KMaxSMTBenchmarkBasedTest {
    data class BenchmarkTestArguments(
        val name: String,
        val samplePath: Path,
    ) : Arguments {
        override fun get() = arrayOf(name, samplePath)
    }

    companion object {
        private fun testDataLocation(): Path =
            this::class.java.classLoader.getResource("maxSmtBenchmark")?.toURI()
                ?.let { Paths.get(it) }
                ?: error("No test data")

        private fun prepareTestData(): List<BenchmarkTestArguments> {
            val testDataLocation = testDataLocation()
            return testDataLocation.toFile().walkTopDown().filter { f -> f.isFile && f.extension == "smt2" }.toList()
                .sorted()
                .map {
                    BenchmarkTestArguments(
                        it.relativeTo(testDataLocation.toFile()).toString(),
                        it.toPath(),
                    )
                }
        }

        private val testData by lazy {
            prepareTestData()
        }

        @JvmStatic
        fun maxSMTTestData() = testData
    }
}
