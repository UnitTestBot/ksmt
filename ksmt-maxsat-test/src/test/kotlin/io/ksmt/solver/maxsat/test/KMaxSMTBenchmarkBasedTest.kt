package io.ksmt.solver.maxsat.test

import org.junit.jupiter.params.provider.Arguments
import java.nio.file.Path
import java.nio.file.Paths
import kotlin.io.path.listDirectoryEntries
import kotlin.io.path.relativeTo

interface KMaxSMTBenchmarkBasedTest {
    data class BenchmarkTestArguments(
        val name: String,
        val samplePath: Path,
    ) : Arguments {
        override fun get() = arrayOf(name, samplePath)
    }

    companion object {
        private fun testDataLocation(): Path =
            this::class.java.classLoader.getResource("testData")?.toURI()?.let { Paths.get(it) }
                ?: error("No test data")

        private fun prepareTestData(): List<BenchmarkTestArguments> {
            val testDataLocation = testDataLocation()
            return testDataLocation.listDirectoryEntries("*.smt2").sorted()
                .map {
                    BenchmarkTestArguments(
                        it.relativeTo(testDataLocation).toString(),
                        it,
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
