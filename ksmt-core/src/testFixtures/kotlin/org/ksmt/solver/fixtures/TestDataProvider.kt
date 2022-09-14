package org.ksmt.solver.fixtures

import java.nio.file.Path
import java.nio.file.Paths
import kotlin.io.path.listDirectoryEntries

object TestDataProvider {
    fun testDataLocation(): Path = this::class.java.classLoader
        .getResource("testData")
        ?.toURI()
        ?.let { Paths.get(it) }
        ?: error("No test data")

    fun testData(): List<Path> {
        val testData = testDataLocation()
        return testData.listDirectoryEntries("*.smt2").sorted()
    }
}
