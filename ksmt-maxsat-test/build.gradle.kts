plugins {
    id("io.ksmt.ksmt-base")
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(project(":ksmt-core"))
    implementation(project(":ksmt-maxsat"))

    testImplementation("org.junit.jupiter:junit-jupiter-api:5.8.2")
    testImplementation("org.junit.jupiter", "junit-jupiter-params", "5.8.2")
    testImplementation(project(":ksmt-z3"))
    testImplementation(project(":ksmt-test"))
    testImplementation(project(":ksmt-runner"))
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.6.4")
}

val maxSmtBenchmarks = listOfNotNull(
    "QF_ABVFP", // 8.84M
    "QF_BV-1", // 17.2M
    "QF_BV-2", // 20.8М
)

val runMaxSMTBenchmarkBasedTests = project.booleanProperty("runMaxSMTBenchmarkBasedTests") ?: false
val runMaxSATBenchmarkBasedTests = project.booleanProperty("runMaxSATBenchmarkBasedTests") ?: false

// use benchmarks from testData directory instead of downloading
val usePreparedMaxSMTBenchmarks = project.booleanProperty("usePreparedMaxSMTBenchmarks") ?: true
val usePreparedMaxSATBenchmark = project.booleanProperty("usePreparedMaxSATBenchmark") ?: true

val testDataDir = projectDir.resolve("testData")
val unpackedTestDataDir = testDataDir.resolve("data")
val downloadedTestData = testDataDir.resolve("testData.zip")

val maxSmtTestDataRevision = project.stringProperty("maxSmtTestDataRevision") ?: "no-revision"
val maxSatTestDataRevision = project.stringProperty("maxSatTestDataRevision") ?: "no-revision"

val usePreparedMaxSmtTestData = usePreparedMaxSmtBenchmarkTestData(unpackedTestDataDir)
val downloadPreparedMaxSmtBenchmarkTestData =
    maxSmtBenchmarks.map { maxSmtBenchmarkTestData(it, maxSmtTestDataRevision) }

val usePreparedMaxSMTTestData by tasks.registering {
    tasks.withType<ProcessResources>().forEach { it.enabled = false }
    if (usePreparedMaxSMTBenchmarks) {
        dependsOn.add(usePreparedMaxSmtTestData)
    } else {
        dependsOn.addAll(downloadPreparedMaxSmtBenchmarkTestData)
    }
}

tasks.withType<Test> {
    if (runMaxSMTBenchmarkBasedTests) {
        dependsOn.add(usePreparedMaxSMTTestData)
    }
}

val downloadPreparedMaxSATBenchmarkTestData = downloadMaxSATBenchmarkTestData(
    downloadPath = downloadedTestData,
    testDataPath = unpackedTestDataDir,
    testDataRevision = maxSatTestDataRevision,
)

val preparedMaxSATBenchmarkTestData = usePreparedMaxSATBenchmarkTestData(unpackedTestDataDir)

val usePreparedMaxSATTestData by tasks.registering {
    tasks.withType<ProcessResources>().forEach { it.enabled = false }
    if (!usePreparedMaxSATBenchmark) {
        dependsOn.add(downloadPreparedMaxSATBenchmarkTestData)
    }

    finalizedBy(preparedMaxSATBenchmarkTestData)
}

tasks.withType<Test> {
    if (runMaxSATBenchmarkBasedTests) {
        dependsOn.add(usePreparedMaxSATTestData)
    }
}
