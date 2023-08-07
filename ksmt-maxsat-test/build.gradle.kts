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
}

val runBenchmarksBasedTests = project.booleanProperty("runBenchmarksBasedTests") ?: false

// use benchmarks from testData directory instead of downloading
val usePreparedBenchmarks = project.booleanProperty("usePreparedBenchmarks") ?: true

val testDataDir = projectDir.resolve("testData")
val unpackedTestDataDir = testDataDir.resolve("data")
val downloadedTestData = testDataDir.resolve("testData.zip")
val testDataRevision = project.stringProperty("testDataRevision") ?: "no-revision"

val downloadPreparedBenchmarksTestData = downloadPreparedMaxSATBenchmarkTestData(
    downloadPath = downloadedTestData,
    testDataPath = unpackedTestDataDir,
    testDataRevision = testDataRevision,
)

val preparedMaxSATBenchmarkTestData = usePreparedMaxSATBenchmarkTestData(unpackedTestDataDir)

val usePreparedTestData by tasks.registering {
    tasks.withType<ProcessResources>().forEach { it.enabled = false }
    if (!usePreparedBenchmarks) {
        dependsOn.add(downloadPreparedBenchmarksTestData)
    }

    finalizedBy(preparedMaxSATBenchmarkTestData)
}

tasks.withType<Test> {
    if (runBenchmarksBasedTests) {
        dependsOn.add(usePreparedTestData)
    }
}
