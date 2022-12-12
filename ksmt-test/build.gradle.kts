plugins {
    id("org.ksmt.ksmt-base")
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(project(":ksmt-core"))
    implementation(project(":ksmt-z3"))
    implementation(project(":ksmt-bitwuzla"))
    implementation(project(":ksmt-runner"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.6.4")

    testImplementation("org.junit.jupiter", "junit-jupiter-params", "5.8.2")
}

val runBenchmarksBasedTests = project.booleanProperty("runBenchmarksBasedTests") ?: false

// Split all benchmarks test data on a several [benchmarkChunkMaxSize] sized chunks
val benchmarkChunkMaxSize = project.intProperty("benchmarkChunkSize") ?: Int.MAX_VALUE
// Use only a [benchmarkChunk] chunk of test data
val benchmarkChunk = project.intProperty("benchmarkChunk") ?: 0

// use benchmarks from testData directory instead of downloading
val usePreparedBenchmarks = project.booleanProperty("usePreparedBenchmarks") ?: true

// skip big benchmarks to achieve faster tests build and run time
val skipBigBenchmarks = project.booleanProperty("skipBigBenchmarks") ?: true

val smtLibBenchmarks = listOfNotNull(
    "QF_ALIA", // 12M
    "QF_AUFLIA", // 1.5M
    if (!skipBigBenchmarks) "QF_LIA" else null, // 5.2G
    "QF_LIRA", // 500K
    if (!skipBigBenchmarks) "QF_LRA" else null, // 1.1G
    if (!skipBigBenchmarks) "QF_UF" else null, // 100M
    "QF_UFLIA",// 1.3M
    if (!skipBigBenchmarks) "QF_UFLRA" else null, // 422M
    "ALIA", // 400K
    "AUFLIA", // 12M
    "AUFLIRA", // 19M
    "LIA", // 1.2M
    "LRA", // 4.5M
//    "UFLIA", // 64M // skipped, because systematically fails to download
    "UFLRA", // 276K
    "ABV", // 400K
    if (!skipBigBenchmarks) "AUFBV" else null, // 1.2G
    if (!skipBigBenchmarks) "BV" else null, // 847M
    "QF_ABV", // 253M
    if (!skipBigBenchmarks) "QF_BV" else null,// 12.3G
    "ABVFP", // 276K
    "ABVFPLRA", // 246K
    "AUFBVFP", // 14M
    "BVFP", // 400K
    "BVFPLRA", // 500K
    "FP", // 1M
    "FPLRA", // 700K
    "QF_ABVFP", // 13M
    "QF_ABVFPLRA", // 300K
    "QF_AUFBVFP", // 200K
    "QF_BVFP", // 7M
    "QF_BVFPLRA", // 300K
    "QF_FPLRA", // 250K
    "QF_FP", // 30M
)

val testDatDir = projectDir.resolve("testData")

val smtLibBenchmarkTestDataDownload = smtLibBenchmarks.map { mkSmtLibBenchmarkTestData(it) }
val smtLibBenchmarkTestDataPrepared = usePreparedSmtLibBenchmarkTestData(testDatDir)

val prepareTestData by tasks.registering {
    tasks.withType<ProcessResources>().forEach { it.enabled = false }
    if (usePreparedBenchmarks) {
        dependsOn.add(smtLibBenchmarkTestDataPrepared)
    } else {
        dependsOn.addAll(smtLibBenchmarkTestDataDownload)
    }
}

val testDataRevision = project.stringProperty("testDataRevision") ?: "no-revision"
val downloadPreparedBenchmarksTestData = downloadPreparedSmtLibBenchmarkTestData(testDatDir, testDataRevision)

tasks.withType<Test> {
    enabled = runBenchmarksBasedTests
    if (runBenchmarksBasedTests) {
        dependsOn.add(prepareTestData)
    }
    environment("benchmarkChunkMaxSize", benchmarkChunkMaxSize)
    environment("benchmarkChunk", benchmarkChunk)
}
