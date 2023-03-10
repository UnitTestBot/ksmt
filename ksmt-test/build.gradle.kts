plugins {
    id("org.ksmt.ksmt-base")
    id("me.champeau.jmh") version "0.6.8"
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(project(":ksmt-core"))
    implementation(project(":ksmt-z3"))
    implementation(project(":ksmt-cvc5"))
    implementation(project(":ksmt-bitwuzla"))
    implementation(project(":ksmt-yices"))
    implementation(testFixtures(project(":ksmt-yices")))
    implementation(project(":ksmt-runner"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.6.4")
    implementation(kotlin("reflect"))

    testImplementation("org.junit.jupiter", "junit-jupiter-params", "5.8.2")

    jmh("org.openjdk.jmh:jmh-core:1.36")
    jmh("org.openjdk.jmh:jmh-generator-annprocess:1.36")
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
    "UFLIA", // 64M
    "UFLRA", // 276K
    if (!skipBigBenchmarks) "BV" else null, // 847M
    "QF_ABV", // 253M
    if (!skipBigBenchmarks) "QF_BV" else null,// 12.3G
    "ABVFP", // 276K
    "BVFP", // 400K
    "FP", // 1M
    "QF_ABVFP", // 13M
    "QF_BVFP", // 7M
    "QF_BVFPLRA", // 300K
    "QF_FPLRA", // 250K
    "QF_FP", // 30M
    if (!skipBigBenchmarks) "UFNIA" else null, //112M
    "UF", //47M
    "UFBV", //13M
    "NIA", //13K
    "NRA", //4.2M
    "AUFNIA", //3.1K
    "AUFNIRA", //4.6M
    "QF_UFNIA", //583K
    "QF_UFNRA", //186K
    "QF_UFBV", //37M
    if (!skipBigBenchmarks) "QF_NIA" else null, //281M
    "QF_NIRA", //139K
    if (!skipBigBenchmarks) "QF_NRA" else null, //182M
    "QF_ANIA", //2.1M
    "QF_AUFBV", //1.1M
    "QF_AUFNIA", //531K
)

val testDataDir = projectDir.resolve("testData")
val unpackedTestDataDir = testDataDir.resolve("data")
val downloadedTestData = testDataDir.resolve("testData.zip")

val smtLibBenchmarkTestDataDownload = smtLibBenchmarks.map { mkSmtLibBenchmarkTestData(it) }
val smtLibBenchmarkTestDataPrepared = usePreparedSmtLibBenchmarkTestData(unpackedTestDataDir)

val prepareTestData by tasks.registering {
    tasks.withType<ProcessResources>().forEach { it.enabled = false }
    if (usePreparedBenchmarks) {
        dependsOn.add(smtLibBenchmarkTestDataPrepared)
    } else {
        dependsOn.addAll(smtLibBenchmarkTestDataDownload)
    }
}

val testDataRevision = project.stringProperty("testDataRevision") ?: "no-revision"
val downloadPreparedBenchmarksTestData = downloadPreparedSmtLibBenchmarkTestData(
    downloadPath = downloadedTestData,
    testDataPath = unpackedTestDataDir,
    version = testDataRevision
)

tasks.withType<Test> {
    if (runBenchmarksBasedTests) {
        dependsOn.add(prepareTestData)
        environment("benchmarkChunkMaxSize", benchmarkChunkMaxSize)
        environment("benchmarkChunk", benchmarkChunk)
    } else {
        exclude("org/ksmt/test/benchmarks/**")
    }
}

/**
 * Merge all binary test reports starting with [testReportMergePrefix]
 * from the [reports] directory into a single HTML report.
 * Used in CI in a combination with [benchmarkChunk].
 * */
task<TestReport>("mergeTestReports") {
    val mergePrefix = stringProperty("testReportMergePrefix")
    if (mergePrefix != null) {
        destinationDir = rootDir.resolve(mergePrefix)
        val reports = rootDir.resolve("reports").listFiles { f: File -> f.name.startsWith(mergePrefix) }
        reportOn(*reports)
    }
}

jmh {
    stringProperty("jmhIncludes")?.let { includes.add(it) }
}
