import org.jetbrains.kotlin.konan.properties.loadProperties

plugins {
    id("io.ksmt.ksmt-base")
}

repositories {
    mavenCentral()
}

val versions = loadProperties(projectDir.parentFile.resolve("version.properties").absolutePath)

dependencies {
    implementation(project(":ksmt-core"))
    implementation(project(":ksmt-maxsmt"))

    testImplementation("org.junit.jupiter:junit-jupiter-api:${versions["junit-jupiter"]}")
    testImplementation("org.junit.jupiter:junit-jupiter-params:${versions["junit-jupiter"]}")

    testImplementation(project(":ksmt-z3"))
    testImplementation(project(":ksmt-bitwuzla"))
    testImplementation(project(":ksmt-cvc5"))
    testImplementation(project(":ksmt-yices"))
    testImplementation(project(":ksmt-symfpu"))

    testImplementation(project(":ksmt-test"))
    testImplementation(project(":ksmt-runner"))

    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:${versions["kotlinx-coroutines"]}")

    testImplementation("io.github.oshai:kotlin-logging-jvm:${versions["kotlin-logging-jvm"]}")
    testImplementation("org.slf4j:slf4j-log4j12:${versions["slf4j-log4j12"]}")

    testImplementation("com.google.code.gson:gson:${versions["gson"]}")
}

val maxSmtBenchmarks = listOfNotNull(
    "QF_ABV-light", // 2.56M
    "QF_ABVFP-light", // 650K
    "QF_AUFBV", // 233K
    "QF_AUFBVLIA-light", // 3.8M
    "QF_AUFLIA", // 244K
    "QF_BV-light", // 8.55M
    "QF_FP", // 250K
    "QF_UF-light", // 3.79M
    "QF_UFBV-light", // 6.27M
    "QF_UFLIA-light", // 488K
    "QF_UFLRA-light", // 2.23M
)

val maxSatBenchmarks = listOfNotNull(
    "maxsat-benchmark-1", // 48.6M
    "maxsat-benchmark-2", // 34.1M
    "maxsat-benchmark-3", // 56.8M
    "maxsat-benchmark-4", // 17.2M
    "maxsat-benchmark-5", // 90.4M
    "maxsat-benchmark-6", // 37.9M
)

val runMaxSmtBenchmarkTests = project.booleanProperty("runMaxSmtBenchmarkTests") ?: false
val runMaxSatBenchmarkTests = project.booleanProperty("runMaxSatBenchmarkTests") ?: false

// Use benchmarks from maxSmtBenchmark directory (test resources) instead of downloading
val usePreparedBenchmarks = project.booleanProperty("usePreparedBenchmarks") ?: true


val downloadPreparedMaxSmtBenchmarkTestData =
    maxSmtBenchmarks.map { maxSmtBenchmarkTestData(it, testDataRevision) }

val prepareMaxSmtTestData by tasks.registering {
    tasks.withType<ProcessResources>().forEach { it.enabled = false }
    if (!usePreparedBenchmarks) {
        dependsOn.addAll(downloadPreparedMaxSmtBenchmarkTestData)
    }
}

tasks.withType<Test> {
    if (runMaxSmtBenchmarkTests) {
        dependsOn.add(prepareMaxSmtTestData)
    }
}

val downloadPreparedMaxSatBenchmarkTestData =
    maxSatBenchmarks.map { maxSatBenchmarkTestData(it, testDataRevision) }

val prepareMaxSatTestData by tasks.registering {
    tasks.withType<ProcessResources>().forEach { it.enabled = false }
    if (!usePreparedBenchmarks) {
        dependsOn.addAll(downloadPreparedMaxSatBenchmarkTestData)
    }
}

tasks.withType<Test> {
    if (runMaxSatBenchmarkTests) {
        dependsOn.add(prepareMaxSatTestData)
    }
}
