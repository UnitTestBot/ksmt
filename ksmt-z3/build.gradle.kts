plugins {
    id("org.ksmt.ksmt-base")
    `java-test-fixtures`
}

repositories {
    mavenCentral()
    flatDir { dirs("dist") }
}

val z3native by configurations.creating

dependencies {
    implementation(project(":ksmt-core"))
    implementation("org.sosy-lab", "javasmt-solver-z3", "4.8.9-sosy1")

    testImplementation("org.junit.jupiter", "junit-jupiter-params", "5.8.2")
    testImplementation(testFixtures(project(":ksmt-core")))
    testFixturesImplementation(testFixtures(project(":ksmt-core")))
    testFixturesImplementation("org.sosy-lab", "javasmt-solver-z3", "4.8.9-sosy1")

    z3native("com.microsoft.z3", "z3-native-win64", "4.8.9.1", ext = "zip")
    z3native("com.microsoft.z3", "z3-native-linux64", "4.8.9.1", ext = "zip")
    z3native("com.microsoft.z3", "z3-native-osx", "4.8.9.1", ext = "zip")
}

tasks.withType<ProcessResources> {
    z3native.resolvedConfiguration.resolvedArtifacts.forEach { artifact ->
        from(zipTree(artifact.file)) {
            into("lib/x64")
        }
    }
}

// skip big benchmarks to achieve faster tests build and run time
val skipBigBenchmarks = true

// skip to achieve faster tests run time
val skipZ3SolverTest = true

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
    if (!skipBigBenchmarks) "QF_ABV" else null, // 253M
    if (!skipBigBenchmarks) "QF_BV" else null// 12.3G
)

val smtLibBenchmarkTestData = smtLibBenchmarks.map { mkSmtLibBenchmarkTestData(it) }

val prepareTestData by tasks.registering {
    dependsOn.addAll(smtLibBenchmarkTestData)
}

tasks.withType<Test> {
    dependsOn.add(prepareTestData)
    if (!skipZ3SolverTest) {
        environment("z3.testSolver", "enabled")
    }
}
