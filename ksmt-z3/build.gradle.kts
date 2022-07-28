import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    id("org.ksmt.ksmt-base")
    `java-test-fixtures`
    id("com.github.johnrengelman.shadow") version "7.1.2"
}

repositories {
    mavenCentral()
}

val z3native by configurations.creating

val z3Version = "4.10.2"

val z3JavaJar by lazy { z3Release("x64-win", "*.jar") }

dependencies {
    implementation(project(":ksmt-core"))
    implementation(z3JavaJar)

    testImplementation(testFixtures(project(":ksmt-core")))
    testFixturesApi(testFixtures(project(":ksmt-core")))
    testFixturesImplementation(z3Release("x64-win", "*.jar"))

    z3native(z3Release("x64-win", "*.dll"))
    z3native(z3Release("x64-glibc-2.31", "*.so"))
    z3native(z3Release("x64-osx-10.16", "*.dylib"))
}

tasks.withType<ProcessResources> {
    from(z3native.resolvedConfiguration.files) {
        into("lib/x64")
    }
}

fun z3Release(arch: String, artifactPattern: String): FileTree {
    val z3ReleaseBaseUrl = "https://github.com/Z3Prover/z3/releases/download"
    val releaseName = "z3-${z3Version}"
    val packageName = "z3-${z3Version}-${arch}.zip"
    val packageDownloadTarget = buildDir.resolve("dist").resolve(releaseName).resolve(packageName)
    download(listOf(z3ReleaseBaseUrl, releaseName, packageName).joinToString("/"), packageDownloadTarget)
    return zipTree(packageDownloadTarget).matching { include("**/$artifactPattern") }
}

val runBenchmarksBasedTests = project.booleanProperty("z3.runBenchmarksBasedTests") ?: true

// skip big benchmarks to achieve faster tests build and run time
val skipBigBenchmarks = project.booleanProperty("z3.skipBigBenchmarks") ?: true

// skip to achieve faster tests run time
val skipZ3SolverTest = project.booleanProperty("z3.skipSolverTest") ?: true

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
    if (!skipBigBenchmarks) "QF_BV" else null// 12.3G
)

val smtLibBenchmarkTestData = smtLibBenchmarks.map { mkSmtLibBenchmarkTestData(it) }

val prepareTestData by tasks.registering {
    dependsOn.addAll(smtLibBenchmarkTestData)
}

tasks.withType<Test> {
    if (runBenchmarksBasedTests) {
        environment("z3.benchmarksBasedTests", "enabled")
        dependsOn.add(prepareTestData)
        if (!skipZ3SolverTest) {
            environment("z3.testSolver", "enabled")
        }
    }
}

tasks.withType<ShadowJar> {
    archiveClassifier.set("")
    dependencies {
        include(dependency(z3JavaJar))
    }
    val implementation = project.configurations["implementation"].dependencies.toSet()
    val runtimeOnly = project.configurations["runtimeOnly"].dependencies.toSet()
    val dependencies =  (implementation + runtimeOnly)
    project.configurations.shadow.get().dependencies.addAll(dependencies)
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            project.shadow.component(this)
            artifact(tasks["kotlinSourcesJar"])
        }
    }
}
