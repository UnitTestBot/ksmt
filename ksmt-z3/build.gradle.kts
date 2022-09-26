import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    id("org.ksmt.ksmt-base")
    `java-test-fixtures`
    id("com.github.johnrengelman.shadow") version "7.1.2"
}

repositories {
    mavenCentral()
}

val z3Version = "4.11.2"

val z3JavaJar by lazy { mkZ3ReleaseDownloadTask("x64-win", "*.jar") }

val z3Binaries = listOf(
    mkZ3ReleaseDownloadTask("x64-win", "*.dll"),
    mkZ3ReleaseDownloadTask("x64-glibc-2.31", "*.so"),
    mkZ3ReleaseDownloadTask("x64-osx-10.16", "*.dylib")
)

dependencies {
    implementation(project(":ksmt-core"))
    implementation(fileTree(z3JavaJar.outputDirectory) {
        builtBy(z3JavaJar)
    })

    testImplementation(testFixtures(project(":ksmt-core")))
    testFixturesApi(testFixtures(project(":ksmt-core")))
    testFixturesImplementation(fileTree(z3JavaJar.outputDirectory) {
        builtBy(z3JavaJar)
    })
}

tasks.withType<ProcessResources> {
    dependsOn.addAll(z3Binaries)
    z3Binaries.forEach { z3BinaryTask ->
        from(z3BinaryTask.outputFiles) {
            into("lib/x64")
        }
    }
}

fun Project.mkZ3ReleaseDownloadTask(arch: String, artifactPattern: String): TaskProvider<Task> {
    val z3ReleaseBaseUrl = "https://github.com/Z3Prover/z3/releases/download"
    val releaseName = "z3-${z3Version}"
    val packageName = "z3-${z3Version}-${arch}.zip"
    val packageDownloadTarget = buildDir.resolve("dist").resolve(releaseName).resolve(packageName)
    val downloadUrl = listOf(z3ReleaseBaseUrl, releaseName, packageName).joinToString("/")
    val downloadTaskName = "z3-release-$releaseName-$arch-${artifactPattern.replace('*', '-')}"
    return tasks.register(downloadTaskName) {
        val outputDir = buildDir.resolve("dist").resolve(downloadTaskName)
        doLast {
            download(downloadUrl, packageDownloadTarget)
            val files = zipTree(packageDownloadTarget).matching { include("**/$artifactPattern") }
            copy {
                from(files.files)
                into(outputDir)
            }
        }
        outputs.dir(outputDir)
    }
}

val runBenchmarksBasedTests = project.booleanProperty("z3.runBenchmarksBasedTests") ?: false

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
        include(dependency(z3JavaJar.outputFiles))
    }
    val implementation = project.configurations["implementation"].dependencies.toSet()
    val runtimeOnly = project.configurations["runtimeOnly"].dependencies.toSet()
    val dependencies = (implementation + runtimeOnly)
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

val TaskProvider<Task>.outputDirectory: File
    get() = get().outputs.files.singleFile

val TaskProvider<Task>.outputFiles: FileTree
    get() = fileTree(outputDirectory)
