import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.6.21"
    id("io.gitlab.arturbosch.detekt") version "1.20.0"
    id("de.undercouch.download") version "5.1.0"
}

repositories {
    mavenCentral()
    flatDir { dirs("dist") }
}

val z3native by configurations.creating

dependencies {
    implementation(project(":ksmt-core"))
    implementation("org.sosy-lab", "javasmt-solver-z3", "4.8.9-sosy1")
    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter", "junit-jupiter-params", "5.8.2")
    z3native("com.microsoft.z3", "z3-native-win64", "4.8.9.1", ext = "zip")
    z3native("com.microsoft.z3", "z3-native-linux64", "4.8.9.1", ext = "zip")
    z3native("com.microsoft.z3", "z3-native-osx", "4.8.9.1", ext = "zip")
}

tasks.getByName<KotlinCompile>("compileKotlin") {
    kotlinOptions.allWarningsAsErrors = true
}

detekt {
    buildUponDefaultConfig = true
    config = files(rootDir.resolve("detekt.yml"))
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
    useJUnitPlatform()
    dependsOn.add(prepareTestData)
}

fun mkSmtLibBenchmarkTestData(name: String) = tasks.register("smtLibBenchmark-$name") {
    doLast {
        val path = buildDir.resolve("smtLibBenchmark/$name")
        val downloadTarget = path.resolve("$name.zip")
        val repoUrl = "https://clc-gitlab.cs.uiowa.edu:2443"
        val url = "$repoUrl/api/v4/projects/SMT-LIB-benchmarks%2F$name/repository/archive.zip"
        download.run {
            src(url)
            dest(downloadTarget)
            overwrite(false)
        }
        copy {
            from(zipTree(downloadTarget))
            into(path)
        }
        val smtFiles = path.walkTopDown().filter { it.extension == "smt2" }.toList()
        val testResources = sourceSets.test.get().output.resourcesDir!!
        val testData = testResources.resolve("testData")
        copy {
            from(smtFiles.toTypedArray())
            into(testData)
            rename { "${name}_$it" }
            duplicatesStrategy = DuplicatesStrategy.EXCLUDE
        }
    }
}
