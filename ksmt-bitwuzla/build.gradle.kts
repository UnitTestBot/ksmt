import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    id("org.ksmt.ksmt-base")
}

repositories {
    mavenCentral()
    flatDir { dirs("dist") }
}

val bitwuzlaNative by configurations.creating

dependencies {
    implementation(project(":ksmt-core"))
    implementation("net.java.dev.jna:jna:5.12.0")
    implementation("net.java.dev.jna:jna-platform:5.12.0")

    testImplementation(project(":ksmt-z3"))
    testImplementation(testFixtures(project(":ksmt-z3")))

    bitwuzlaNative("bitwuzla", "bitwuzla-native-linux-x86-64", "1.0", ext = "zip")
    bitwuzlaNative("bitwuzla", "bitwuzla-native-win32-x86-64", "1.0", ext = "zip")
}

tasks.withType<KotlinCompile> {
    kotlinOptions.freeCompilerArgs += "-opt-in=kotlin.RequiresOptIn"
}

tasks.withType<ProcessResources> {
    bitwuzlaNative.resolvedConfiguration.resolvedArtifacts.forEach { artifact ->
        // destination must be in format {OS}-{ARCH} according to JNA docs
        // https://github.com/java-native-access/jna/blob/master/www/GettingStarted.md
        val destination = artifact.name.removePrefix("bitwuzla-native-")
        from(zipTree(artifact.file)) {
            into(destination)
        }
    }
}

val runBenchmarksBasedTests = project.booleanProperty("bitwuzla.runBenchmarksBasedTests") ?: true

// skip big benchmarks to achieve faster tests build and run time
val skipBigBenchmarks = project.booleanProperty("bitwuzla.skipBigBenchmarks") ?: true

val smtLibBenchmarks = listOfNotNull(
    "QF_UF", // 100M
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
        environment("bitwuzla.benchmarksBasedTests", "enabled")
        dependsOn.add(prepareTestData)
    }
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])
            artifact(tasks["kotlinSourcesJar"])
        }
    }
}
