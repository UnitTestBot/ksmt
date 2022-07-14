import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.6.21"
    id("io.gitlab.arturbosch.detekt") version "1.20.0"
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

    testImplementation(kotlin("test"))
    bitwuzlaNative("bitwuzla", "bitwuzla-native-linux-x86-64", "1.0", ext = "zip")
    bitwuzlaNative("bitwuzla", "bitwuzla-native-win32-x86-64", "1.0", ext = "zip")
}

tasks.getByName<KotlinCompile>("compileKotlin") {
    kotlinOptions.freeCompilerArgs += "-opt-in=kotlin.RequiresOptIn"
    kotlinOptions.allWarningsAsErrors = true
}

detekt {
    buildUponDefaultConfig = true
    config = files(rootDir.resolve("detekt.yml"))
}

tasks.withType<Test> {
    useJUnitPlatform()
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
