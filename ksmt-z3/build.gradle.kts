import org.gradle.language.jvm.tasks.ProcessResources
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.6.21"
    id("io.gitlab.arturbosch.detekt") version "1.20.0"
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
    z3native("com.microsoft.z3", "z3-native-win64", "4.8.9.1", ext = "zip")
    z3native("com.microsoft.z3", "z3-native-linux64", "4.8.9.1", ext = "zip")
    z3native("com.microsoft.z3", "z3-native-osx", "4.8.9.1", ext = "zip")
}

tasks.withType<KotlinCompile> {
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
    z3native.resolvedConfiguration.resolvedArtifacts.forEach { artifact ->
        from(zipTree(artifact.file)) {
            into("lib/x64")
        }
    }
}
