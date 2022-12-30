import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    id("org.ksmt.ksmt-base")
}

repositories {
    mavenCentral()
    flatDir { dirs("dist") }
}

val bitwuzlaNativeJna by configurations.creating
val bitwuzlaNativeDependency by configurations.creating

dependencies {
    implementation(project(":ksmt-core"))
    api("net.java.dev.jna:jna:5.12.0")
    implementation("net.java.dev.jna:jna-platform:5.12.0")

    bitwuzlaNativeJna("bitwuzla", "bitwuzla-native-linux-x86-64", "1.0", ext = "zip")
    bitwuzlaNativeDependency("bitwuzla", "bitwuzla-native-dependency-linux-x86-64", "1.0", ext = "zip")
    bitwuzlaNativeJna("bitwuzla", "bitwuzla-native-win32-x86-64", "1.0", ext = "zip")
    bitwuzlaNativeDependency("bitwuzla", "bitwuzla-native-dependency-win32-x86-64", "1.0", ext = "zip")
}

tasks.withType<KotlinCompile> {
    kotlinOptions.freeCompilerArgs += "-opt-in=kotlin.RequiresOptIn"
}

tasks.withType<ProcessResources> {
    bitwuzlaNativeJna.resolvedConfiguration.resolvedArtifacts.forEach { artifact ->
        // destination must be in format {OS}-{ARCH} according to JNA docs
        // https://github.com/java-native-access/jna/blob/master/www/GettingStarted.md
        val destination = artifact.name.removePrefix("bitwuzla-native-")
        from(zipTree(artifact.file)) {
            into(destination)
        }
    }
    bitwuzlaNativeDependency.resolvedConfiguration.resolvedArtifacts.forEach { artifact ->
        from(zipTree(artifact.file)) {
            into("lib/x64")
        }
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
