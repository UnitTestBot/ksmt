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
