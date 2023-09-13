import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    id("io.ksmt.ksmt-base")
}

repositories {
    mavenCentral()
    flatDir { dirs("dist") }
}

val bitwuzlaNativeWindowsX64 by configurations.creating
val bitwuzlaNativeLinuxX64 by configurations.creating
val bitwuzlaNativeMacArm by configurations.creating

dependencies {
    implementation(project(":ksmt-core"))

    bitwuzlaNativeLinuxX64("bitwuzla", "bitwuzla-linux64", "1.0", ext = "zip")
    bitwuzlaNativeWindowsX64("bitwuzla", "bitwuzla-win64", "1.0", ext = "zip")
    bitwuzlaNativeMacArm("bitwuzla", "bitwuzla-osx-arm64", "1.0", ext = "zip")
}

tasks.withType<KotlinCompile> {
    kotlinOptions.freeCompilerArgs += "-opt-in=kotlin.RequiresOptIn"
}

tasks.withType<ProcessResources> {
    sequenceOf(
        "windows/x64" to bitwuzlaNativeWindowsX64,
        "linux/x64" to bitwuzlaNativeLinuxX64,
        "mac/arm" to bitwuzlaNativeMacArm,
    ).forEach { (osArch, config) ->
        config.resolvedConfiguration.resolvedArtifacts.forEach { artifact ->
            from(zipTree(artifact.file)) {
                into("lib/$osArch/bitwuzla")
            }
        }
    }
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])

            addKsmtPom()
            addSourcesAndJavadoc(project)
            signKsmtPublication(project)
        }
    }
}
