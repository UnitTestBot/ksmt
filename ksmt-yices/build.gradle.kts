import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    id("io.ksmt.ksmt-base")
    id("com.github.johnrengelman.shadow") version "7.1.2"
    `java-test-fixtures`
}

val distDir = projectDir.resolve("dist")

repositories {
    mavenCentral()
    flatDir { dirs(distDir) }
}

val yicesNativeLinuxX64 by configurations.creating
val yicesNativeWindowsX64 by configurations.creating
val yicesNativeMacArm by configurations.creating

dependencies {
    implementation(project(":ksmt-core"))
    testFixturesImplementation(project(":ksmt-core"))

    yicesNativeLinuxX64("yices", "yices-native-linux-x86-64", "0.0", ext = "zip")
    yicesNativeWindowsX64("yices", "yices-native-win32-x86-64", "0.0", ext = "zip")
    yicesNativeMacArm("yices", "yices-native-osx-arm64", "0.0", ext = "zip")
    api(files("$distDir/com.sri.yices.jar"))
}

tasks.withType<ProcessResources> {
    sequenceOf(
        "linux/x64" to yicesNativeLinuxX64,
        "windows/x64" to yicesNativeWindowsX64,
        "mac/arm" to yicesNativeMacArm,
    ).forEach { (osArch, config) ->
        config.resolvedConfiguration.resolvedArtifacts.forEach { artifact ->
            val destination = "lib/$osArch/yices"
            from(zipTree(artifact.file)) {
                into(destination)
            }
        }
    }
}

tasks.withType<ShadowJar> {
    archiveClassifier.set("")
    dependencies {
        exclude { true }
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

            addKsmtPom()
            addSourcesAndJavadoc(project)
            signKsmtPublication(project)
        }
    }
}
