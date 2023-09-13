import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar


plugins {
    id("io.ksmt.ksmt-base")
    id("com.github.johnrengelman.shadow") version "7.1.2"
}

repositories {
    mavenCentral()
    flatDir { dirs("dist") }
}

val cvc5Version = "1.0.2"
val cvc5Jar = projectDir.resolve("dist/cvc5-$cvc5Version.jar")

val cvc5NativeLinuxX64 by configurations.creating
val cvc5NativeWindowsX64 by configurations.creating
val cvc5NativeMacArm by configurations.creating

dependencies {
    implementation(project(":ksmt-core"))

    cvc5NativeLinuxX64("cvc5", "cvc5-native-linux-x86-64", cvc5Version, ext = "zip")
    cvc5NativeWindowsX64("cvc5", "cvc5-native-win-x86-64", cvc5Version, ext = "zip")
    cvc5NativeMacArm("cvc5", "cvc5-native-osx-arm64", cvc5Version, ext = "zip")

    api(files(cvc5Jar))
}

tasks.withType<ProcessResources> {
    dependsOn(tasks.compileJava)

    sequenceOf(
        "linux/x64" to cvc5NativeLinuxX64,
        "windows/x64" to cvc5NativeWindowsX64,
        "mac/arm" to cvc5NativeMacArm,
    ).forEach { (osName, config) ->
        config.resolvedConfiguration.resolvedArtifacts.forEach { artifact ->
            from(zipTree(artifact.file)) {
                into("lib/$osName/cvc5")
            }
        }
    }
}

tasks.withType<ShadowJar> {
    archiveClassifier.set("")
    dependencies {
        include(dependency(files(cvc5Jar)))
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

val TaskProvider<Task>.outputDirectory: File
    get() = get().outputs.files.singleFile

val TaskProvider<Task>.outputFiles: FileTree
    get() = fileTree(outputDirectory)
