import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar


plugins {
    id("org.ksmt.ksmt-base")
    id("com.github.johnrengelman.shadow") version "7.1.2"
}

repositories {
    mavenCentral()
    flatDir { dirs("dist") }
}

val cvc5Version = "1.0.2"
val cvc5Jar = projectDir.resolve("dist/cvc5-$cvc5Version.jar")

val cvc5NativeLibX64 by configurations.creating
val cvc5NativeLibArm by configurations.creating

dependencies {
    implementation(project(":ksmt-core"))

    cvc5NativeLibX64("cvc5", "cvc5-native-linux-x86-64", cvc5Version, ext = "zip")
    cvc5NativeLibX64("cvc5", "cvc5-native-win-x86-64", cvc5Version, ext = "zip")
    cvc5NativeLibArm("cvc5", "cvc5-native-osx-arm64", cvc5Version, ext = "zip")

    api(files(cvc5Jar))
}

tasks.withType<ProcessResources> {
    dependsOn(tasks.compileJava)

    sequenceOf("x64" to cvc5NativeLibX64, "arm" to cvc5NativeLibArm).forEach { (arch, config) ->
        config.resolvedConfiguration.resolvedArtifacts.forEach { artifact ->
            from(zipTree(artifact.file)) {
                into("lib/$arch")
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
            artifact(tasks["kotlinSourcesJar"])

            addKsmtPom()
            signKsmtPublication(project)
        }
    }
}

val TaskProvider<Task>.outputDirectory: File
    get() = get().outputs.files.singleFile

val TaskProvider<Task>.outputFiles: FileTree
    get() = fileTree(outputDirectory)
