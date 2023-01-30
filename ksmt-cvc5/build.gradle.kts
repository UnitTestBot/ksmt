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
val cvc5Jar = File(projectDir, "dist/cvc5-$cvc5Version.jar")

val cvc5NativeLib by configurations.creating
val cvc5NativeDependency by configurations.creating

dependencies {
    implementation(project(":ksmt-core"))

    cvc5NativeLib("cvc5", "cvc5-native-linux-x86-64", cvc5Version, ext = "zip")
    cvc5NativeDependency("cvc5", "cvc5-native-dependency-linux-x86-64", cvc5Version, ext = "zip")
    cvc5NativeLib("cvc5", "cvc5-native-win32-x86-64", cvc5Version, ext = "zip")
    cvc5NativeDependency("cvc5", "cvc5-native-dependency-win32-x86-64", cvc5Version, ext = "zip")

    api(fileTree(File(project.projectDir, "dist")) {
        include(cvc5Jar.name)
    })
}

tasks.withType<ProcessResources> {
    dependsOn(tasks.compileJava)

    cvc5NativeLib.resolvedConfiguration.resolvedArtifacts.forEach { artifact ->
        from(zipTree(artifact.file)) {
            into("lib/x64")
        }
    }
    cvc5NativeDependency.resolvedConfiguration.resolvedArtifacts.forEach { artifact ->
        from(zipTree(artifact.file)) {
            into("lib/x64")
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
        }
    }
}

val TaskProvider<Task>.outputDirectory: File
    get() = get().outputs.files.singleFile

val TaskProvider<Task>.outputFiles: FileTree
    get() = fileTree(outputDirectory)
