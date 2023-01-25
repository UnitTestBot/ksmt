import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    id("org.ksmt.ksmt-base")
    id("com.github.johnrengelman.shadow") version "7.1.2"
}

repositories {
    mavenCentral()
}

val cvc5Version = "1.0.2"

val cvc5Jar = File(projectDir, "dist/cvc5-$cvc5Version.jar")
val cvc5NativeLibs = listOf(
    File(projectDir, "dist/libcvc5.so"),
    File(projectDir, "dist/libcvc5jni.so")
)

dependencies {
    implementation(project(":ksmt-core"))
    api(fileTree(File(project.projectDir, "dist")) {
        include(cvc5Jar.name)
    })
}

tasks.withType<ProcessResources> {
    dependsOn(tasks.compileJava)

    cvc5NativeLibs.forEach { cvc5NativeLib ->
        from(cvc5NativeLib) {
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
