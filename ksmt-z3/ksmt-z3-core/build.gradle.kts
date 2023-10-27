import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    id("io.ksmt.ksmt-base")
    id("com.github.johnrengelman.shadow") version "7.1.2"
}

repositories {
    mavenCentral()
}

val z3Version = "4.12.2"

val z3JavaJar by lazy { mkZ3ReleaseDownloadTask(z3Version, "x64-win", "*.jar") }

dependencies {
    implementation(project(":ksmt-core"))

    api(fileTree(z3JavaJar.outputDirectory) {
        builtBy(z3JavaJar)
    })

    testImplementation(project(":ksmt-z3:ksmt-z3-native"))
}

tasks.withType<ShadowJar> {
    archiveClassifier.set("")
    dependencies {
        include(dependency(z3JavaJar.outputFiles))
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
