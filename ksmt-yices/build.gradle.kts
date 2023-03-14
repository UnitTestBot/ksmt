import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    id("org.ksmt.ksmt-base")
    id("com.github.johnrengelman.shadow") version "7.1.2"
    `java-test-fixtures`
}

val distDir = projectDir.resolve("dist")

repositories {
    mavenCentral()
    flatDir { dirs(distDir) }
}

val yicesNative by configurations.creating

dependencies {
    implementation(project(":ksmt-core"))
    testFixturesImplementation(project(":ksmt-core"))

    yicesNative("yices", "yices-native-linux-x86-64", "0.0", ext = "zip")
    yicesNative("yices", "yices-native-win32-x86-64", "0.0", ext = "zip")
    api(files("$distDir/com.sri.yices.jar"))
}

tasks.withType<ProcessResources> {
    yicesNative.resolvedConfiguration.resolvedArtifacts.forEach { artifact ->
        val name = constructLibraryFolderNameByArtifact(artifact)

        val destination = "lib/x64/$name"

        from(zipTree(artifact.file)) {
            into(destination)
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
            artifact(tasks["kotlinSourcesJar"])
        }
    }
}
