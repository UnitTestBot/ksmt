import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    id("io.ksmt.ksmt-base")
    id("com.gradleup.shadow") version "9.0.0-beta13" apply false
}

repositories {
    mavenCentral()
}

val z3Version = "4.15.3"

val z3JavaJar by lazy { mkZ3ReleaseDownloadTask(z3Version, "x64-win", listOf("**/com.microsoft.z3.jar")) }

dependencies {
    implementation(project(":ksmt-core"))

    api(fileTree(z3JavaJar.outputDirectory) {
        builtBy(z3JavaJar)
    })

    testImplementation(project(":ksmt-z3:ksmt-z3-native"))
}

val publishJar = tasks.register<ShadowJar>("publish-jar") {
    dependsOn(tasks.named("jar"))

    archiveClassifier.set("pub")
    dependencies {
        include(dependency(z3JavaJar.outputFiles))
    }

    configurations = listOf(project.configurations.runtimeClasspath.get())

    with(tasks.jar.get() as CopySpec)
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            artifact(publishJar.get())

            addKsmtPom()
            generateMavenMetadata(project)
            addSourcesAndJavadoc(project)
            signKsmtPublication(project)
        }
    }
}

val TaskProvider<Task>.outputDirectory: File
    get() = get().outputs.files.singleFile

val TaskProvider<Task>.outputFiles: FileTree
    get() = fileTree(outputDirectory)
