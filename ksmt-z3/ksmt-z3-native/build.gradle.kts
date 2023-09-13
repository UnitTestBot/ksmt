plugins {
    id("io.ksmt.ksmt-base")
}

repositories {
    mavenCentral()
}

val compileConfig by configurations.creating

val `windows-x64` by sourceSets.creating
val `linux-x64` by sourceSets.creating
val `mac-x64` by sourceSets.creating
val `mac-arm` by sourceSets.creating

val z3Version = "4.11.2"

val z3Binaries = mapOf(
    `windows-x64` to mkZ3ReleaseDownloadTask(z3Version, "x64-win", "*.dll"),
    `linux-x64` to mkZ3ReleaseDownloadTask(z3Version, "x64-glibc-2.31", "*.so"),
    `mac-x64` to mkZ3ReleaseDownloadTask(z3Version, "x64-osx-10.16", "*.dylib"),
    `mac-arm` to mkZ3ReleaseDownloadTask(z3Version, "arm64-osx-11.0", "*.dylib")
)

z3Binaries.keys.forEach { it.compileClasspath = compileConfig }

dependencies {
    compileConfig(project(":ksmt-core"))
    compileConfig(project(":ksmt-z3:ksmt-z3-core"))
}

z3Binaries.entries.forEach { (sourceSet, z3BinaryTask) ->
    val name = sourceSet.name
    val systemArch = name.replace('-', '/')

    val jarTask = tasks.register<Jar>("$name-jar") {
        dependsOn(z3BinaryTask)

        from(sourceSet.output)
        from(z3BinaryTask.outputFiles) {
            into("lib/$systemArch/z3")
        }
    }

    val sourcesJarTask = tasks.register<Jar>("$name-sources-jar") {
        archiveClassifier.set("sources")
        from(sourceSet.allSource)
    }

    publishing.publications {
        register<MavenPublication>("maven-$name") {
            artifactId = "ksmt-z3-native-$name"

            artifact(jarTask.get())
            artifact(sourcesJarTask.get())
            artifact(project.tasks["dokkaJavadocJar"])

            addKsmtPom()
            signKsmtPublication(project)
        }
    }
}

tasks.getByName<Jar>("jar") {
    dependsOn.addAll(z3Binaries.values)

    z3Binaries.forEach { (sourceSet, z3BinaryTask) ->
        from(sourceSet.output)
        from(z3BinaryTask.outputFiles) {
            val systemArch = sourceSet.name.replace('-', '/')
            into("lib/$systemArch/z3")
        }
    }
}

val TaskProvider<Task>.outputDirectory: File
    get() = get().outputs.files.singleFile

val TaskProvider<Task>.outputFiles: FileTree
    get() = fileTree(outputDirectory)
