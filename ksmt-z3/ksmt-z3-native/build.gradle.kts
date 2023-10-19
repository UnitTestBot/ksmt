plugins {
    id("io.ksmt.ksmt-base")
}

val distDir = projectDir.parentFile.resolve("dist")

repositories {
    mavenCentral()
    flatDir { dirs(distDir) }
}

val compileConfig by configurations.creating
val z3NativeLinuxX64 by configurations.creating

val `windows-x64` by sourceSets.creating
val `linux-x64` by sourceSets.creating
val `mac-x64` by sourceSets.creating
val `mac-arm` by sourceSets.creating

val z3Version = "4.12.2"

val z3Binaries = listOf(
    Triple(`windows-x64`, mkZ3ReleaseDownloadTask(z3Version, "x64-win", "*.dll"), null),
    Triple(`linux-x64`, null, z3NativeLinuxX64),
    Triple(`mac-x64`, mkZ3ReleaseDownloadTask(z3Version, "x64-osx-10.16", "*.dylib"), null),
    Triple(`mac-arm`, mkZ3ReleaseDownloadTask(z3Version, "arm64-osx-11.0", "*.dylib"), null),
)

z3Binaries.forEach { it.first.compileClasspath = compileConfig }

dependencies {
    compileConfig(project(":ksmt-core"))
    compileConfig(project(":ksmt-z3:ksmt-z3-core"))

    z3NativeLinuxX64("z3", "z3-native-linux-x86-64", z3Version, ext = "zip")
}

z3Binaries.forEach { (sourceSet, z3BinaryTask, nativeConfig) ->
    val name = sourceSet.name
    val systemArch = name.replace('-', '/')

    val jarTask = tasks.register<Jar>("$name-jar") {
        from(sourceSet.output)

        z3BinaryTask?.let {
            dependsOn(it)
            from(it.outputFiles) { into("lib/$systemArch/z3") }
        }

        nativeConfig?.let {
            copyArtifactsIntoJar(it, this, "lib/$systemArch/z3")
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
    z3Binaries.forEach { (sourceSet, z3BinaryTask, nativeConfig) ->
        from(sourceSet.output)

        val systemArch = sourceSet.name.replace('-', '/')
        z3BinaryTask?.let {
            dependsOn(it)
            from(it.outputFiles) { into("lib/$systemArch/z3") }
        }
        nativeConfig?.let {
            copyArtifactsIntoJar(it, this, "lib/$systemArch/z3")
        }
    }
}

val TaskProvider<Task>.outputDirectory: File
    get() = get().outputs.files.singleFile

val TaskProvider<Task>.outputFiles: FileTree
    get() = fileTree(outputDirectory)
