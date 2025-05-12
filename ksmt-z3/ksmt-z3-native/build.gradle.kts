import gratatouille.capitalizeFirstLetter

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
val `windows-arm` by sourceSets.creating
val `linux-arm` by sourceSets.creating

val z3Version = "4.13.4"

val winDllPath = listOf("**/vcruntime140.dll", "**/vcruntime140_1.dll", "**/libz3.dll", "**/libz3java.dll")
val linuxSoPath = listOf("**/libz3.so", "**/libz3java.so")
val macDylibPath = listOf("**/libz3.dylib", "**/libz3java.dylib")

val z3Binaries = listOf(
    Triple(`windows-x64`, mkZ3ReleaseDownloadTask(z3Version, "x64-win", winDllPath), null),
    Triple(`linux-x64`, null, z3NativeLinuxX64),
    Triple(`mac-x64`, mkZ3ReleaseDownloadTask(z3Version, "x64-osx-13.7.1", macDylibPath), null),
    Triple(`mac-arm`, mkZ3ReleaseDownloadTask(z3Version, "arm64-osx-13.7.1", macDylibPath), null),
    Triple(`linux-arm`, mkZ3ReleaseDownloadTask(z3Version, "arm64-glibc-2.34", linuxSoPath), null),
)

z3Binaries.forEach { it.first.compileClasspath = compileConfig }

dependencies {
    compileConfig(project(":ksmt-core"))
    compileConfig(project(":ksmt-z3:ksmt-z3-core"))

    z3NativeLinuxX64("z3", "z3-native-linux-x86-64", z3Version, ext = "zip")
}

z3Binaries.forEach { (sourceSet, z3BinaryTask, nativeConfig) ->
    val name = sourceSet.name
    val artifactName = "ksmt-z3-native-$name"
    val systemArch = name.replace('-', '/')

    tasks.getByName("compile${name.capitalizeFirstLetter()}Kotlin") {
        mustRunAfter(project(":ksmt-z3:ksmt-z3-core").tasks.named("jar"))
    }

    val jarTask = tasks.register<Jar>("$name-jar") {
        archiveBaseName.set(artifactName)
        from(sourceSet.output)

        z3BinaryTask?.let {
            dependsOn(it)
            from(it.outputFiles) { into("lib/$systemArch/z3") }
        }

        nativeConfig?.let {
            copyArtifactsIntoJar(it, this, "lib/$systemArch/z3")
        }
    }

    publishing.publications {
        register<MavenPublication>("maven-$name") {
            artifactId = artifactName

            artifact(jarTask.get())
            addSourcesAndJavadoc(project, sourceSet, name, artifactName)

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
