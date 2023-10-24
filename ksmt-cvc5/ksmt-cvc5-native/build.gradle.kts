plugins {
    id("io.ksmt.ksmt-base")
}

val distDir = projectDir.parentFile.resolve("dist")

repositories {
    mavenCentral()
    flatDir { dirs(distDir) }
}

val compileConfig by configurations.creating

val cvc5NativeLinuxX64 by configurations.creating
val cvc5NativeWindowsX64 by configurations.creating
val cvc5NativeMacArm by configurations.creating

val `windows-x64` by sourceSets.creating
val `linux-x64` by sourceSets.creating
val `mac-arm` by sourceSets.creating

val cvc5Binaries = mapOf(
    `windows-x64` to cvc5NativeWindowsX64,
    `linux-x64` to cvc5NativeLinuxX64,
    `mac-arm` to cvc5NativeMacArm,
)

cvc5Binaries.keys.forEach { it.compileClasspath = compileConfig }

val cvc5Version = "1.0.2"

dependencies {
    compileConfig(project(":ksmt-cvc5:ksmt-cvc5-core"))

    cvc5NativeLinuxX64("cvc5", "cvc5-native-linux-x86-64", cvc5Version, ext = "zip")
    cvc5NativeWindowsX64("cvc5", "cvc5-native-win-x86-64", cvc5Version, ext = "zip")
    cvc5NativeMacArm("cvc5", "cvc5-native-osx-arm64", cvc5Version, ext = "zip")
}

cvc5Binaries.entries.forEach { (sourceSet, nativeConfig) ->
    val name = sourceSet.name
    val artifactName = "ksmt-cvc5-native-$name"
    val systemArch = name.replace('-', '/')

    val jarTask = tasks.register<Jar>("$name-jar") {
        archiveBaseName.set(artifactName)
        from(sourceSet.output)
        copyArtifactsIntoJar(nativeConfig, this, "lib/$systemArch/cvc5")
    }

    val sourcesJarTask = tasks.register<Jar>("$name-sources-jar") {
        archiveBaseName.set(artifactName)
        archiveClassifier.set("sources")
        from(sourceSet.allSource)
    }

    publishing.publications {
        register<MavenPublication>("maven-$name") {
            artifactId = artifactName

            artifact(jarTask.get())
            artifact(sourcesJarTask.get())
            artifact(project.tasks["dokkaJavadocJar"])

            addKsmtPom()
            signKsmtPublication(project)
        }
    }
}

tasks.getByName<Jar>("jar") {
    cvc5Binaries.forEach { (sourceSet, nativeConfig) ->
        from(sourceSet.output)
        val systemArch = sourceSet.name.replace('-', '/')
        copyArtifactsIntoJar(nativeConfig, this, "lib/$systemArch/cvc5")
    }
}
