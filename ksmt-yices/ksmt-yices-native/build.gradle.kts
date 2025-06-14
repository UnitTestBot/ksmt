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

val yicesNativeLinuxX64 by configurations.creating
val yicesNativeWindowsX64 by configurations.creating
val yicesNativeMacArm by configurations.creating

val `windows-x64` by sourceSets.creating
val `linux-x64` by sourceSets.creating
val `mac-arm` by sourceSets.creating

val yicesBinaries = mapOf(
    `windows-x64` to yicesNativeWindowsX64,
    `linux-x64` to yicesNativeLinuxX64,
    `mac-arm` to yicesNativeMacArm,
)

yicesBinaries.keys.forEach { it.compileClasspath = compileConfig }

dependencies {
    compileConfig(project(":ksmt-yices:ksmt-yices-core"))

    yicesNativeLinuxX64("yices", "yices-native-linux-x86-64", "0.0", ext = "zip")
    yicesNativeWindowsX64("yices", "yices-native-win32-x86-64", "0.0", ext = "zip")
    yicesNativeMacArm("yices", "yices-native-osx-arm64", "0.0", ext = "zip")
}

yicesBinaries.entries.forEach { (sourceSet, nativeConfig) ->
    val name = sourceSet.name
    val artifactName = "ksmt-yices-native-$name"
    val systemArch = name.replace('-', '/')

    tasks.getByName("compile${name.capitalizeFirstLetter()}Kotlin") {
        mustRunAfter(project(":ksmt-yices:ksmt-yices-core").tasks.named("jar"))
    }

    val jarTask = tasks.register<Jar>("$name-jar") {
        archiveBaseName.set(artifactName)
        from(sourceSet.output)
        copyArtifactsIntoJar(nativeConfig, this, "lib/$systemArch/yices")
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
    yicesBinaries.forEach { (sourceSet, nativeConfig) ->
        from(sourceSet.output)
        val systemArch = sourceSet.name.replace('-', '/')
        copyArtifactsIntoJar(nativeConfig, this, "lib/$systemArch/yices")
    }
}
