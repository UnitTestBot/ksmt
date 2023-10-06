import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    id("io.ksmt.ksmt-base")
}

val distDir = projectDir.parentFile.resolve("dist")

repositories {
    mavenCentral()
    flatDir { dirs(distDir) }
}

val compileConfig by configurations.creating

val bitwuzlaNativeWindowsX64 by configurations.creating
val bitwuzlaNativeLinuxX64 by configurations.creating
val bitwuzlaNativeMacArm by configurations.creating

val `windows-x64` by sourceSets.creating
val `linux-x64` by sourceSets.creating
val `mac-arm` by sourceSets.creating

val bitwuzlaBinaries = mapOf(
    `windows-x64` to bitwuzlaNativeWindowsX64,
    `linux-x64` to bitwuzlaNativeLinuxX64,
    `mac-arm` to bitwuzlaNativeMacArm,
)

bitwuzlaBinaries.keys.forEach { it.compileClasspath = compileConfig }

dependencies {
    compileConfig(project(":ksmt-bitwuzla:ksmt-bitwuzla-core"))

    bitwuzlaNativeLinuxX64("bitwuzla", "bitwuzla-linux64", "1.0", ext = "zip")
    bitwuzlaNativeWindowsX64("bitwuzla", "bitwuzla-win64", "1.0", ext = "zip")
    bitwuzlaNativeMacArm("bitwuzla", "bitwuzla-osx-arm64", "1.0", ext = "zip")
}

bitwuzlaBinaries.entries.forEach { (sourceSet, nativeConfig) ->
    val name = sourceSet.name
    val systemArch = name.replace('-', '/')

    val jarTask = tasks.register<Jar>("$name-jar") {
        from(sourceSet.output)
        copyArtifactsIntoJar(nativeConfig, this, "lib/$systemArch/bitwuzla")
    }

    val sourcesJarTask = tasks.register<Jar>("$name-sources-jar") {
        archiveClassifier.set("sources")
        from(sourceSet.allSource)
    }

    publishing.publications {
        register<MavenPublication>("maven-$name") {
            artifactId = "ksmt-bitwuzla-native-$name"

            artifact(jarTask.get())
            artifact(sourcesJarTask.get())
            artifact(project.tasks["dokkaJavadocJar"])

            addKsmtPom()
            signKsmtPublication(project)
        }
    }
}

tasks.getByName<Jar>("jar") {
    bitwuzlaBinaries.forEach { (sourceSet, nativeConfig) ->
        from(sourceSet.output)
        val systemArch = sourceSet.name.replace('-', '/')
        copyArtifactsIntoJar(nativeConfig, this, "lib/$systemArch/bitwuzla")
    }
}
