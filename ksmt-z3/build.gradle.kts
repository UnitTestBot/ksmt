import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    id("org.ksmt.ksmt-base")
    id("com.github.johnrengelman.shadow") version "7.1.2"
}

repositories {
    mavenCentral()
}

val z3Version = "4.11.2"

val z3JavaJar by lazy { mkZ3ReleaseDownloadTask("x64-win", "*.jar") }

val z3Binaries = mapOf(
    "windows" to mkZ3ReleaseDownloadTask("x64-win", "*.dll"),
    "linux" to mkZ3ReleaseDownloadTask("x64-glibc-2.31", "*.so"),
    "mac64" to mkZ3ReleaseDownloadTask("x64-osx-10.16", "*.dylib"),
    "macArm" to mkZ3ReleaseDownloadTask("arm64-osx-11.0", "*.dylib")
)

dependencies {
    implementation(project(":ksmt-core"))
    api(fileTree(z3JavaJar.outputDirectory) {
        builtBy(z3JavaJar)
    })
}

tasks.withType<ProcessResources> {
    dependsOn.addAll(z3Binaries.values)

    z3Binaries.forEach { (systemName, z3BinaryTask) ->
        from(z3BinaryTask.outputFiles) {
            into("lib/x64/${systemName}")
        }
    }
}

fun Project.mkZ3ReleaseDownloadTask(arch: String, artifactPattern: String): TaskProvider<Task> {
    val z3ReleaseBaseUrl = "https://github.com/Z3Prover/z3/releases/download"
    val releaseName = "z3-${z3Version}"
    val packageName = "z3-${z3Version}-${arch}.zip"
    val packageDownloadTarget = buildDir.resolve("dist").resolve(releaseName).resolve(packageName)
    val downloadUrl = listOf(z3ReleaseBaseUrl, releaseName, packageName).joinToString("/")
    val downloadTaskName = "z3-release-$releaseName-$arch-${artifactPattern.replace('*', '-')}"
    return tasks.register(downloadTaskName) {
        val outputDir = buildDir.resolve("dist").resolve(downloadTaskName)
        doLast {
            download(downloadUrl, packageDownloadTarget)
            val files = zipTree(packageDownloadTarget).matching { include("**/$artifactPattern") }
            copy {
                from(files.files)
                into(outputDir)
            }
        }
        outputs.dir(outputDir)
    }
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
            artifact(tasks["kotlinSourcesJar"])
        }
    }
}

val TaskProvider<Task>.outputDirectory: File
    get() = get().outputs.files.singleFile

val TaskProvider<Task>.outputFiles: FileTree
    get() = fileTree(outputDirectory)
