plugins {
    id("io.ksmt.ksmt-base")
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(project(":ksmt-core"))
    implementation(project("utils"))

    implementation("com.microsoft.onnxruntime:onnxruntime:1.15.1")
    implementation("com.github.ajalt.clikt:clikt:3.5.2")
}

tasks {
    val standaloneModelRunnerFatJar = register<Jar>("standaloneModelRunnerFatJar") {
        dependsOn.addAll(listOf("compileJava", "compileKotlin", "processResources"))

        archiveFileName.set("standalone-model-runner.jar")
        destinationDirectory.set(File("."))
        duplicatesStrategy = DuplicatesStrategy.EXCLUDE

        manifest {
            attributes(mapOf("Main-Class" to "io.ksmt.solver.neurosmt.runtime.standalone.StandaloneModelRunnerKt"))
        }

        val sourcesMain = sourceSets.main.get()
        val contents = configurations.runtimeClasspath.get()
            .map { if (it.isDirectory) it else zipTree(it) } + sourcesMain.output

        from(contents)
    }

    build {
        dependsOn(standaloneModelRunnerFatJar)
    }
}