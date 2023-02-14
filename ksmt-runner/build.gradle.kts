import com.jetbrains.rd.generator.gradle.RdGenExtension
import com.jetbrains.rd.generator.gradle.RdGenTask
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    id("org.ksmt.ksmt-base")
    id("com.jetbrains.rdgen") version "2022.3.2"
}

repositories {
    mavenCentral()
}

val rdgenModelsCompileClasspath by configurations.creating {
    extendsFrom(configurations.compileClasspath.get())
}

kotlin {
    sourceSets.create("rdgenModels").apply {
        kotlin.srcDir("src/main/rdgen")
    }
}

dependencies {
    implementation(project(":ksmt-core"))

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.6.4")

    api("com.jetbrains.rd:rd-core:2022.3.2")
    api("com.jetbrains.rd:rd-framework:2022.3.2")

    rdgenModelsCompileClasspath("com.jetbrains.rd:rd-gen:2022.3.2")

    testImplementation(kotlin("test"))

    testImplementation(project(":ksmt-z3"))
    testImplementation(project(":ksmt-bitwuzla"))
    testImplementation(project(":ksmt-yices"))
}

val generateModels = tasks.register<RdGenTask>("generateProtocolModels") {
    val rdParams = extensions.getByName("params") as RdGenExtension
    val sourcesDir = projectDir.resolve("src/main/rdgen").resolve("org/ksmt/runner/models")
    val sourcesBaseDir = projectDir.resolve("src/main/kotlin")
    val generatedModelsPackage = "org.ksmt.runner.models.generated"
    val generatedSourceDir = sourcesBaseDir.resolve(generatedModelsPackage.replace('.', '/'))

    group = "rdgen"
    rdParams.verbose = true
    rdParams.sources(sourcesDir)
    rdParams.hashFolder = buildDir.resolve("rdgen/hashes").absolutePath
    // where to search roots
    rdParams.packages = "org.ksmt.runner.models"

    rdParams.generator {
        language = "kotlin"
        transform = "symmetric"
        root = "org.ksmt.runner.models.SolverProtocolRoot"

        directory = generatedSourceDir.absolutePath
        namespace = generatedModelsPackage
    }

    rdParams.generator {
        language = "kotlin"
        transform = "symmetric"
        root = "org.ksmt.runner.models.TestProtocolRoot"

        directory = generatedSourceDir.absolutePath
        namespace = generatedModelsPackage
    }

    rdParams.generator {
        language = "kotlin"
        transform = "symmetric"
        root = "org.ksmt.runner.models.SyncProtocolRoot"

        directory = generatedSourceDir.absolutePath
        namespace = generatedModelsPackage
    }
}

tasks.getByName<KotlinCompile>("compileKotlin") {
    // don't treat warnings as errors because of warnings in generated rd models
    kotlinOptions.allWarningsAsErrors = false
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])
            artifact(tasks["kotlinSourcesJar"])
        }
    }
}
