import com.jetbrains.rd.generator.gradle.RdGenExtension
import com.jetbrains.rd.generator.gradle.RdGenTask
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    id("io.ksmt.ksmt-base")
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
    testImplementation(project(":ksmt-cvc5"))
}

val sourcesBaseDir = projectDir.resolve("src/main/kotlin")

val generatedPackage = "io.ksmt.runner.generated"
val generatedSourceDir = sourcesBaseDir.resolve(generatedPackage.replace('.', '/'))

val generatedModelsPackage = "$generatedPackage.models"
val generatedModelsSourceDir = sourcesBaseDir.resolve(generatedModelsPackage.replace('.', '/'))

val generateModels = tasks.register<RdGenTask>("generateProtocolModels") {
    val rdParams = extensions.getByName("params") as RdGenExtension
    val sourcesDir = projectDir.resolve("src/main/rdgen").resolve("io/ksmt/runner/models")

    group = "rdgen"
    rdParams.verbose = true
    rdParams.sources(sourcesDir)
    rdParams.hashFolder = buildDir.resolve("rdgen/hashes").absolutePath
    // where to search roots
    rdParams.packages = "io.ksmt.runner.models"

    rdParams.generator {
        language = "kotlin"
        transform = "symmetric"
        root = "io.ksmt.runner.models.SolverProtocolRoot"

        directory = generatedModelsSourceDir.absolutePath
        namespace = generatedModelsPackage
    }

    rdParams.generator {
        language = "kotlin"
        transform = "symmetric"
        root = "io.ksmt.runner.models.TestProtocolRoot"

        directory = generatedModelsSourceDir.absolutePath
        namespace = generatedModelsPackage
    }

    rdParams.generator {
        language = "kotlin"
        transform = "symmetric"
        root = "io.ksmt.runner.models.SyncProtocolRoot"

        directory = generatedModelsSourceDir.absolutePath
        namespace = generatedModelsPackage
    }
}

val generateSolverUtils = tasks.register("generateSolverUtils") {
    val generatorProject = project(":ksmt-runner:solver-generator")
    generatorProject.ext["generatedSolverUtilsPackage"] = generatedPackage
    generatorProject.ext["generatedSolverUtilsPath"] = generatedSourceDir.absolutePath
    dependsOn(":ksmt-runner:solver-generator:generateSolverUtils")
}

tasks.getByName<KotlinCompile>("compileKotlin") {
    // don't treat warnings as errors because of warnings in generated rd models
    kotlinOptions.allWarningsAsErrors = false
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])

            addKsmtPom()
            addSourcesAndJavadoc(project)
            signKsmtPublication(project)
        }
    }
}
