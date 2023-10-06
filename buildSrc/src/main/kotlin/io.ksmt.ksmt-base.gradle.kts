import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm")
    id("io.gitlab.arturbosch.detekt")
    id("de.undercouch.download")
    id("org.jetbrains.dokka")
    `java-library`
    `maven-publish`
    signing
}

group = "io.ksmt"
version = "0.5.8"

repositories {
    mavenCentral()
}

dependencies {
    // Primitive collections
    implementation("it.unimi.dsi:fastutil-core:8.5.11") // 6.1MB

    implementation("org.jetbrains.kotlinx:kotlinx-collections-immutable:0.3.5")

    testImplementation(kotlin("test"))
}

detekt {
    buildUponDefaultConfig = true
    config = files(rootDir.resolve("detekt.yml"))
}

tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = JavaVersion.VERSION_1_8.toString()
}

tasks.getByName<KotlinCompile>("compileKotlin") {
    kotlinOptions.allWarningsAsErrors = true
}

tasks.withType<Test> {
    useJUnitPlatform()
    systemProperty("junit.jupiter.execution.parallel.enabled", true)
}

tasks.register<Jar>("dokkaJavadocJar") {
    dependsOn(tasks.dokkaJavadoc)
    from(tasks.dokkaJavadoc.flatMap { it.outputDirectory })
    archiveClassifier.set("javadoc")
}

publishing {
    repositories {
        maven {
            name = "releaseDir"
            url = uri(layout.buildDirectory.dir("release"))
        }

        val mavenDeployUrl = project.stringProperty("mavenDeployUrl")
        if (mavenDeployUrl != null) {
            maven {
                name = "central"
                url = uri(mavenDeployUrl)

                credentials {
                    username = project.stringProperty("mavenDeployUser") ?: ""
                    password = project.stringProperty("mavenDeployPassword") ?: ""
                }
            }
        }
    }
}
