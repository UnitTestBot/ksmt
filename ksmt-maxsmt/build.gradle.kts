import org.jetbrains.kotlin.konan.properties.loadProperties

plugins {
    id("io.ksmt.ksmt-base")
}

repositories {
    mavenCentral()
}

val versions = loadProperties(projectDir.parentFile.resolve("version.properties").absolutePath)

dependencies {
    implementation(project(":ksmt-core"))
    implementation(project(":ksmt-runner"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:${versions["kotlinx-coroutines"]}")

    implementation("io.github.oshai:kotlin-logging-jvm:${versions["kotlin-logging-jvm"]}")
    implementation("org.slf4j:slf4j-log4j12:${versions["slf4j-log4j12"]}")

    testImplementation("org.junit.jupiter:junit-jupiter-api:${versions["junit-jupiter"]}")
    testImplementation(project(":ksmt-z3"))
}
