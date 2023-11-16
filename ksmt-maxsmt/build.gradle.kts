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
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:${versions["kotlinx-coroutines"]}")

    testImplementation("org.junit.jupiter:junit-jupiter-api:${versions["junit-jupiter"]}")
    testImplementation(project(":ksmt-z3"))
}
