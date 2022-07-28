import org.jetbrains.kotlin.konan.properties.loadProperties

plugins {
    `kotlin-dsl`
}

repositories {
    gradlePluginPortal()
    mavenCentral()
}

val versions = loadProperties(rootDir.parentFile.resolve("version.properties").absolutePath)

dependencies {
    implementation("org.jetbrains.kotlin:kotlin-gradle-plugin:${versions["kotlin"]}")
    implementation("io.gitlab.arturbosch.detekt:detekt-gradle-plugin:${versions["detekt"]}")
    implementation("de.undercouch.download:de.undercouch.download.gradle.plugin:5.1.0")
}
