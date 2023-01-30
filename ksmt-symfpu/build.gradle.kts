import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    id("org.ksmt.ksmt-base")
}

dependencies {
    implementation(project(mapOf("path" to ":ksmt-core")))
    testImplementation(project(mapOf("path" to ":ksmt-z3")))
}

tasks.withType<KotlinCompile> {
    kotlinOptions.freeCompilerArgs += listOf("-Xjvm-default=all")
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])
            artifact(tasks["kotlinSourcesJar"])
        }
    }
}

repositories {
    mavenCentral()
}

