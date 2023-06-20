plugins {
    id("io.ksmt.ksmt-base")
}

dependencies {
    implementation(project(":ksmt-core"))
    testImplementation(project(":ksmt-z3"))
    testImplementation(project(":ksmt-bitwuzla"))
    testImplementation(project(":ksmt-yices"))
    testImplementation(project(":ksmt-runner"))
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.6.4")
    testImplementation(project(":ksmt-test"))
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

