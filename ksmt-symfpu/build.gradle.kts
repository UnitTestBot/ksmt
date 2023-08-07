plugins {
    id("io.ksmt.ksmt-base")
    id("com.github.johnrengelman.shadow") version "7.1.2"
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(project(":ksmt-core"))

    testImplementation(project(":ksmt-z3"))
    testImplementation(project(":ksmt-bitwuzla"))
    testImplementation(project(":ksmt-yices"))
    testImplementation(project(":ksmt-runner"))
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
