plugins {
    kotlin("jvm") version "1.7.20"
    java
}

repositories {
    mavenCentral()
}

dependencies {
    // core
    implementation("io.ksmt:ksmt-core:0.5.11")
    // z3 solver
    implementation("io.ksmt:ksmt-z3:0.5.11")
    // Runner and portfolio solver
    implementation("io.ksmt:ksmt-runner:0.5.11")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(11))
    }
}