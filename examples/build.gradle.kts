plugins {
    kotlin("jvm") version "1.7.20"
    java
}

repositories {
    mavenCentral()
    maven { url = uri("https://jitpack.io") }
}

dependencies {
    // core
    implementation("com.github.UnitTestBot.ksmt:ksmt-core:0.3.1")
    // z3 solver
    implementation("com.github.UnitTestBot.ksmt:ksmt-z3:0.3.1")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(11))
    }
}