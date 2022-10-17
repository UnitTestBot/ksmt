plugins {
    kotlin("jvm") version "1.7.0"
}

repositories {
    mavenCentral()
    maven { url = uri("https://jitpack.io") }
}

dependencies {
    // core
    implementation("com.github.UnitTestBot.ksmt:ksmt-core:main-SNAPSHOT")
    // z3 solver
    implementation("com.github.UnitTestBot.ksmt:ksmt-z3:main-SNAPSHOT")
}
