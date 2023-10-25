plugins {
    id("io.ksmt.ksmt-base")
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(project(":ksmt-core"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.2")

    testImplementation("org.junit.jupiter:junit-jupiter-api:5.8.2")
    testImplementation(project(":ksmt-z3"))
}
