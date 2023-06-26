plugins {
    id("io.ksmt.ksmt-base")
}

group = "org.example"
version = "unspecified"

repositories {
    mavenCentral()
}

dependencies {
    implementation(project(":ksmt-core"))
    implementation(project(":ksmt-z3"))

    testImplementation(project(":ksmt-core"))
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.8.1")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.8.1")
}
