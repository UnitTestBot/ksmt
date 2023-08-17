plugins {
    kotlin("jvm") // id("io.ksmt.ksmt-base") -- need to be returned in future
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(project(":ksmt-core"))
    // implementation(project(":ksmt-z3"))

    implementation("com.microsoft.onnxruntime:onnxruntime:1.15.1")
}