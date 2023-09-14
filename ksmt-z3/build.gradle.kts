plugins {
    id("io.ksmt.ksmt-base")
}

repositories {
    mavenCentral()
}

dependencies {
    api(project(":ksmt-z3:ksmt-z3-core"))
    implementation(project(":ksmt-z3:ksmt-z3-native"))
}

publishing.publications {
    register<MavenPublication>("maven") {
        addKsmtPom()
        addMavenDependencies(configurations.default.get().allDependencies)
        addEmptyArtifact(project)
        signKsmtPublication(project)
    }
}
