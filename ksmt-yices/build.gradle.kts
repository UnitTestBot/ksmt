plugins {
    id("io.ksmt.ksmt-base")
}

repositories {
    mavenCentral()
}

dependencies {
    api(project(":ksmt-yices:ksmt-yices-core"))
    implementation(project(":ksmt-yices:ksmt-yices-native"))
}

publishing.publications {
    register<MavenPublication>("maven") {
        addKsmtPom()
        addMavenDependencies(configurations.default.get().allDependencies)
        addEmptyArtifact(project)
        signKsmtPublication(project)
    }
}
