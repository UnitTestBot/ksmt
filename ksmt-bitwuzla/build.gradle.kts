plugins {
    id("io.ksmt.ksmt-base")
}

repositories {
    mavenCentral()
}

dependencies {
    api(project(":ksmt-bitwuzla:ksmt-bitwuzla-core"))
    implementation(project(":ksmt-bitwuzla:ksmt-bitwuzla-native"))
}

publishing.publications {
    register<MavenPublication>("maven") {
        addKsmtPom()
        addMavenDependencies(configurations.default.get().allDependencies)
        addEmptyArtifact(project)
        signKsmtPublication(project)
    }
}
