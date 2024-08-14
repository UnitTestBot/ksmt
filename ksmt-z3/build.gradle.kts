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

        addMavenDependencies(configurations.default.get().allDependencies) { dependency ->
            // exclude linux arm from default ksmt-z3 configuration
            dependency.artifactId.let { !it.endsWith("linux-arm") }
        }

        addEmptyArtifact(project)
        signKsmtPublication(project)
    }
}
