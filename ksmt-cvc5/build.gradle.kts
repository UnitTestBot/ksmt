plugins {
    id("io.ksmt.ksmt-base")
}

repositories {
    mavenCentral()
}

dependencies {
    api(project(":ksmt-cvc5:ksmt-cvc5-core"))
    implementation(project(":ksmt-cvc5:ksmt-cvc5-native"))
}

publishing.publications {
    register<MavenPublication>("maven") {
        addKsmtPom()
        addMavenDependencies(configurations.default.get().allDependencies)
        addEmptyArtifact(project)
        signKsmtPublication(project)
    }
}
