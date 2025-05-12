import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    id("io.ksmt.ksmt-base")
    id("com.gradleup.shadow") version "9.0.0-beta13"
    `java-test-fixtures`
}

val distDir = projectDir.parentFile.resolve("dist")

repositories {
    mavenCentral()
}

dependencies {
    implementation(project(":ksmt-core"))
    testFixturesImplementation(project(":ksmt-core"))

    api(files(distDir.resolve("com.sri.yices.jar")))

    testImplementation(project(":ksmt-yices:ksmt-yices-native"))
}

tasks.withType<ShadowJar> {
    archiveClassifier.set("")
    dependencies {
        exclude { true }
    }
    val implementation = project.configurations["implementation"].dependencies.toSet()
    val runtimeOnly = project.configurations["runtimeOnly"].dependencies.toSet()
    val dependencies = (implementation + runtimeOnly)
    project.configurations.shadow.get().dependencies.addAll(dependencies)
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            project.shadow.component(this)

            addKsmtPom()
            generateMavenMetadata(project)
            addSourcesAndJavadoc(project)
            signKsmtPublication(project)
        }
    }
}
