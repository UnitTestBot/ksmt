import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    id("io.ksmt.ksmt-base")
    id("com.github.johnrengelman.shadow") version "7.1.2"
}

val distDir = projectDir.parentFile.resolve("dist")

repositories {
    mavenCentral()
}

val cvc5Version = "1.0.2"
val cvc5Jar = distDir.resolve("cvc5-$cvc5Version.jar")

dependencies {
    implementation(project(":ksmt-core"))

    api(files(cvc5Jar))
    testImplementation(project(":ksmt-cvc5:ksmt-cvc5-native"))
}

tasks.withType<ShadowJar> {
    archiveClassifier.set("")
    dependencies {
        include(dependency(files(cvc5Jar)))
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
            addSourcesAndJavadoc(project)
            signKsmtPublication(project)
        }
    }
}
