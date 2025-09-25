import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    id("io.ksmt.ksmt-base")
    id("com.gradleup.shadow") version "9.0.0-beta13" apply false
}

val distDir = projectDir.parentFile.resolve("dist")

repositories {
    mavenCentral()
}

val cvc5Version = "1.3.0"
val cvc5Jar = distDir.resolve("cvc5-$cvc5Version.jar")

dependencies {
    implementation(project(":ksmt-core"))

    api(files(cvc5Jar))
    testImplementation(project(":ksmt-cvc5:ksmt-cvc5-native"))
}

tasks.named<Jar>("jar") {
    archiveClassifier.set("jar")
}

val publishJar = tasks.register<ShadowJar>("publish-jar") {
    dependsOn(tasks.named("jar"))

    archiveClassifier.set("")
    dependencies {
        include(dependency(files(cvc5Jar)))
    }

    configurations = listOf(project.configurations.runtimeClasspath.get())

    with(tasks.jar.get() as CopySpec)
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            artifact(publishJar.get())

            addKsmtPom()
            generateMavenMetadata(project)
            addSourcesAndJavadoc(project)
            signKsmtPublication(project)
        }
    }
}
