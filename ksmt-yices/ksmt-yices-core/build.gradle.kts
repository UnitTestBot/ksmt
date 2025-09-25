import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    id("io.ksmt.ksmt-base")
    id("com.gradleup.shadow") version "9.0.0-beta13" apply false
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

val publishJar = tasks.register<ShadowJar>("publish-jar") {
    dependsOn(tasks.named("jar"))

    archiveClassifier.set("pub")
    dependencies {
        exclude { true }
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
