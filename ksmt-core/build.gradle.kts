import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    id("io.ksmt.ksmt-base")
}

dependencies {
    testImplementation(project(":ksmt-z3"))
    testImplementation("org.junit.jupiter", "junit-jupiter-params", "5.8.2")
}

tasks.withType<KotlinCompile> {
    kotlinOptions.freeCompilerArgs += listOf("-Xjvm-default=all")
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])

            addKsmtPom()
            addSourcesAndJavadoc(project)
            signKsmtPublication(project)
        }
    }
}
