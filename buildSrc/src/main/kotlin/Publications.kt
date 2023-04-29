import gradle.kotlin.dsl.accessors._87b80c14bf1c4d505c7a71d7741e0994.signing
import org.gradle.api.Project
import org.gradle.api.publish.maven.MavenPublication
import org.gradle.kotlin.dsl.get

fun MavenPublication.addKsmtPom() {
    pom {
        packaging = "jar"
        name.set("io.ksmt")
        description.set("Kotlin API for various SMT solvers")
        url.set("https://www.ksmt.io/")

        licenses {
            license {
                name.set("The Apache License, Version 2.0")
                url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
            }
        }

        issueManagement {
            url.set("https://github.com/UnitTestBot/ksmt/issues")
        }

        scm {
            connection.set("scm:git:https://github.com/UnitTestBot/ksmt.git")
            developerConnection.set("scm:git:https://github.com/UnitTestBot/ksmt.git")
            url.set("https://github.com/UnitTestBot/ksmt")
        }

        developers {
            developer {
                id.set("saloed")
                name.set("Valentyn Sobol")
                email.set("vo.sobol@mail.ru")
            }

            developer {
                id.set("CaelmBleidd")
                name.set("Alexey Menshutin")
                email.set("alex.menshutin99@gmail.com")
            }
        }
    }
}

fun MavenPublication.signKsmtPublication(project: Project) = with(project) {
    signing {
        val gpgKey = project.stringProperty("mavenSignGpgKey")?.removeSurrounding("\"")
        val gpgPassword = project.stringProperty("mavenSignGpgPassword")

        if (gpgKey != null && gpgPassword != null) {
            useInMemoryPgpKeys(gpgKey, gpgPassword)
            sign(this@signKsmtPublication)
        }
    }
}

fun MavenPublication.addSourcesAndJavadoc(project: Project) {
    artifact(project.tasks["kotlinSourcesJar"])
    artifact(project.tasks["dokkaJavadocJar"])
}
