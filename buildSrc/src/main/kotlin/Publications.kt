import gradle.kotlin.dsl.accessors._a772fae24f61b6fdfe5e797ccdf7ddf5.publishing
import gradle.kotlin.dsl.accessors._a772fae24f61b6fdfe5e797ccdf7ddf5.signing
import groovy.util.Node
import org.gradle.api.Project
import org.gradle.api.artifacts.Dependency
import org.gradle.api.artifacts.DependencySet
import org.gradle.api.artifacts.ProjectDependency
import org.gradle.api.publish.maven.MavenPublication
import org.gradle.api.tasks.SourceSet
import org.gradle.api.tasks.bundling.Jar
import org.gradle.kotlin.dsl.get
import org.gradle.kotlin.dsl.register

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

fun MavenPublication.generateMavenMetadata(project: Project) {
    project.tasks.getByName("generateMetadataFileForMavenPublication") {
        mustRunAfter(project.tasks.named("jar"))
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

fun MavenPublication.addSourcesAndJavadoc(
    project: Project, sourceSet: SourceSet,
    name: String, artifactName: String
) {
    val sourcesJarTask = project.tasks.register<Jar>("$name-sources-jar") {
        archiveBaseName.set(artifactName)
        archiveClassifier.set("sources")
        from(sourceSet.allSource)
    }

    val javaDocJarTask = project.tasks.register<Jar>("$name-javadoc-jar") {
        archiveBaseName.set(artifactName)
        archiveClassifier.set("javadoc")

        val javaDocTask = project.tasks["dokkaJavadocJar"]
        dependsOn(javaDocTask)

        from(javaDocTask.outputs)
    }

    artifact(sourcesJarTask.get())
    artifact(javaDocJarTask.get())
}

fun MavenPublication.addEmptyArtifact(project: Project): Unit = with(project) {
    artifact(generateEmptyJar())
    artifact(generateEmptyJar("sources"))
    artifact(generateEmptyJar("javadoc"))
}

fun MavenPublication.addMavenDependencies(
    dependencies: DependencySet,
    dependencyFilter: (MavenPublication) -> Boolean = { true }
) {
    pom.withXml {
        val dependenciesNode: Node = asNode().appendNode("dependencies")
        dependencies.forEach {
            addDependencyPublications(dependenciesNode, it, dependencyFilter)
        }
    }
}

private fun addDependencyPublications(node: Node, dependency: Dependency, dependencyFilter: (MavenPublication) -> Boolean) {
    val project = (dependency as? ProjectDependency)?.dependencyProject ?: return
    project.publishing.publications.filterIsInstance<MavenPublication>().filter(dependencyFilter).forEach {
        val dependencyNode = node.appendNode("dependency")
        addMavenPublicationDependency(dependencyNode, it)
    }
}

private fun addMavenPublicationDependency(node: Node, publication: MavenPublication) {
    node.appendNode("groupId", publication.groupId)
    node.appendNode("artifactId", publication.artifactId)
    node.appendNode("version", publication.version)
}

private fun Project.generateEmptyJar(classifier: String = "") =
    tasks.register<Jar>("$name-$classifier-empty-jar") {
        archiveClassifier.set(classifier)
    }
