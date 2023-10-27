import org.gradle.api.Project
import org.gradle.api.artifacts.Configuration
import org.gradle.api.tasks.bundling.Jar

fun Project.copyArtifactsIntoJar(configuration: Configuration, jar: Jar, path: String) = with(jar) {
    configuration.resolvedConfiguration.resolvedArtifacts.forEach { artifact ->
        from(zipTree(artifact.file)) {
            into(path)
        }
    }
}
