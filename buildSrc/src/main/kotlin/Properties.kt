import org.gradle.api.Project

fun Project.booleanProperty(name: String): Boolean? {
    if (!project.hasProperty(name)) {
        return null
    }

    return project.property(name)?.toString()?.toBoolean()
}
