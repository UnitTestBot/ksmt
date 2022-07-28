import org.gradle.api.Project

fun Project.booleanProperty(name: String): Boolean? {
    if (!project.hasProperty(name)) return null
    val value = project.property(name)
    return value?.toString()?.toBoolean()
}
