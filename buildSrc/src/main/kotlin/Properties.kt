import org.gradle.api.Project

fun Project.booleanProperty(name: String): Boolean? {
    if (!project.hasProperty(name)) {
        return null
    }

    return project.property(name)?.toString()?.toBoolean()
}

fun Project.intProperty(name: String): Int? {
    if (!project.hasProperty(name)) {
        return null
    }

    return project.property(name)?.toString()?.toIntOrNull()
}

fun Project.stringProperty(name: String): String? {
    if (!project.hasProperty(name)) {
        return null
    }

    return project.property(name).toString()
}
