import org.gradle.api.artifacts.ResolvedArtifact

fun constructLibraryFolderNameByArtifact(artifact: ResolvedArtifact): String {
    val name = when {
        artifact.name.contains("win") -> "windows"
        artifact.name.contains("linux") -> "linux"
        else -> error("Unsupported system ${artifact.name}")
    }
    return name
}