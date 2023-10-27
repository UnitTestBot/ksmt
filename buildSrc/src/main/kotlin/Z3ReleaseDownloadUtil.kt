import org.gradle.api.Project
import org.gradle.api.Task
import org.gradle.api.tasks.TaskProvider

fun Project.mkZ3ReleaseDownloadTask(z3Version: String, arch: String, artifactPattern: String): TaskProvider<Task> {
    val z3ReleaseBaseUrl = "https://github.com/Z3Prover/z3/releases/download"
    val releaseName = "z3-${z3Version}"
    val packageName = "z3-${z3Version}-${arch}.zip"
    val packageDownloadTarget = buildDir.resolve("dist").resolve(releaseName).resolve(packageName)
    val downloadUrl = listOf(z3ReleaseBaseUrl, releaseName, packageName).joinToString("/")
    val downloadTaskName = "z3-release-$releaseName-$arch-${artifactPattern.replace('*', '-')}"
    return tasks.register(downloadTaskName) {
        val outputDir = buildDir.resolve("dist").resolve(downloadTaskName)
        doLast {
            download(downloadUrl, packageDownloadTarget)
            val files = zipTree(packageDownloadTarget).matching { include("**/$artifactPattern") }
            copy {
                from(files.files)
                into(outputDir)
            }
        }
        outputs.dir(outputDir)
    }
}
