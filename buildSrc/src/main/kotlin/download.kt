import de.undercouch.gradle.tasks.download.DownloadExtension
import org.gradle.api.Project
import org.gradle.api.plugins.ExtensionAware
import java.io.File

fun Project.download(url: String, target: File) {
    if (!target.exists()) {
        download().run {
            src(url)
            dest(target)
            overwrite(false)
        }
    }
}

private fun Project.download(): DownloadExtension =
    (this as ExtensionAware).extensions.getByName("download") as DownloadExtension
