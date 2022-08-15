import de.undercouch.gradle.tasks.download.DownloadExtension
import org.gradle.api.Project
import org.gradle.api.plugins.ExtensionAware
import java.io.File

fun Project.download(url: String, target: File) {
    if (!target.exists()) {
        donwloadWithRetries(url, target, downloadRetries = 3)
    }
}

private fun Project.download(): DownloadExtension =
    (this as ExtensionAware).extensions.getByName("download") as DownloadExtension


private fun Project.donwloadWithRetries(url: String, target: File, downloadRetries: Int) {
    for (attempt in 0..downloadRetries) {
        try {
            if (attempt != 0) {
                println("Download attempt $attempt")
            }
            download().run {
                src(url)
                dest(target)
                overwrite(false)
            }
            return
        } catch (ex: Exception) {
            println("Download failed with: ${ex.message}")
            if (attempt == downloadRetries) {
                throw ex
            }
        }
    }
}
