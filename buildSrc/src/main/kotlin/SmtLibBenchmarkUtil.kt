import org.gradle.api.Project
import org.gradle.api.file.DuplicatesStrategy
import org.gradle.api.plugins.ExtensionAware
import org.gradle.api.tasks.SourceSetContainer
import org.gradle.kotlin.dsl.get
import java.io.File

fun Project.mkSmtLibBenchmarkTestData(name: String) = tasks.register("smtLibBenchmark-$name") {
    doLast {
        val path = buildDir.resolve("smtLibBenchmark/$name")
        val downloadTarget = path.resolve("$name.zip")
        val repoUrl = "https://clc-gitlab.cs.uiowa.edu:2443"
        val url = "$repoUrl/api/v4/projects/SMT-LIB-benchmarks%2F$name/repository/archive.zip"
        download(url, downloadTarget)

        val unpackCompleteMarker = path.resolve("unpack-complete")
        if (!unpackCompleteMarker.exists()) {
            copy {
                from(zipTree(downloadTarget))
                into(path)
                duplicatesStrategy = DuplicatesStrategy.EXCLUDE
            }
            unpackCompleteMarker.createNewFile()
        }

        val testResources = testResourceDir()!!
        val testData = testResources.resolve("testData")

        val testDataCopyCompleteMarker = testData.resolve("$name-copy-complete")
        if (!testDataCopyCompleteMarker.exists()) {
            val smtFiles = path.walkTopDown().filter { it.extension == "smt2" }.toList()
            copy {
                from(smtFiles.toTypedArray())
                into(testData)
                rename { "${name}_$it" }
                duplicatesStrategy = DuplicatesStrategy.EXCLUDE
            }
            testDataCopyCompleteMarker.createNewFile()
        }
    }
}

private fun Project.testResourceDir(): File? {
    val sourceSets = (this as ExtensionAware).extensions.getByName("sourceSets") as SourceSetContainer
    return sourceSets["test"].output.resourcesDir
}
