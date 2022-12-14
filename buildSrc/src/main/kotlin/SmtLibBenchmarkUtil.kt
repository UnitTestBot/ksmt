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
        val url = "$BENCHMARK_REPO_URL/zip/$name.zip"

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

        val testResources = testResourceDir() ?: error("No resource directory found for $name benchmark")
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

fun Project.usePreparedSmtLibBenchmarkTestData(path: File) = tasks.register("smtLibBenchmark-data-use") {
    doLast {
        check(path.exists()) { "No test data provided" }
        val testResources = testResourceDir() ?: error("No resource directory found for benchmarks")
        val testData = testResources.resolve("testData")
        copy {
            from(path)
            into(testData)
            duplicatesStrategy = DuplicatesStrategy.EXCLUDE
        }
    }
}

fun Project.downloadPreparedSmtLibBenchmarkTestData(downloadPath: File, testDataPath: File, version: String) =
    tasks.register("downloadPreparedSmtLibBenchmarkTestData") {
        doLast {
            val benchmarksUrl = "https://github.com/UnitTestBot/ksmt/releases/download/$version/benchmarks.zip"

            download(benchmarksUrl, downloadPath)

            copy {
                from(zipTree(downloadPath))
                into(testDataPath)
                duplicatesStrategy = DuplicatesStrategy.EXCLUDE
            }
        }
    }

private fun Project.testResourceDir(): File? {
    val sourceSets = (this as ExtensionAware).extensions.getByName("sourceSets") as SourceSetContainer
    return sourceSets["test"]?.output?.resourcesDir
}

private const val BENCHMARK_REPO_URL = "http://smt-lib.loria.fr"
