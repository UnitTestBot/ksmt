import org.gradle.api.Project
import org.gradle.api.Task
import org.gradle.api.file.DuplicatesStrategy
import org.gradle.kotlin.dsl.support.unzipTo
import java.io.File

fun Project.usePreparedMaxSmtBenchmarkTestData(path: File) =
    usePreparedBenchmarkTestData(path, "maxSmt")

fun Project.maxSmtBenchmarkTestData(name: String, testDataRevision: String) = tasks.register("maxSmtBenchmark-$name") {
    doLast {
        val path = testResourcesDir().resolve("maxSmtBenchmark/$name")
        val downloadTarget = path.resolve("$name.zip")
        val url = "$MAXSMT_BENCHMARK_REPO_URL/releases/download/$testDataRevision/$name.zip"

        download(url, downloadTarget)

        path.executeIfNotReady("unpack-complete") {
            copy {
                from(zipTree(downloadTarget))
                into(path)
                duplicatesStrategy = DuplicatesStrategy.EXCLUDE
            }
        }
    }
}

fun Project.usePreparedMaxSATBenchmarkTestData(path: File): Task =
    usePreparedBenchmarkTestData(path, "maxSat")
        .get()
        .finalizedBy(unzipMaxSATBenchmarkTestFiles())

fun Project.downloadMaxSATBenchmarkTestData(downloadPath: File, testDataPath: File, testDataRevision: String) =
    downloadPreparedBenchmarkTestData(
        downloadPath,
        testDataPath,
        "MaxSat",
        "$MAXSMT_BENCHMARK_REPO_URL/releases/download/$testDataRevision/maxsat-benchmark.zip",
    )

fun Project.unzipMaxSATBenchmarkTestFiles() =
    tasks.register("unzipMaxSatBenchmarkFiles") {
        doLast {
            val testData = testResourcesDir()
            testData.listFiles()?.forEach { if (it.isFile && it.extension == "zip") unzipTo(it.parentFile, it) }
        }
    }

private fun Project.testResourcesDir() = projectDir.resolve("src/test/resources")

private const val MAXSMT_BENCHMARK_REPO_URL = "https://github.com/victoriafomina/ksmt"
