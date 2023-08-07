import org.gradle.api.Project
import org.gradle.api.Task
import org.gradle.kotlin.dsl.support.unzipTo
import java.io.File

fun Project.usePreparedMaxSATBenchmarkTestData(path: File): Task =
    usePreparedBenchmarkTestData(path, MAXSAT_BENCHMARK_NAME_L)
        .get()
        .finalizedBy(unzipMaxSATBenchmarkTestFiles())

fun Project.downloadPreparedMaxSATBenchmarkTestData(downloadPath: File, testDataPath: File, testDataRevision: String) =
    downloadPreparedBenchmarkTestData(
        downloadPath,
        testDataPath,
        MAXSAT_BENCHMARK_NAME_U,
        "$MAXSAT_BENCHMARK_REPO_URL/releases/download/$testDataRevision/maxsat-benchmark.zip",
    )

fun Project.unzipMaxSATBenchmarkTestFiles() =
    tasks.register("unzip${MAXSAT_BENCHMARK_NAME_U}BenchmarkFiles") {
        doLast {
            val testResources = testResourceDir() ?: error("No resource directory found for benchmarks")
            val testData = testResources.resolve("testData")

            testData.listFiles()?.forEach { if (it.isFile && it.extension == "zip") unzipTo(it.parentFile, it) }
        }
    }

private const val MAXSAT_BENCHMARK_REPO_URL = "https://github.com/victoriafomina/ksmt"

private const val MAXSAT_BENCHMARK_NAME_U = "MaxSat"

private const val MAXSAT_BENCHMARK_NAME_L = "maxSat"
