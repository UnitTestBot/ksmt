import org.gradle.api.Project
import org.gradle.api.Task
import org.gradle.api.file.DuplicatesStrategy
import org.gradle.api.plugins.ExtensionAware
import org.gradle.api.tasks.SourceSetContainer
import org.gradle.api.tasks.TaskProvider
import org.gradle.kotlin.dsl.get
import java.io.File

fun Project.downloadPreparedBenchmarkTestData(
    downloadPath: File,
    testDataPath: File,
    benchmarkName: String,
    benchmarkUrl: String,
): TaskProvider<Task> =
    tasks.register("downloadPrepared${benchmarkName}BenchmarkTestData") {
        doLast {
            download(benchmarkUrl, downloadPath)

            copy {
                from(zipTree(downloadPath))
                into(testDataPath)
                duplicatesStrategy = DuplicatesStrategy.EXCLUDE
            }
        }
    }

fun Project.usePreparedBenchmarkTestData(path: File, benchmarkName: String): TaskProvider<Task> =
    tasks.register("${benchmarkName}Benchmark-data-use") {
        doLast {
            check(path.exists()) { "No test data provided" }
            val testResources = testResourceDir() ?: error("No resource directory found for benchmarks")
            val testData = testResources.resolve("testData")

            testData.executeIfNotReady("test-data-ready") {
                copy {
                    from(path)
                    into(testData)
                    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
                }
            }
        }
    }

fun Project.testResourceDir(): File? {
    val sourceSets = (this as ExtensionAware).extensions.getByName("sourceSets") as SourceSetContainer
    return sourceSets["test"]?.output?.resourcesDir
}

inline fun File.executeIfNotReady(markerName: String, body: () -> Unit) {
    val marker = this.resolve(markerName)
    if (marker.exists()) return

    body()

    marker.createNewFile()
}
