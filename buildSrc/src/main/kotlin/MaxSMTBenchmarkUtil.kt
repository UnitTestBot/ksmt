import org.gradle.api.Project
import org.gradle.api.file.DuplicatesStrategy
import org.gradle.kotlin.dsl.support.unzipTo

fun Project.maxSmtBenchmarkTestData(name: String, testDataRevision: String) = tasks.register("maxSmtBenchmark-$name") {
    doLast {
        downloadBenchmarkTestData(name, testDataRevision)
    }
}

fun Project.maxSatBenchmarkTestData(name: String, testDataRevision: String) = tasks.register(name) {
    doLast {
        downloadBenchmarkTestData(name, testDataRevision)
        unzipMaxSATBenchmarkTestFiles()
    }
}

fun Project.downloadBenchmarkTestData(name: String, testDataRevision: String) {
    val path = testResourcesDir().resolve("maxSmtBenchmark/$name")
    val downloadTarget = path.resolve("$name.zip")
    val url = "$BENCHMARKS_REPO_URL/releases/download/$testDataRevision/$name.zip"

    download(url, downloadTarget)

    path.executeIfNotReady("unpack-complete") {
        copy {
            from(zipTree(downloadTarget))
            into(path)
            duplicatesStrategy = DuplicatesStrategy.EXCLUDE
        }
    }
}

fun Project.unzipMaxSATBenchmarkTestFiles() {
    val testData = testResourcesDir().resolve("maxSmtBenchmark")
    testData.walk().forEach { if (it.isFile && it.extension == "zip") unzipTo(it.parentFile, it) }
}

private fun Project.testResourcesDir() = projectDir.resolve("src/test/resources")

private const val BENCHMARKS_REPO_URL = "https://github.com/victoriafomina/ksmt"
