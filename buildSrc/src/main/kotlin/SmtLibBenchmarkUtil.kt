import org.gradle.api.Project
import org.gradle.api.file.DuplicatesStrategy
import java.io.File

fun Project.mkSmtLibBenchmarkTestData(name: String) = tasks.register("smtLibBenchmark-$name") {
    doLast {
        val path = buildDir.resolve("smtLibBenchmark/$name")
        val downloadTarget = path.resolve("$name.zip")
        val url = "$MK_SMTLIB_BENCHMARK_REPO_URL/zip/$name.zip"

        download(url, downloadTarget)

        path.executeIfNotReady("unpack-complete") {
            copy {
                from(zipTree(downloadTarget))
                into(path)
                duplicatesStrategy = DuplicatesStrategy.EXCLUDE
            }
        }

        val testResources = testResourceDir() ?: error("No resource directory found for $name benchmark")
        val testData = testResources.resolve("testData")

        testData.executeIfNotReady("$name-copy-complete") {
            val smtFiles = path.walkTopDown().filter { it.extension == "smt2" }.toList()
            copy {
                from(smtFiles.toTypedArray())
                into(testData)
                rename { "${name}_$it" }
                duplicatesStrategy = DuplicatesStrategy.EXCLUDE
            }
        }
    }
}

fun Project.usePreparedSmtLibBenchmarkTestData(path: File) {
    usePreparedBenchmarkTestData(path, SMTLIB_BENCHMARK_NAME_L)
}

fun Project.downloadPreparedSmtLibBenchmarkTestData(downloadPath: File, testDataPath: File, testDataRevision: String) {
    downloadPreparedBenchmarkTestData(
        downloadPath,
        testDataPath,
        SMTLIB_BENCHMARK_NAME_U,
        "https://github.com/UnitTestBot/ksmt/releases/download/$testDataRevision/benchmarks.zip",
    )
}

private const val MK_SMTLIB_BENCHMARK_REPO_URL = "http://smt-lib.loria.fr"

private const val SMTLIB_BENCHMARK_NAME_U = "SmtLib"

private const val SMTLIB_BENCHMARK_NAME_L = "smtLib"
