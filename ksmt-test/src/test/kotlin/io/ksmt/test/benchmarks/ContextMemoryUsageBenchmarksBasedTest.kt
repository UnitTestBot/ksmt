package io.ksmt.test.benchmarks

import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import io.ksmt.KContext
import java.nio.file.Path

class ContextMemoryUsageBenchmarksBasedTest : BenchmarksBasedTest() {

    @Disabled
    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("testData")
    fun testMemoryUsage(
        name: String,
        samplePath: Path,
    ) = handleIgnoredTests("testMemoryUsage[$name]") {
        ignoreNoTestDataStub(name)
        testWorkers.withWorker(sharedCtx) { worker ->
            worker.skipBadTestCases {
                val assertions = worker.parseFile(samplePath)
                val ksmtAssertions = worker.convertAssertions(assertions)

                ksmtAssertions.forEach { SortChecker(sharedCtx).apply(it) }

                worker.performEqualityChecks {
                    for ((originalZ3Expr, ksmtExpr) in assertions.zip(ksmtAssertions)) {
                        areEqual(actual = ksmtExpr, expected = originalZ3Expr)
                    }
                    check { "expressions are not equal" }
                }
            }
        }
    }

    companion object {
        @JvmStatic
        fun testData() = testData { name -> name.startsWith("QF_") }

        val sharedCtx: KContext
            get() = sharedCtxField ?: error("Context is not initialized")

        private var sharedCtxField: KContext? = null

        @BeforeAll
        @JvmStatic
        fun initContext() {
            sharedCtxField = KContext(
                operationMode = KContext.OperationMode.CONCURRENT,
                astManagementMode = KContext.AstManagementMode.GC
            )
        }

        @AfterAll
        @JvmStatic
        fun clearContext() {
            sharedCtxField?.close()
            sharedCtxField = null
        }
    }
}
