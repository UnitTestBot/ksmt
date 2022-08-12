package org.ksmt.solver.bitwuzla

import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.fixtures.TestDataProvider
import org.ksmt.solver.fixtures.parseAndSkipTestIfError
import org.ksmt.solver.fixtures.skipUnsupportedSolverFeatures
import org.ksmt.solver.fixtures.z3.Z3SmtLibParser
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KSort
import java.nio.file.Path
import kotlin.io.path.relativeTo
import kotlin.test.assertEquals
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds
import kotlin.time.ExperimentalTime
import kotlin.time.TimeMark
import kotlin.time.TimeSource

class BenchmarksBasedTest {

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("testData")
    fun testConverter(name: String, samplePath: Path) = skipUnsupportedSolverFeatures {
        val ctx = KContext()

        val ksmtAssertions = parseAndSkipTestIfError {
            parser.parse(ctx, samplePath)
        }

        KBitwuzlaContext().use { bitwuzlaCtx ->
            val internalizer = KBitwuzlaExprInternalizer(ctx, bitwuzlaCtx)
            val bitwuzlaAssertions = with(internalizer) {
                ksmtAssertions.map { it.internalize() }
            }

            val converter = KBitwuzlaExprConverter(ctx, bitwuzlaCtx)
            val recoveredKsmtAssertions = with(converter) {
                bitwuzlaAssertions.map { it.convertExpr(ctx.boolSort) }
            }

            withTimeoutChecks(10.seconds) {
                for ((original, converted) in ksmtAssertions.zip(recoveredKsmtAssertions)) {

                    // avoid too big test samples
                    if (timeLimitReached()) return

                    KZ3Solver(ctx).use {
                        val checkExpr = with(ctx) { !(original eq converted) }
                        it.assert(checkExpr)
                        val checkResult = it.check(timeout = 1.seconds)
                        assertEquals(KSolverStatus.UNSAT, checkResult, "converter failed: $original -> $converted")
                    }
                }
            }
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("testData")
    fun testSolver(name: String, samplePath: Path) = skipUnsupportedSolverFeatures {
        val ctx = KContext()

        val ksmtAssertions = parseAndSkipTestIfError {
            parser.parse(ctx, samplePath)
        }

        KBitwuzlaSolver(ctx).use { bitwuzla ->
            ksmtAssertions.forEach { bitwuzla.assert(it) }
            val status = bitwuzla.check(timeout = 1.seconds)

            if (status == KSolverStatus.UNKNOWN) return

            val expectedStatus = KZ3Solver(ctx).use { z3Solver ->
                ksmtAssertions.forEach { z3Solver.assert(it) }
                z3Solver.check(timeout = 1.seconds)
            }

            if (expectedStatus == KSolverStatus.UNKNOWN) return

            assertEquals(expectedStatus, status, "bitwuzla check-sat result differ from z3")

            if (status == KSolverStatus.UNSAT) return

            val model = bitwuzla.model()
            val modelAssignments = with(ctx) {
                val vars = model.declarations.map { mkConstApp(it) }
                vars.map {
                    @Suppress("UNCHECKED_CAST")
                    it as KExpr<KSort> eq model.eval(it)
                }
            }
            KZ3Solver(ctx).use { z3Solver ->
                ksmtAssertions.forEach { z3Solver.assert(it) }
                modelAssignments.forEach { z3Solver.assert(it) }
                val modelCheckStatus = z3Solver.check(timeout = 1.seconds)
                assertEquals(KSolverStatus.SAT, modelCheckStatus, "invalid model")
            }
        }
    }

    @OptIn(ExperimentalTime::class)
    private inline fun <reified T> withTimeoutChecks(timeout: Duration, block: TimeoutCheck.() -> T): T =
        TimeoutCheck(TimeSource.Monotonic.markNow(), timeout).block()

    @OptIn(ExperimentalTime::class)
    private class TimeoutCheck(private val begin: TimeMark, private val limit: Duration) {
        fun timeLimitReached(): Boolean = begin.elapsedNow() > limit
    }

    companion object {
        val parser = Z3SmtLibParser()

        @JvmStatic
        fun testData(): List<Arguments> {
            val testDataLocation = TestDataProvider.testDataLocation()
            return TestDataProvider.testData().map {
                Arguments.of(it.relativeTo(testDataLocation).toString(), it)
            }
        }
    }
}
