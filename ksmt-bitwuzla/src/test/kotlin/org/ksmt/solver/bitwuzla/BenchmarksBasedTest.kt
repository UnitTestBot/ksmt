package org.ksmt.solver.bitwuzla

import org.junit.jupiter.api.Assumptions
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.expr.transformer.KNonRecursiveTransformer
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
import kotlin.test.assertTrue
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

@EnabledIfEnvironmentVariable(
    named = "bitwuzla.benchmarksBasedTests",
    matches = "enabled",
    disabledReason = "bitwuzla benchmarks based test"
)
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

            recoveredKsmtAssertions.forEach {
                ExpressionValidator(ctx).apply(it)
            }

            with(EqualityChecker(ctx)) {
                ksmtAssertions.zip(recoveredKsmtAssertions).forEach { (original, converted) ->
                    areEqual(actual = converted, expected = original)
                }
                check(timeout = 1.seconds) { "converter failed" }
            }
        }
    }

    @Execution(ExecutionMode.CONCURRENT)
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

            if (status == KSolverStatus.UNKNOWN) {
                Assumptions.assumeTrue(false, "bitwuzla solver unknown -- nothing to test")
            }

            if (status == KSolverStatus.SAT) {
                val model = bitwuzla.model()
                // check no exceptions during model detach
                model.detach()
            }

            val expectedStatus = KZ3Solver(ctx).use { z3Solver ->
                ksmtAssertions.forEach { z3Solver.assert(it) }
                z3Solver.check(timeout = 1.seconds)
            }

            if (expectedStatus == KSolverStatus.UNKNOWN) {
                Assumptions.assumeTrue(false, "expected status unknown -- nothing to test")
            }

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

    private class ExpressionValidator(ctx: KContext) : KNonRecursiveTransformer(ctx) {
        override fun <T : KSort> transformApp(expr: KApp<T, *>): KExpr<T> = with(ctx) {
            // apply internally check arguments sorts
            expr.decl.apply(expr.args)
            return super.transformApp(expr)
        }
    }

    private data class EqualityCheck(val actual: KExpr<KSort>, val expected: KExpr<KSort>)

    private class EqualityChecker(val ctx: KContext) {
        private val equalityChecks = mutableListOf<EqualityCheck>()

        @Suppress("UNCHECKED_CAST")
        fun <T : KSort> areEqual(actual: KExpr<T>, expected: KExpr<T>) {
            equalityChecks.add(EqualityCheck(actual = actual as KExpr<KSort>, expected = expected as KExpr<KSort>))
        }

        fun check(timeout: Duration, message: () -> String) = with(ctx) {
            KZ3Solver(ctx).use { solver ->
                val bulkCheck = mkOr(equalityChecks.map { !(it.actual eq it.expected) })
                solver.assert(bulkCheck)
                when (solver.check(timeout)) {
                    KSolverStatus.UNSAT -> return
                    KSolverStatus.SAT -> assertTrue(false, message())
                    KSolverStatus.UNKNOWN -> {
                        val testIgnoreReason = "equality check: unknown -- ${solver.reasonOfUnknown()}"
                        System.err.println(testIgnoreReason)
                        Assumptions.assumeTrue(false, testIgnoreReason)
                    }
                }
            }
        }

        private fun findFirstFailedEquality(solver: KZ3Solver): EqualityCheck? = with(ctx) {
            for (check in equalityChecks) {
                solver.push()
                val binding = !(check.actual eq check.expected)
                solver.assert(binding)
                val status = solver.check()
                solver.pop()
                if (status == KSolverStatus.SAT) return check
            }
            return null
        }
    }
}
