package org.ksmt.solver.z3

import com.microsoft.z3.Context
import com.microsoft.z3.Expr
import com.microsoft.z3.Solver
import com.microsoft.z3.Status
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.fixtures.TestDataProvider
import org.ksmt.solver.fixtures.skipUnsupportedSolverFeatures
import org.ksmt.solver.fixtures.z3.Z3SmtLibParser
import org.ksmt.sort.KSort
import java.nio.file.Path
import kotlin.io.path.relativeTo
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds
import kotlin.time.DurationUnit

class BenchmarksBasedTest {

    @ParameterizedTest(name = "{0}")
    @MethodSource("testData")
    fun testConverter(name: String, samplePath: Path) = skipUnsupportedSolverFeatures {
        val ctx = KContext()
        Context().use { parseCtx ->
            val assertions = parser.parseFile(parseCtx, samplePath)
            val ksmtAssertions = parser.convert(ctx, assertions)

            Context().use { checkCtx ->
                checkCtx.performEqualityChecks(ctx) {
                    for ((originalZ3Expr, ksmtExpr) in assertions.zip(ksmtAssertions)) {
                        val internalizedExpr = internalize(ksmtExpr)
                        val z3Expr = originalZ3Expr.translate(checkCtx)
                        assertTrue("expressions are not equal") {
                            areEqual(internalizedExpr, z3Expr)
                        }
                    }
                }
            }
        }
    }

    @EnabledIfEnvironmentVariable(
        named = "z3.testSolver",
        matches = "enabled",
        disabledReason = "z3 solver test"
    )
    @ParameterizedTest(name = "{0}")
    @MethodSource("testData")
    fun testSolver(name: String, samplePath: Path) = skipUnsupportedSolverFeatures {
        val ctx = KContext()
        Context().use { parseCtx ->
            val assertions = parser.parseFile(parseCtx, samplePath)
            val (expectedStatus, expectedModel) = with(parseCtx) {
                val solver = mkSolver().apply {
                    val params = mkParams().apply {
                        add("timeout", 1.seconds.toInt(DurationUnit.MILLISECONDS))
                    }
                    setParameters(params)
                }
                solver.add(*assertions.toTypedArray())
                val status = solver.check()
                when (status) {
                    Status.SATISFIABLE -> KSolverStatus.SAT to solver.model
                    Status.UNSATISFIABLE -> KSolverStatus.UNSAT to null
                    Status.UNKNOWN, null -> return
                }
            }

            val ksmtAssertions = parser.convert(ctx, assertions)

            KZ3Solver(ctx).use { solver ->
                ksmtAssertions.forEach { solver.assert(it) }
                // use greater timeout to avoid false-positive unknowns
                val status = solver.check(timeout = 2.seconds)
                assertEquals(expectedStatus, status, "solver check-sat mismatch")

                if (status != KSolverStatus.SAT || expectedModel == null) return

                val model = solver.model()
                val expectedModelAssignments = run {
                    val internCtx = KZ3InternalizationContext()
                    val converter = KZ3ExprConverter(ctx, internCtx)
                    val z3Constants = expectedModel.decls.map { parseCtx.mkConst(it) }
                    val assignments = z3Constants.associateWith { expectedModel.eval(it, false) }
                    with(converter) { assignments.map { (const, value) -> const.convert<KSort>() to value } }
                }
                val assignmentsToCheck = expectedModelAssignments.map { (const, expectedValue) ->
                    val actualValue = model.eval(const, complete = false)
                    Triple(const, expectedValue, actualValue)
                }

                Context().use { checkCtx ->
                    checkCtx.performEqualityChecks(ctx) {
                        for ((const, expectedValue, actualValue) in assignmentsToCheck) {
                            val internalizedExpr = internalize(actualValue)
                            val z3Expr = expectedValue.translate(checkCtx)
                            assertTrue("model assignments for $const are not equal") {
                                areEqual(internalizedExpr, z3Expr)
                            }
                        }
                    }
                }
            }
        }
    }

    private fun Context.performEqualityChecks(ctx: KContext, checks: EqualityChecker.() -> Unit) {
        val solver = mkSolver()
        val params = mkParams().apply {
            add("timeout", 1.seconds.toInt(DurationUnit.MILLISECONDS))
        }
        solver.setParameters(params)
        val checker = EqualityChecker(this, solver, ctx)
        checker.checks()
    }

    private class EqualityChecker(
        private val ctx: Context,
        private val solver: Solver,
        ksmtCtx: KContext
    ) {
        private val internCtx = KZ3InternalizationContext()
        private val sortInternalizer = KZ3SortInternalizer(ctx, internCtx)
        private val declInternalizer = KZ3DeclInternalizer(ctx, internCtx, sortInternalizer)
        private val internalizer = KZ3ExprInternalizer(ksmtCtx, ctx, internCtx, sortInternalizer, declInternalizer)

        fun internalize(expr: KExpr<*>): Expr = with(internalizer) { expr.internalize() }

        fun areEqual(left: Expr, right: Expr): Boolean {
            solver.push()
            solver.add(ctx.mkNot(ctx.mkEq(left, right)))
            val status = solver.check()
            solver.pop()
            return status == Status.UNSATISFIABLE
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
}
