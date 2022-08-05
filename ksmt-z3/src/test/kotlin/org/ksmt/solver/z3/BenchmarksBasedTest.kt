package org.ksmt.solver.z3

import com.microsoft.z3.Context
import com.microsoft.z3.Expr
import com.microsoft.z3.Model
import com.microsoft.z3.Native
import com.microsoft.z3.Solver
import com.microsoft.z3.Status
import org.junit.jupiter.api.Assumptions
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.expr.KTransformer
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.fixtures.TestDataProvider
import org.ksmt.solver.fixtures.skipUnsupportedSolverFeatures
import org.ksmt.solver.fixtures.z3.Z3SmtLibParser
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import java.nio.file.Path
import kotlin.io.path.relativeTo
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds
import kotlin.time.DurationUnit

class BenchmarksBasedTest {

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("testData")
    fun testConverter(name: String, samplePath: Path) = skipUnsupportedSolverFeatures {
        val ctx = KContext()
        Context().use { parseCtx ->
            val assertions = parser.parseFile(parseCtx, samplePath)
            val ksmtAssertions = parser.convert(ctx, assertions)

            parseCtx.performEqualityChecks(ctx) {
                for ((originalZ3Expr, ksmtExpr) in assertions.zip(ksmtAssertions)) {
                    val internalizedExpr = internalize(ksmtExpr)
                    areEqual(actual = internalizedExpr, expected = originalZ3Expr)
                }
                check { "expressions are not equal" }
            }
        }
    }

    @EnabledIfEnvironmentVariable(
        named = "z3.testSolver",
        matches = "enabled",
        disabledReason = "z3 solver test"
    )
    @Execution(ExecutionMode.CONCURRENT)
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
                        add("random_seed", RANDOM_SEED)
                    }
                    setParameters(params)
                }
                solver.add(*assertions.toTypedArray())
                val status = solver.check()
                when (status) {
                    Status.SATISFIABLE -> KSolverStatus.SAT to solver.model
                    Status.UNSATISFIABLE -> KSolverStatus.UNSAT to null
                    Status.UNKNOWN, null -> {
                        Assumptions.assumeTrue(false, "expected status: unknown")
                        return
                    }
                }
            }

            val ksmtAssertions = parser.convert(ctx, assertions)
            parseCtx.runKsmtZ3Solver(ctx, ksmtAssertions, expectedStatus, expectedModel)
        }
    }

    private fun Context.runKsmtZ3Solver(
        ctx: KContext,
        ksmtAssertions: List<KExpr<KBoolSort>>,
        expectedStatus: KSolverStatus,
        expectedModel: Model?
    ) = TestZ3Solver(ctx).use { solver ->
        ksmtAssertions.forEach { solver.assert(it) }
        // use greater timeout to avoid false-positive unknowns
        val status = solver.check(timeout = 2.seconds)
        val message by lazy {
            val failInfo = if (status == KSolverStatus.UNKNOWN) " -- ${solver.reasonOfUnknown()}" else ""
            "solver check-sat mismatch$failInfo"
        }
        assertEquals(expectedStatus, status, message)

        if (status != KSolverStatus.SAT || expectedModel == null) return

        checkModelAssignments(
            ctx = ctx,
            expectedModel = expectedModel,
            actualModel = wrapModel(ctx, expectedModel)
        )
    }

    private fun Context.checkModelAssignments(
        ctx: KContext,
        expectedModel: Model,
        actualModel: KZ3Model
    ) {
        // check no exceptions during model detach
        val detachedModel = actualModel.detach()
        checkAsArrayDeclsPresentInModel(ctx, detachedModel)

        val expectedModelAssignments = run {
            val internCtx = KZ3InternalizationContext()
            val converter = KZ3ExprConverter(ctx, internCtx)
            val z3Constants = expectedModel.constDecls.map { mkConst(it) }
            val z3Functions = expectedModel.funcDecls.map { decl ->
                val args = decl.domain.map { mkFreshConst("x", it) }
                decl.apply(*args.toTypedArray())
            }
            val z3ModelKeys = z3Constants + z3Functions
            val assignments = z3ModelKeys.associateWith { expectedModel.eval(it, false) }
            with(converter) { assignments.map { (const, value) -> const.convert<KSort>() to value } }
        }
        val assignmentsToCheck = expectedModelAssignments.map { (const, expectedValue) ->
            val actualValue = actualModel.eval(const, complete = false)
            Triple(const, expectedValue, actualValue)
        }

        performEqualityChecks(ctx) {
            for ((_, expectedValue, actualValue) in assignmentsToCheck) {
                areEqual(actual = internalize(actualValue), expected = expectedValue)
            }
            check { "model assignments are not equal" }
        }
    }

    private fun Context.performEqualityChecks(
        ctx: KContext,
        checks: EqualityChecker.() -> Unit
    ) {
        val solver = mkSolver()
        val params = mkParams().apply {
            add("timeout", 1.seconds.toInt(DurationUnit.MILLISECONDS))
            add("model", false)
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

        fun internalize(expr: KExpr<*>): Expr<*> = with(internalizer) { expr.internalize() }

        private val equalityChecks = mutableListOf<Pair<Expr<*>, Expr<*>>>()

        fun areEqual(actual: Expr<*>, expected: Expr<*>) {
            equalityChecks.add(actual to expected)
        }

        fun check(message: () -> String) {
            val equalityBindings = equalityChecks.map { ctx.mkNot(ctx.mkEq(it.first, it.second)) }
            solver.add(ctx.mkOr(*equalityBindings.toTypedArray()))
            val status = solver.check()
            when (status) {
                Status.UNSATISFIABLE -> return
                Status.SATISFIABLE -> {
                    val (actual, expected) = findFirstFailedEquality()
                    if (actual != null && expected != null) {
                        assertEquals(expected, actual, message())
                    }
                    assertTrue(false, message())
                }
                null, Status.UNKNOWN -> {
                    val testIgnoreReason = "equality check: unknown -- ${solver.reasonUnknown}"
                    System.err.println(testIgnoreReason)
                    Assumptions.assumeTrue(false, testIgnoreReason)
                }
            }
        }

        private fun findFirstFailedEquality(): Pair<Expr<*>?, Expr<*>?> {
            for ((actual, expected) in equalityChecks) {
                solver.push()
                val binding = ctx.mkNot(ctx.mkEq(actual, expected))
                solver.add(binding)
                val status = solver.check()
                solver.pop()
                if (status == Status.SATISFIABLE) return actual to expected
            }
            return null to null
        }
    }

    companion object {
        const val RANDOM_SEED = 12345
        val parser = Z3SmtLibParser()

        init {
            // Limit z3 native memory usage to avoid OOM
            Native.globalParamSet("memory_max_size", "8192") // 8192 megabytes (hard limit)
            Native.globalParamSet("memory_high_watermark", "${2047 * 1024 * 1024}") // 2047 megabytes
        }

        @JvmStatic
        fun testData(): List<Arguments> {
            val testDataLocation = TestDataProvider.testDataLocation()
            return TestDataProvider.testData().map {
                Arguments.of(it.relativeTo(testDataLocation).toString(), it)
            }
        }
    }

    private class TestZ3Solver(ctx: KContext, override val randomSeed: Int = RANDOM_SEED) : KZ3Solver(ctx)

    private fun Context.wrapModel(ctx: KContext, model: Model): KZ3Model {
        val internCtx = KZ3InternalizationContext()
        val sortInternalizer = KZ3SortInternalizer(this, internCtx)
        val declInternalizer = KZ3DeclInternalizer(this, internCtx, sortInternalizer)
        val internalizer = KZ3ExprInternalizer(ctx, this, internCtx, sortInternalizer, declInternalizer)
        val converter = KZ3ExprConverter(ctx, internCtx)
        return KZ3Model(model, ctx, internCtx, internalizer, converter)
    }

    private fun checkAsArrayDeclsPresentInModel(ctx: KContext, model: KModel) {
        val checker = AsArrayDeclChecker(ctx, model)
        model.declarations.forEach { decl ->
            model.interpretation(decl)?.let { interpretation ->
                interpretation.entries.forEach { it.value.accept(checker) }
                interpretation.default?.accept(checker)
            }
        }
    }

    private class AsArrayDeclChecker(override val ctx: KContext, val model: KModel) : KTransformer {
        override fun <D : KSort, R : KSort> transform(expr: KFunctionAsArray<D, R>): KExpr<KArraySort<D, R>> {
            assertNotNull(model.interpretation(expr.function), "no interpretation for as-array: $expr")
            return expr
        }
    }
}
