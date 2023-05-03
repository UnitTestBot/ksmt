package io.ksmt.test

import io.ksmt.KContext
import io.ksmt.KContext.SimplificationMode.NO_SIMPLIFY
import io.ksmt.expr.KExpr
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.bitwuzla.KBitwuzlaSolver
import io.ksmt.solver.yices.KYicesSolver
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KSort
import kotlin.test.Test
import kotlin.test.assertEquals

class TestBvOverflowChecks {

    @Test
    fun testBvAddNoOverflowSignedZ3() = testBoolOperation({ KZ3Solver(it) }) {
        bitwuzlaSampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvAddNoOverflowExpr(l, r, isSigned = true)
        }
    }

    @Test
    fun testBvAddNoUnderflowSignedZ3() = testBoolOperation({ KZ3Solver(it) }) {
        bitwuzlaSampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvAddNoUnderflowExpr(l, r)
        }
    }

    @Test
    fun testBvAddNoOverflowUnsignedZ3() = testBoolOperation({ KZ3Solver(it) }) {
        bitwuzlaSampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvAddNoOverflowExpr(l, r, isSigned = false)
        }
    }

    @Test
    fun testBvSubNoOverflowSignedZ3() = testBoolOperation({ KZ3Solver(it) }) {
        bitwuzlaSampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvSubNoOverflowExpr(l, r)
        }
    }

    @Test
    fun testBvSubNoUnderflowSignedZ3() = testBoolOperation({ KZ3Solver(it) }) {
        bitwuzlaSampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvSubNoUnderflowExpr(l, r, isSigned = true)
        }
    }

    @Test
    fun testBvSubNoUnderflowUnsignedZ3() = testBoolOperation({ KZ3Solver(it) }) {
        bitwuzlaSampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvSubNoUnderflowExpr(l, r, isSigned = false)
        }
    }

    @Test
    fun testBvMulNoOverflowSignedZ3() = testBoolOperation({ KZ3Solver(it) }) {
        bitwuzlaSampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvMulNoOverflowExpr(l, r, isSigned = true)
        }
    }

    @Test
    fun testBvMulNoUnderflowSignedZ3() = testBoolOperation({ KZ3Solver(it) }) {
        bitwuzlaSampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvMulNoUnderflowExpr(l, r)
        }
    }

    @Test
    fun testBvMulNoOverflowUnsignedZ3() = testBoolOperation({ KZ3Solver(it) }) {
        bitwuzlaSampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvMulNoOverflowExpr(l, r, isSigned = false)
        }
    }

    @Test
    fun testBvDivNoOverflowSignedZ3() = testBoolOperation({ KZ3Solver(it) }) {
        bitwuzlaSampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvDivNoOverflowExpr(l, r)
        }
    }

    @Test
    fun testBvAddNoOverflowSignedBitwuzla() = testBoolOperation({ KBitwuzlaSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvAddNoOverflowExpr(l, r, isSigned = true)
        }
    }

    @Test
    fun testBvAddNoUnderflowSignedBitwuzla() = testBoolOperation({ KBitwuzlaSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvAddNoUnderflowExpr(l, r)
        }
    }

    @Test
    fun testBvAddNoOverflowUnsignedBitwuzla() = testBoolOperation({ KBitwuzlaSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvAddNoOverflowExpr(l, r, isSigned = false)
        }
    }

    @Test
    fun testBvSubNoOverflowSignedBitwuzla() = testBoolOperation({ KBitwuzlaSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvSubNoOverflowExpr(l, r)
        }
    }

    @Test
    fun testBvSubNoUnderflowSignedBitwuzla() = testBoolOperation({ KBitwuzlaSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvSubNoUnderflowExpr(l, r, isSigned = true)
        }
    }

    @Test
    fun testBvSubNoUnderflowUnsignedBitwuzla() = testBoolOperation({ KBitwuzlaSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvSubNoUnderflowExpr(l, r, isSigned = false)
        }
    }

    @Test
    fun testBvMulNoOverflowSignedBitwuzla() = testBoolOperation({ KBitwuzlaSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvMulNoOverflowExpr(l, r, isSigned = true)
        }
    }

    @Test
    fun testBvMulNoUnderflowSignedBitwuzla() = testBoolOperation({ KBitwuzlaSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvMulNoUnderflowExpr(l, r)
        }
    }

    @Test
    fun testBvMulNoOverflowUnsignedBitwuzla() = testBoolOperation({ KBitwuzlaSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvMulNoOverflowExpr(l, r, isSigned = false)
        }
    }

    @Test
    fun testBvDivNoOverflowSignedBitwuzla() = testBoolOperation({ KBitwuzlaSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvDivNoOverflowExpr(l, r)
        }
    }

    @Test
    fun testBvAddNoOverflowSignedYices() = testBoolOperation({ KYicesSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvAddNoOverflowExpr(l, r, isSigned = true)
        }
    }

    @Test
    fun testBvAddNoUnderflowSignedYices() = testBoolOperation({ KYicesSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvAddNoUnderflowExpr(l, r)
        }
    }

    @Test
    fun testBvAddNoOverflowUnsignedYices() = testBoolOperation({ KYicesSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvAddNoOverflowExpr(l, r, isSigned = false)
        }
    }

    @Test
    fun testBvSubNoOverflowSignedYices() = testBoolOperation({ KYicesSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvSubNoOverflowExpr(l, r)
        }
    }

    @Test
    fun testBvSubNoUnderflowSignedYices() = testBoolOperation({ KYicesSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvSubNoUnderflowExpr(l, r, isSigned = true)
        }
    }

    @Test
    fun testBvSubNoUnderflowUnsignedYices() = testBoolOperation({ KYicesSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvSubNoUnderflowExpr(l, r, isSigned = false)
        }
    }

    @Test
    fun testBvMulNoOverflowSignedYices() = testBoolOperation({ KYicesSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvMulNoOverflowExpr(l, r, isSigned = true)
        }
    }

    @Test
    fun testBvMulNoUnderflowSignedYices() = testBoolOperation({ KYicesSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvMulNoUnderflowExpr(l, r)
        }
    }

    @Test
    fun testBvMulNoOverflowUnsignedYices() = testBoolOperation({ KYicesSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvMulNoOverflowExpr(l, r, isSigned = false)
        }
    }

    @Test
    fun testBvDivNoOverflowSignedYices() = testBoolOperation({ KYicesSolver(it) }) {
        z3SampleBinaryBoolExprValues(bv32Sort) { l, r ->
            mkBvDivNoOverflowExpr(l, r)
        }
    }

    private fun <T : KSort> KContext.z3SampleBinaryBoolExprValues(
        sort: T,
        mkExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
    ): List<BinaryExprSample<T, KBoolSort>> = KZ3Solver(this).use { solver ->
        sampleBinaryBoolExprValues(solver, sort) { l, r -> mkExpr(l, r) }
    }

    private fun <T : KSort> KContext.bitwuzlaSampleBinaryBoolExprValues(
        sort: T,
        mkExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
    ): List<BinaryExprSample<T, KBoolSort>> = KBitwuzlaSolver(this).use { solver ->
        sampleBinaryBoolExprValues(solver, sort) { l, r -> mkExpr(l, r) }
    }

    private fun <T : KSort> KContext.sampleBinaryBoolExprValues(
        solver: KSolver<*>,
        sort: T,
        mkExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
    ): List<BinaryExprSample<T, KBoolSort>> {
        val positive = sampleBinaryExprValues(solver, sort) { l, r ->
            mkExpr(l, r)
        }
        val negative = sampleBinaryExprValues(solver, sort) { l, r ->
            mkExpr(l, r).not()
        }
        return positive + negative
    }

    private fun <T : KSort, R : KSort> KContext.sampleBinaryExprValues(
        solver: KSolver<*>,
        sort: T,
        mkExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<R>
    ): List<BinaryExprSample<T, R>> {
        val lhs = mkFreshConst("lhs", sort)
        val rhs = mkFreshConst("rhs", sort)
        val expr = mkExpr(lhs, rhs)
        val result = mkFreshConst("result", expr.sort)
        val samples = arrayListOf<BinaryExprSample<T, R>>()

        solver.assert(result eq expr)

        while (solver.check() == KSolverStatus.SAT && samples.size < NUM_SAMPLES) {
            val model = solver.model()

            val lhsValue = model.eval(lhs)
            val rhsValue = model.eval(rhs)
            val resultValue = model.eval(result)
            samples += BinaryExprSample(mkExpr, lhsValue, rhsValue, resultValue)

            solver.assert((lhs neq lhsValue) or (rhs neq rhsValue))
        }

        return samples
    }

    private fun <T : KSort> testBoolOperation(
        mkSolver: (KContext) -> KSolver<*>,
        mkSamples: KContext.() -> List<BinaryExprSample<T, KBoolSort>>
    ) = with(KContext(simplificationMode = NO_SIMPLIFY)) {
        val samples = mkSamples()
        mkSolver(this).use { solver ->
            samples.forEach { sample ->
                testOperationSample(solver, sample)
            }
        }
    }

    private fun <T : KSort, R : KSort> KContext.testOperationSample(
        solver: KSolver<*>,
        sample: BinaryExprSample<T, R>
    ) = try {
        solver.push()
        val operationValue = sample.operation(this, sample.lhs, sample.rhs)
        solver.assert(operationValue neq sample.result)

        val status = solver.check()
        assertEquals(KSolverStatus.UNSAT, status)
    } finally {
        solver.pop()
    }

    data class BinaryExprSample<T : KSort, R : KSort>(
        val operation: KContext.(KExpr<T>, KExpr<T>) -> KExpr<R>,
        val lhs: KExpr<T>,
        val rhs: KExpr<T>,
        val result: KExpr<R>
    )

    companion object {
        private const val NUM_SAMPLES = 10
    }
}
