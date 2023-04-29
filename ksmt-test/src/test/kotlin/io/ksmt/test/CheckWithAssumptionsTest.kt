package io.ksmt.test

import io.ksmt.KContext
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.bitwuzla.KBitwuzlaSolver
import io.ksmt.solver.cvc5.KCvc5Solver
import io.ksmt.solver.yices.KYicesSolver
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.utils.getValue
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CheckWithAssumptionsTest {

    @Test
    fun testComplexAssumptionZ3() = testComplexAssumption { KZ3Solver(it) }

    @Test
    fun testComplexAssumptionBitwuzla() = testComplexAssumption { KBitwuzlaSolver(it) }

    @Test
    fun testComplexAssumptionYices() = testComplexAssumption { KYicesSolver(it) }

    @Test
    fun testComplexAssumptionCvc() = testComplexAssumption { KCvc5Solver(it) }

    @Test
    fun testUnsatCoreGenerationZ3() = testUnsatCoreGeneration { KZ3Solver(it) }

    @Test
    fun testUnsatCoreGenerationBitwuzla() = testUnsatCoreGeneration { KBitwuzlaSolver(it) }

    @Test
    fun testUnsatCoreGenerationYices() = testUnsatCoreGeneration { KYicesSolver(it) }

    @Test
    fun testUnsatCoreGenerationCvc() = testUnsatCoreGeneration { KCvc5Solver(it) }

    private fun testComplexAssumption(mkSolver: (KContext) -> KSolver<*>) = with(KContext()) {
        mkSolver(this).use { solver ->
            val a by bv32Sort
            val b by bv32Sort

            solver.assert(a eq mkBv(0))
            solver.assert(b eq mkBv(0))

            val expr = mkBvUnsignedGreaterExpr(a, b)
            val status = solver.checkWithAssumptions(listOf(expr))

            assertEquals(KSolverStatus.UNSAT, status)
        }
    }

    private fun testUnsatCoreGeneration(mkSolver: (KContext) -> KSolver<*>) = with(KContext()) {
        mkSolver(this).use { solver ->
            val a by boolSort
            val b by boolSort
            val c by boolSort

            val e1 = (a and b) or c
            val e2 = !(a and b)
            val e3 = !c

            solver.assert(e1)
            solver.assertAndTrack(e2)

            val status = solver.checkWithAssumptions(listOf(e3))
            assertEquals(KSolverStatus.UNSAT, status)

            val core = solver.unsatCore()
            assertEquals(2, core.size)

            assertTrue(e2 in core)
            assertTrue(e3 in core)
        }
    }
}
