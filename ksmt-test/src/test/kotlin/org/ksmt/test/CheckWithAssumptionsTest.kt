package org.ksmt.test

import org.ksmt.KContext
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.bitwuzla.KBitwuzlaSolver
import org.ksmt.solver.cvc5.KCvc5Solver
import org.ksmt.solver.yices.KYicesSolver
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.utils.getValue
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
