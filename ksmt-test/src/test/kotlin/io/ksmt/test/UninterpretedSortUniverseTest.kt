package io.ksmt.test

import io.ksmt.KContext
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.bitwuzla.KBitwuzlaSolver
import io.ksmt.solver.cvc5.KCvc5Solver
import io.ksmt.solver.yices.KYicesSolver
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.utils.getValue
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertNotNull

class UninterpretedSortUniverseTest {

    @Nested
    inner class UninterpretedSortUniverseTestBitwuzla {
        @Test
        fun testUniverseContainsValue() = testUniverseContainsValue(::mkBitwuzlaSolver)

        private fun mkBitwuzlaSolver(ctx: KContext) = KBitwuzlaSolver(ctx)
    }

    @Nested
    inner class UninterpretedSortUniverseTestCvc5 {
        @Test
        fun testUniverseContainsValue() = testUniverseContainsValue(::mkCvc5Solver)

        private fun mkCvc5Solver(ctx: KContext) = KCvc5Solver(ctx)
    }

    @Nested
    inner class UninterpretedSortUniverseTestYices {
        @Test
        fun testUniverseContainsValue() = testUniverseContainsValue(::mkYicesSolver)

        private fun mkYicesSolver(ctx: KContext) = KYicesSolver(ctx)
    }

    @Nested
    inner class UninterpretedSortUniverseTestZ3 {
        @Test
        fun testUniverseContainsValue() = testUniverseContainsValue(::mkZ3Solver)

        private fun mkZ3Solver(ctx: KContext) = KZ3Solver(ctx)
    }

    fun testUniverseContainsValue(mkSolver: (KContext) -> KSolver<*>): Unit =
        KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY).use { ctx ->
            mkSolver(ctx).use { s ->
                with(ctx) {
                    val u = mkUninterpretedSort("u")
                    val u1 by u
                    val uval5 = mkUninterpretedSortValue(u, 5)

                    s.assert(u1 neq uval5)
                    assertEquals(KSolverStatus.SAT, s.check())
                    val u1v = s.model().eval(u1)
                    assertNotEquals(uval5, u1v)

                    s.assert(u1 neq u1v)
                    assertEquals(KSolverStatus.SAT, s.check())

                    val universe = s.model().uninterpretedSortUniverse(u)
                    assertNotNull(universe)

                    assertContains(universe, u1v)
                    assertContains(universe, uval5)
                }
            }
        }
}
