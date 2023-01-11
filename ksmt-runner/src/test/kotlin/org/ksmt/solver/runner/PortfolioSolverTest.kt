package org.ksmt.solver.runner

import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.BeforeEach
import org.ksmt.KContext
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.bitwuzla.KBitwuzlaSolver
import org.ksmt.solver.portfolio.KPortfolioSolver
import org.ksmt.solver.portfolio.KPortfolioSolverManager
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.utils.getValue
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class PortfolioSolverTest {
    private lateinit var context: KContext
    private lateinit var solver: KPortfolioSolver

    @BeforeEach
    fun createNewEnvironment() {
        context = KContext()
        solver = solverManager.createPortfolioSolver(context)
    }

    @AfterEach
    fun clearResources() {
        solver.close()
        context.close()
    }

    companion object {
        private lateinit var solverManager: KPortfolioSolverManager

        @BeforeAll
        @JvmStatic
        fun initSolverManager() {
            solverManager = KPortfolioSolverManager(listOf(KZ3Solver::class, KBitwuzlaSolver::class))
        }

        @AfterAll
        @JvmStatic
        fun closeSolverManager() {
            solverManager.close()
        }
    }

    @Test
    fun testUnsatCoreGeneration(): Unit = with(context) {
        val a by boolSort
        val b by boolSort
        val c by boolSort

        val e1 = (a and b) or c
        val e2 = !(a and b)
        val e3 = !c

        solver.assert(e1)
        val e2Track = solver.assertAndTrack(e2)
        val status = solver.checkWithAssumptions(listOf(e3))

        assertEquals(KSolverStatus.UNSAT, status)

        val core = solver.unsatCore()

        assertEquals(2, core.size)
        assertTrue(e2Track in core)
        assertTrue(e3 in core)
    }

    @Test
    fun testUnsatCoreGenerationNoAssumptions(): Unit = with(context) {
        val a by boolSort
        val b by boolSort

        val e1 = (a and b)
        val e2 = !(a and b)

        solver.assert(e1)
        val e2Track = solver.assertAndTrack(e2)
        val status = solver.check()

        assertEquals(KSolverStatus.UNSAT, status)

        val core = solver.unsatCore()

        assertEquals(1, core.size)
        assertTrue(e2Track in core)
    }

    @Test
    fun testPushPop(): Unit = with(context) {
        val a by boolSort

        solver.assert(a)
        solver.push()
        val track = solver.assertAndTrack(!a)
        var status = solver.check()

        assertEquals(KSolverStatus.UNSAT, status)

        val core = solver.unsatCore()

        assertEquals(1, core.size)
        assertTrue(track in core)

        solver.pop()
        status = solver.check()

        assertEquals(KSolverStatus.SAT, status)
    }

    @Test
    fun testUnsupportedFeaturesHandling(): Unit = with(context) {
        val a by intSort

        solver.assert(a eq 0.expr)
        solver.assert(a eq 1.expr)

        val status = solver.check()

        assertEquals(KSolverStatus.UNSAT, status)
    }

}
