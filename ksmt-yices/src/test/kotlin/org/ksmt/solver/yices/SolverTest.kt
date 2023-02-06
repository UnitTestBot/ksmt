package org.ksmt.solver.yices

import org.ksmt.KContext
import org.ksmt.solver.KSolverException
import org.ksmt.solver.KSolverStatus
import org.ksmt.utils.getValue
import kotlin.test.Test
import kotlin.test.Ignore
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds

class SolverTest {
    private val ctx = KContext()

    @Test
    fun testAbortHandling(): Unit = with(ctx) {
        val solver = KYicesSolver(ctx)
        val x by mkBv32Sort()
        val array by mkArraySort(mkBv32Sort(), mkBv32Sort())
        val body = array.select(x) eq x
        val quantifier = mkUniversalQuantifier(body, listOf(x.decl))
        assertFailsWith(KSolverException::class) {
            solver.assert(quantifier)
        }
    }

    @Test
    fun testUnsatCoreGeneration(): Unit = with(ctx) {
        val solver = KYicesSolver(ctx)
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
    fun testUnsatCoreGenerationNoAssumptions(): Unit = with(ctx) {
        val solver = KYicesSolver(ctx)
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
    fun testPushPop(): Unit = with(ctx) {
        val solver = KYicesSolver(ctx)
        val a by boolSort
        val b by boolSort
        val c by boolSort

        solver.assert(!c)
        solver.assert(a or b)

        solver.push()

        val track1 = solver.assertAndTrack(!a)

        solver.push()

        val track2 = solver.assertAndTrack(!b)

        var status = solver.check()
        assertEquals(KSolverStatus.UNSAT, status)
        var core = solver.unsatCore()
        assertEquals(2, core.size)
        assertTrue(track1 in core && track2 in core)

        solver.pop()

        solver.assert(a)

        status = solver.check()
        assertEquals(KSolverStatus.UNSAT, status)
        core = solver.unsatCore()
        assertEquals(1, core.size)
        assertTrue(track1 in core)

        solver.pop()

        status = solver.check()
        assertEquals(KSolverStatus.SAT, status)
    }

    @Ignore
    @Test
    fun testTimeout(): Unit = with(ctx) {
        val solver = KYicesSolver(ctx)
        val arrayBase by mkArraySort(mkBv32Sort(), mkBv32Sort())
        val x by mkBv32Sort()

        var array = arrayBase
        for (i in 0..1024) {
            val v = mkBv((i xor 1024))
            array = array.store(mkBv(4198400 + i), v)
        }

        var xoredX = x
        for (i in 0..1000) {
            val selectedValue = array.select(mkBv(4198500 + i))
            xoredX = mkBvXorExpr(xoredX, selectedValue)
            xoredX = mkBvOrExpr(mkBvUnsignedDivExpr(xoredX, mkBv(3)), mkBvUnsignedRemExpr(xoredX, mkBv(3)))
        }
        val someRandomValue = mkBv(42)
        val assertion = xoredX eq someRandomValue

        solver.assert(assertion)

        val status = solver.check(timeout = 1.milliseconds)

        assertEquals(KSolverStatus.UNKNOWN, status)
    }

}
