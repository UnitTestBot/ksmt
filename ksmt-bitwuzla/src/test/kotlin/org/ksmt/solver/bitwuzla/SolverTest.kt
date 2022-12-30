package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.solver.KSolverException
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaKind
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.utils.getValue
import kotlin.test.Ignore
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds

class SolverTest {
    private val ctx = KContext()
    private val solver = KBitwuzlaSolver(ctx)

    @Ignore
    @Test
    fun testAbortHandling(): Unit = with(ctx) {
        val x by mkBv32Sort()
        val array by mkArraySort(mkBv32Sort(), mkBv32Sort())
        val body = array.select(x) eq x
        val quantifier = mkUniversalQuantifier(body, listOf(x.decl))
        solver.assert(quantifier)
        assertFailsWith(KSolverException::class) {
            solver.check()
        }
    }

    @Test
    fun testUnsatCoreGeneration(): Unit = with(ctx) {
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

    @Test
    fun testPushPopAndAssumptions(): Unit = with(ctx) {
        val a by boolSort
        solver.assert(a)
        solver.push()
        val track = solver.assertAndTrack(!a)
        var status = solver.checkWithAssumptions(listOf(a))
        assertEquals(KSolverStatus.UNSAT, status)
        val core = solver.unsatCore()
        assertEquals(2, core.size)
        assertTrue(track in core)
        assertTrue(a in core)
        solver.pop()
        status = solver.check()
        assertEquals(KSolverStatus.SAT, status)
    }

    @Test
    @Suppress("ForbiddenComment") // todo: don't use native api
    fun testTimeout(): Unit = with(ctx) {
        val arrayBase by mkArraySort(mkBv32Sort(), mkBv8Sort())
        val x by mkBv8Sort()

        var array = arrayBase
        for (i in 0..1024) {
            val v = mkBv((i xor 1024).toByte())
            array = array.store(mkBv(4198400 + i), v)
        }

        var xoredX = with(solver.exprInternalizer) { x.internalize() }
        for (i in 0..3000) {
            val selectedValue = array.select(mkBv(4198500 + i))
            val selectedValueTerm = with(solver.exprInternalizer) { selectedValue.internalize() }
            xoredX = Native.bitwuzlaMkTerm2(
                solver.bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_BV_XOR, xoredX, selectedValueTerm
            )
        }
        val someRandomValue = with(solver.exprInternalizer) { mkBv(42.toByte()).internalize() }
        val assertion = Native.bitwuzlaMkTerm2(
            solver.bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_EQUAL, xoredX, someRandomValue
        )
        Native.bitwuzlaAssert(solver.bitwuzlaCtx.bitwuzla, assertion)

        val status = solver.check(timeout = 1.milliseconds)

        assertEquals(KSolverStatus.UNKNOWN, status)
    }

    @Test
    fun testFpSupport(): Unit = with(ctx) {
        val a by fp32Sort
        val expr = mkFpGreaterExpr(a, 0.0f.expr) and mkFpLessExpr(a, 10.0f.expr)
        solver.assert(expr)
        val status = solver.check()
        assertEquals(KSolverStatus.SAT, status)
    }

}
