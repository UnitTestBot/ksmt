package io.ksmt.solver.yices

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KSolverException
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KBv32Sort
import io.ksmt.utils.getValue
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
        solver.assertAndTrack(e2)
        val status = solver.checkWithAssumptions(listOf(e3))
        assertEquals(KSolverStatus.UNSAT, status)
        val core = solver.unsatCore()
        assertEquals(2, core.size)
        assertTrue(e2 in core)
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
        solver.assertAndTrack(e2)
        val status = solver.check()
        assertEquals(KSolverStatus.UNSAT, status)
        val core = solver.unsatCore()
        assertEquals(1, core.size)
        assertTrue(e2 in core)
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

        solver.assertAndTrack(!a)

        solver.push()

        solver.assertAndTrack(!b)

        var status = solver.check()
        assertEquals(KSolverStatus.UNSAT, status)
        var core = solver.unsatCore()
        assertEquals(2, core.size)
        assertTrue(!a in core && !b in core)

        solver.pop()

        solver.assert(a)

        status = solver.check()
        assertEquals(KSolverStatus.UNSAT, status)
        core = solver.unsatCore()
        assertEquals(1, core.size)
        assertTrue(!a in core)

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

        var array: KExpr<KArraySort<KBv32Sort, KBv32Sort>> = arrayBase
        for (i in 0..1024) {
            val v = mkBv((i xor 1024))
            array = array.store(mkBv(4198400 + i), v)
        }

        var xoredX: KExpr<KBv32Sort> = x
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

    @Test
    fun testUninterpretedSortValueModel(): Unit = with(ctx) {
        KYicesSolver(this).use { solver ->
            val sort = mkUninterpretedSort("sort")

            val value = mkUninterpretedSortValue(sort, 0)
            val a by sort

            solver.assert(a neq value)
            solver.check()

            val model = solver.model()
            val sortValues = model.uninterpretedSortUniverse(sort)!!

            assertTrue { value in sortValues }
            assertTrue { sortValues.size >= 2 }
        }
    }

    @Test
    fun testUninterpretedSortMultipleValueModel(): Unit = with(ctx) {
        KYicesSolver(this).use { solver ->
            val s1 = mkUninterpretedSort("s1")
            val s2 = mkUninterpretedSort("s2")

            val s1value = mkUninterpretedSortValue(s1, 1)
            val s2value = mkUninterpretedSortValue(s2, 1)
            val s1Var by s1
            val s2Var by s2

            solver.assert(s1Var neq s1value)
            solver.assert(s2Var neq s2value)
            solver.check()

            val model = solver.model()
            val s1Values = model.uninterpretedSortUniverse(s1)!!
            val s2Values = model.uninterpretedSortUniverse(s2)!!

            assertTrue { s1value in s1Values }
            assertTrue { s1Values.size >= 2 }

            assertTrue { s2value in s2Values }
            assertTrue { s2Values.size >= 2 }
        }
    }
}
