package io.ksmt.solver.cvc5

import io.ksmt.KContext
import io.ksmt.expr.KApp
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KBv32Sort
import io.ksmt.sort.KBv8Sort
import io.ksmt.utils.getValue
import io.ksmt.utils.mkConst
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds

class IncrementalApiTest {
    private val ctx = KContext()
    private val solver = KCvc5Solver(ctx)

    @Test
    fun testScopedSorts(): Unit = with(ctx) {
        val sort1 = mkUninterpretedSort("us")
        val sort2 = mkUninterpretedSort("usus")
        val sort3 = mkUninterpretedSort("ususus")

        val a1 = sort1.mkConst("us_a")
        val b1 = sort1.mkConst("us_b")

        val a2 = sort2.mkConst("usus_a")
        val b2 = sort2.mkConst("usus_b")

        val a3 = sort3.mkConst("ususus_a")
        val b3 = sort3.mkConst("ususus_b")

        val f1 = a1 neq b1
        val f2 = a2 neq b2
        val f3 = a3 neq b3

        solver.assert(mkTrue())
        solver.push() // => level 1

        // let context know about exprs
        solver.assert(f1)
        solver.assert(f2)
        solver.assert(f3)

        solver.pop() // => level 0

        solver.check().also { assertEquals(KSolverStatus.SAT, it) }
        solver.model().also { model -> assertFalse { sort1 in model.uninterpretedSorts } }

        solver.assert(f1)

        solver.push() // => level 1
        solver.push() // => level 2

        solver.assert(f2)

        solver.pop() // => level 1
        solver.pop() // => level 0
        solver.push() // => level 1
        solver.push() // => level 2

        solver.assert(f3)

        solver.check().also { assertEquals(KSolverStatus.SAT, it) }
        solver.model().also { model ->
            assertContains(model.uninterpretedSorts, sort1)
            assertFalse { sort2 in model.uninterpretedSorts }
            assertContains(model.uninterpretedSorts, sort1)
        }
    }

    @Test
    fun testScopedDeclsAndSorts(): Unit = with(ctx) {
        val sort = mkUninterpretedSort("us")

        val a = sort.mkConst("a")
        val b = sort.mkConst("b")
        val c = sort.mkConst("c")

        solver.assert(a neq b)
        solver.push() // => level 1

        solver.check().also { assertEquals(KSolverStatus.SAT, it) }
        solver.model().also { model ->
            assertContains(model.declarations, a.decl)
            assertContains(model.declarations, b.decl)
            assertFalse { c.decl in model.declarations }

            assertContains(model.uninterpretedSorts, sort)
        }

        solver.pop() // => level 0

        solver.check().also { assertEquals(KSolverStatus.SAT, it) }
        solver.model().also { model ->
            assertContains(model.declarations, a.decl)
            assertContains(model.declarations, b.decl)
            assertFalse { c.decl in model.declarations }

            assertContains(model.uninterpretedSorts, sort)
        }

        solver.push() // => level 1
        solver.push() // => level 2

        solver.assert(b neq c)

        solver.check().also { assertEquals(KSolverStatus.SAT, it) }
        solver.model().also { model ->
            assertContains(model.declarations, a.decl)
            assertContains(model.declarations, b.decl)
            assertContains(model.declarations, c.decl)

            assertContains(model.uninterpretedSorts, sort)
        }

        solver.pop() // => level 1
        solver.pop() // => level 0

        solver.check().also { assertEquals(KSolverStatus.SAT, it) }
        solver.model().also { model ->
            assertContains(model.declarations, a.decl)
            assertContains(model.declarations, b.decl)
            assertFalse { c.decl in model.declarations }

            assertContains(model.uninterpretedSorts, sort)
        }
    }

    @Test
    fun testUnsatCoreGeneration(): Unit = with(ctx) {
        val a = boolSort.mkConst("a")
        val b = boolSort.mkConst("b")
        val c = boolSort.mkConst("c")

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
        val a = boolSort.mkConst("a")
        val b = boolSort.mkConst("b")

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
        val a = boolSort.mkConst("a")
        solver.assert(a)
        solver.push()
        solver.assertAndTrack(!a)
        var status = solver.check()
        assertEquals(KSolverStatus.UNSAT, status)
        val core = solver.unsatCore()
        assertEquals(1, core.size)
        assertTrue(!a in core)
        solver.pop()
        status = solver.check()
        assertEquals(KSolverStatus.SAT, status)
    }

    @Test
    fun testTimeout(): Unit = with(ctx) {
        val array by mkArraySort(intSort, mkArraySort(intSort, intSort))
        val result by mkArraySort(intSort, intSort)

        val i = intSort.mkConst("i")
        val j = intSort.mkConst("i")
        val idx = mkIte((i mod j) eq mkIntNum(100), i, j)
        val body = result.select(idx) eq array.select(i).select(j)
        val rule = mkUniversalQuantifier(body, listOf(i.decl, j.decl))
        solver.assert(rule)

        val x = intSort.mkConst("x")
        val queryBody = result.select(x) gt result.select(x + mkIntNum(10))
        val query = mkUniversalQuantifier(queryBody, listOf(x.decl))
        solver.assert(query)

        val status = solver.checkWithAssumptions(emptyList(), timeout = 0.5.milliseconds)
        assertEquals(KSolverStatus.UNKNOWN, status)
    }

    @Test
    fun testTimeout2(): Unit = with(ctx) {
        val srt = mkArraySort(mkBv32Sort(), mkBv8Sort())
        val arrayBase = mkConst("arr", srt)
        val x = mkConst("x", bv8Sort)

        var array = arrayBase
        for (i in 0..1024) {
            val v = mkBv((i xor 1024).toByte())
            array = array.store(mkBv(4198400 + i), v) as KApp<KArraySort<KBv32Sort, KBv8Sort>, *>
        }

        var xoredX = x
        // ~1927 is max til crash
        for (i in 0..1000) {
            val selectedValue = array.select(mkBv(4198500 + i))
            xoredX = mkBvXorExpr(xoredX, selectedValue) as KApp<KBv8Sort, *>
        }

        val someRandomValue = mkBv(42.toByte())

        solver.assert(xoredX eq someRandomValue)

        val status = solver.check(timeout = 1.milliseconds)

        assertEquals(KSolverStatus.UNKNOWN, status)
    }

}
