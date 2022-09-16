package org.ksmt.solver.z3

import org.ksmt.KContext
import org.ksmt.solver.KSolverStatus
import org.ksmt.sort.KArraySort
import kotlin.test.*

class Example {

    @Test
    @Suppress("USELESS_IS_CHECK")
    fun test() = with(KContext()) {
        val a = boolSort.mkConst("a")
        val b = mkFuncDecl("b", boolSort, listOf(intSort))
        val c = intSort.mkConst("c")
        val e = mkArraySort(intSort, intSort).mkConst("e")
        val solver = KZ3Solver(this)
        solver.assert(a)
        solver.assert(
            mkUniversalQuantifier(
                !(c gt 0.intExpr and !(c eq 17.intExpr)) or b.apply(listOf(c)), listOf(c.decl)
            )
        )
        solver.assert(
            mkUniversalQuantifier(
                !(c le 0.intExpr) or !b.apply(listOf(c)), listOf(c.decl)
            )
        )
        solver.assert(e.select(3.intExpr) ge 0.intExpr)
        val status = solver.check()
        assertEquals(status, KSolverStatus.SAT)
        val model = solver.model()
        val aValue = model.eval(a)
        val cValue = model.eval(c)
        val eValue = model.eval(e)
        assertEquals(aValue, trueExpr)
        assertEquals(cValue, c)
        assertTrue(eValue.sort is KArraySort<*, *>)
        val bInterp = model.interpretation(b)
        assertNotNull(bInterp)
        val detachedModel = model.detach()
        solver.close()
        assertFailsWith(IllegalStateException::class) { model.eval(a) }
        assertEquals(aValue, detachedModel.eval(a))
        assertEquals(cValue, detachedModel.eval(c))
        assertEquals(eValue, detachedModel.eval(e))
    }
}
