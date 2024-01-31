package io.ksmt.solver.z3

import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import org.junit.jupiter.api.Test
import io.ksmt.KContext
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KArraySort
import io.ksmt.utils.mkConst

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
                !(c gt 0.expr and !(c eq 17.expr)) or b.apply(listOf(c)), listOf(c.decl)
            )
        )
        solver.assert(
            mkUniversalQuantifier(
                !(c le 0.expr) or !b.apply(listOf(c)), listOf(c.decl)
            )
        )
        solver.assert(e.select(3.expr) ge 0.expr)

        val bvVariable = mkBv32Sort().mkConst("A")
        val bvValue = mkBv(256)
        solver.assert(mkEq(bvValue, bvVariable))

        val status = solver.check()
        assertEquals(status, KSolverStatus.SAT)

        val model = solver.model()
        val aValue = model.eval(a)
        val cValue = model.eval(c)
        val eValue = model.eval(e)
        val bv = model.eval(bvVariable)

        assertEquals(bv, bvValue)
        assertEquals(aValue, trueExpr)
        assertEquals(cValue, c)
        assertTrue(eValue.sort is KArraySort<*, *>)
        val bInterp = model.interpretation(b)
        assertNotNull(bInterp)
        val detachedModel = model.detach()
        solver.close()
        assertEquals(aValue, detachedModel.eval(a))
        assertEquals(cValue, detachedModel.eval(c))
        assertEquals(eValue, detachedModel.eval(e))
    }
}
