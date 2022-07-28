package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.bitwuzla.KBitwuzlaSolver
import org.ksmt.sort.KArraySort
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class Example {

    @Test
    @Suppress("USELESS_IS_CHECK")
    fun test() = with(KContext()) {
        val a = boolSort.mkConst("a")
        val b = mkFuncDecl("b", boolSort, listOf(boolSort))
        val xFun = mkFuncDecl("xFun", boolSort, listOf(boolSort, boolSort, boolSort))
        val c = boolSort.mkConst("c")
        val e = mkArraySort(boolSort, boolSort).mkConst("e")
        val eConst = mkArraySort(boolSort, boolSort).mkConst("eConst")
        val solver = KBitwuzlaSolver(this)
        solver.assert(a)
        solver.assert(b.apply(listOf(trueExpr)) eq falseExpr)
        solver.assert(xFun.apply(listOf(trueExpr, trueExpr, trueExpr)) eq falseExpr)
        solver.assert(xFun.apply(listOf(trueExpr, falseExpr, trueExpr)) eq falseExpr)
        solver.assert(xFun.apply(listOf(falseExpr, falseExpr, trueExpr)) eq trueExpr)
        solver.assert(e.select(a) eq trueExpr)
        solver.assert(e.select(trueExpr) eq trueExpr)
        solver.assert(e.select(falseExpr) eq falseExpr)
        solver.assert(eConst eq mkArrayConst(mkArraySort(boolSort, boolSort), trueExpr))
        val status = solver.check()
        assertEquals(status, KSolverStatus.SAT)
        val model = solver.model()
        val aValue = model.eval(a)
        val cValue = model.eval(c)
        val eValue = model.eval(e)
        assertEquals(aValue, trueExpr)
        assertTrue(eValue.sort is KArraySort<*, *>)
        val bInterp = model.interpretation(b)
        model.interpretation(xFun)
        model.interpretation(e.decl)
        model.interpretation(a.decl)
        model.interpretation(eConst.decl)
        assertNotNull(bInterp)
        val detachedModel = model.detach()
        solver.close()
        assertFailsWith(IllegalStateException::class) { model.eval(a) }
        assertEquals(aValue, detachedModel.eval(a))
        assertEquals(cValue, detachedModel.eval(c))
    }
}
