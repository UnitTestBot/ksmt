package io.ksmt.solver.bitwuzla

import io.ksmt.KContext
import io.ksmt.solver.KSolverStatus
import kotlin.test.Test
import kotlin.test.assertEquals

class UninterpretedSortArrayModelTest {
    private val ctx = KContext()
    private val solver = KBitwuzlaSolver(ctx)

    @Test
    fun detachModelOfSelectedTwoIndexArrayOverUninterpretedSort(): Unit = with(ctx) {
        val address = mkUninterpretedSort("Address")
        val array = mkConst("input", mkArraySort(address, bv32Sort, bv8Sort))
        val a = mkConst("a", address)
        val i = mkConst("i", bv32Sort)
        val selected = array.select(a, i)

        solver.assert(selected eq mkBv(5.toByte()))
        assertEquals(KSolverStatus.SAT, solver.check())

        val model = solver.model().detach()
        assertEquals(mkBv(5.toByte()), model.eval(selected))
    }

    @Test
    fun detachModelOfEqualTwoIndexArraysOverUninterpretedSort(): Unit = with(ctx) {
        val address = mkUninterpretedSort("Address")
        val sort = mkArraySort(address, bv32Sort, bv8Sort)
        val array = mkConst("input", sort)
        val other = mkConst("other", sort)

        solver.assert(array eq other)
        assertEquals(KSolverStatus.SAT, solver.check())

        val model = solver.model().detach()
        assertEquals(model.eval(array), model.eval(other))
    }

    @Test
    fun detachModelOfStoredTwoIndexArrayOverUninterpretedSort(): Unit = with(ctx) {
        val address = mkUninterpretedSort("Address")
        val sort = mkArraySort(address, bv32Sort, bv8Sort)
        val array = mkConst("input", sort)
        val other = mkConst("other", sort)
        val a = mkConst("a", address)
        val i = mkConst("i", bv32Sort)

        solver.assert(array eq other.store(a, i, mkBv(7.toByte())))
        assertEquals(KSolverStatus.SAT, solver.check())

        val model = solver.model().detach()
        assertEquals(mkBv(7.toByte()), model.eval(array.select(a, i)))
    }
}
