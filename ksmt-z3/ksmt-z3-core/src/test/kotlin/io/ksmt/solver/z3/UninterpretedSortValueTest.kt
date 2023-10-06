package io.ksmt.solver.z3

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KUninterpretedSort
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class UninterpretedSortValueTest {
    private val ctx = KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY)
    private val solver = KZ3Solver(ctx)

    @Test
    fun testUninterpretedSortValueDistinct() = with(ctx) {
        val sort = mkUninterpretedSort("T")
        val value0 = mkUninterpretedSortValue(sort, 0)
        val value1 = mkUninterpretedSortValue(sort, 1)

        solver.push()

        solver.assert(value0 eq value1)

        assertEquals(KSolverStatus.UNSAT, solver.check())

        solver.pop()

        // Internalization skipped
        solver.assert(value0 eq value1)

        assertEquals(KSolverStatus.UNSAT, solver.check())
    }

    @Test
    fun testUninterpretedSortValueModel() = with(ctx) {
        val sort = mkUninterpretedSort("T")
        val value0 = mkUninterpretedSortValue(sort, 0)
        val value1 = mkUninterpretedSortValue(sort, 1)

        val var0 = mkConst("v0", sort)
        val var1 = mkConst("v1", sort)
        val var2 = mkConst("v2", sort)

        solver.assert(var0 eq value1)
        solver.assert(var1 eq value0)

        solver.push()

        solver.assert(var0 eq var1)
        assertEquals(KSolverStatus.UNSAT, solver.check())

        solver.pop()

        solver.push()
        checkSatAndValidateModel(var0, var1, var2, value0, value1)
        solver.pop()

        checkSatAndValidateModel(var0, var1, var2, value0, value1)
    }

    private fun KContext.checkSatAndValidateModel(
        var0: KExpr<KUninterpretedSort>,
        var1: KExpr<KUninterpretedSort>,
        var2: KExpr<KUninterpretedSort>,
        value0: KUninterpretedSortValue,
        value1: KUninterpretedSortValue,
    ) {
        solver.assert(mkDistinct(listOf(var0, var1, var2)))
        assertEquals(KSolverStatus.SAT, solver.check())

        val model = solver.model()

        val modelValue0 = model.eval(var0, isComplete = false)
        val modelValue1 = model.eval(var1, isComplete = false)
        val modelValue2 = model.eval(var2, isComplete = false)

        assertEquals(value0, modelValue1)
        assertEquals(value1, modelValue0)

        assertTrue(modelValue2 is KUninterpretedSortValue)
        assertNotEquals(value0, modelValue2)
        assertNotEquals(value1, modelValue2)
    }
}
