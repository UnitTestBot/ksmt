package io.ksmt.solver.z3

import io.ksmt.KContext
import io.ksmt.solver.KSolverStatus
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Z3 assigns the Real sort to `(^ a b)` even when both `a` and `b` are Int sorted,
 * while in ksmt such an expression is Int sorted.
 * Because of this, a model value of an Int sorted expression may be a Real numeral.
 * */
class ModelEvalSortTest {
    private val ctx = KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY)

    @Test
    fun testIntPowerModelValueSort() = with(ctx) {
        KZ3Solver(this).use { solver ->
            val base = mkConst("base", intSort)
            val power = mkConst("power", intSort)
            val expr = mkArithPowerNoSimplify(base, power)

            solver.assert(base eq 2.expr)
            solver.assert(power eq 2.expr)

            assertEquals(KSolverStatus.SAT, solver.check())

            val model = solver.model()
            val value = model.eval(expr)
            val detachedValue = model.detach().eval(expr)

            assertEquals(intSort, value.sort)
            assertEquals(4.expr, value)
            assertEquals(detachedValue, value)
        }
    }
}
