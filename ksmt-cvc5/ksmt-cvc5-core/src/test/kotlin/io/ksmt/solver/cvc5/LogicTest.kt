package io.ksmt.solver.cvc5

import io.ksmt.KContext
import io.ksmt.solver.KTheory
import io.ksmt.solver.smtLib2String
import kotlin.test.Test
import kotlin.test.assertEquals

class LogicTest {
    private val ctx = KContext()
    private val solver = KCvc5Solver(ctx)

    @Test
    fun testSetAllLogic() {
        solver.configure {
            optimizeForTheories(theories = null, quantifiersAllowed = false)
        }

        assertEquals("QF_ALL", solver.nativeSolver().getLogic())
    }

    @Test
    fun testSetPropLogic() {
        solver.configure {
            optimizeForTheories(theories = emptySet(), quantifiersAllowed = false)
        }

        assertEquals("QF_SAT", solver.nativeSolver().getLogic())
    }

    @Test
    fun testSetSpecificLogic() {
        solver.configure {
            optimizeForTheories(theories = setOf(KTheory.BV, KTheory.UF, KTheory.Array), quantifiersAllowed = false)
        }

        assertEquals("QF_AUFBV", solver.nativeSolver().getLogic())
    }

    @Test
    fun testSetSupportedLogic() {
        solver.configure {
            optimizeForTheories(theories = KTheory.values().toSet(), quantifiersAllowed = false)
        }

        val expected = KTheory.values().toSet().smtLib2String(quantifiersAllowed = false)
        assertEquals(expected, solver.nativeSolver().getLogic())
    }
}
