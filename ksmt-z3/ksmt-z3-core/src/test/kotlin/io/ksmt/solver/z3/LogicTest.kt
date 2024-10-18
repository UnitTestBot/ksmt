package io.ksmt.solver.z3

import io.ksmt.KContext
import io.ksmt.solver.KTheory
import io.ksmt.solver.KTheory.Array
import io.ksmt.solver.KTheory.BV
import io.ksmt.solver.KTheory.LIA
import io.ksmt.solver.KTheory.LRA
import io.ksmt.solver.KTheory.NIA
import io.ksmt.solver.KTheory.NRA
import io.ksmt.solver.KTheory.UF
import kotlin.test.Test
import kotlin.test.assertNotNull

class LogicTest {
    private val ctx = KContext()
    private val solver = KZ3Solver(ctx)

    @Test
    fun testSetAllLogic() {
        solver.configure {
            optimizeForTheories(theories = null, quantifiersAllowed = false)
        }

        assertNotNull(solver.nativeSolver())
    }

    @Test
    fun testSetPropLogic() {
        solver.configure {
            optimizeForTheories(theories = emptySet(), quantifiersAllowed = false)
        }

        assertNotNull(solver.nativeSolver())
    }

    @Test
    fun testSetSpecificLogic() {
        solver.configure {
            optimizeForTheories(theories = setOf(BV, UF, Array), quantifiersAllowed = false)
        }

        assertNotNull(solver.nativeSolver())
    }

    @Test
    fun testSetSpecificLogic2() {
        solver.configure {
            optimizeForTheories(theories = setOf(LIA, LRA, NIA, NRA, UF, Array), quantifiersAllowed = false)
        }

        assertNotNull(solver.nativeSolver())
    }

    @Test
    fun testSetSupportedLogic() {
        solver.configure {
            optimizeForTheories(theories = KTheory.values().toSet(), quantifiersAllowed = false)
        }

        assertNotNull(solver.nativeSolver())
    }
}
