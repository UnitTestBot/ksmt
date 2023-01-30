package org.ksmt.symfpu

import org.junit.jupiter.api.Test
import org.ksmt.KContext
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.symfpu.Compare.Companion.fpToBvExpr
import org.ksmt.utils.getValue
import kotlin.test.assertEquals
import kotlin.time.Duration.Companion.seconds

class CompareTest {
    @Test
    fun testFpToBvEqExpr(): Unit = with(KContext()) {
        KZ3Solver(this).use { solver ->
            val a by mkFp32Sort()
            val b by mkFp32Sort()

            solver.assert(fpToBvExpr(mkFpEqualExpr(a, b)) neq mkFpEqualExpr(a, b))

            // check assertions satisfiability with timeout
            val status = solver.check(timeout = 3.seconds)
            assertEquals(KSolverStatus.UNSAT, status)
        }
    }
}
