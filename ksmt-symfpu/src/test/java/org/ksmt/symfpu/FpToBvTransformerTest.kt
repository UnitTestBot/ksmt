package org.ksmt.symfpu

import org.junit.jupiter.api.Test
import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KFpEqualExpr
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KFp32Sort
import org.ksmt.utils.getValue
import kotlin.test.assertEquals
import kotlin.time.Duration.Companion.seconds

class FpToBvTransformerTest {

    @Test
    fun testFpToBvEqExpr(): Unit = with(KContext()) {
        val transformer = FpToBvTransformer(this)

        KZ3Solver(this).use { solver ->
            val (a, b) = createTwoFpVariables()
            checkTransformer(transformer, mkFpEqualExpr(a, b), solver)
        }
    }

    private fun KContext.createTwoFpVariables(): Pair<KApp<KFp32Sort, *>, KApp<KFp32Sort, *>> {
        val a by mkFp32Sort()
        val b by mkFp32Sort()
        return Pair(a, b)
    }

    private fun KContext.checkTransformer(
        transformer: FpToBvTransformer,
        exprToTransform: KFpEqualExpr<KFp32Sort>,
        solver: KZ3Solver
    ) {
        val transformedExpr = transformer.apply(exprToTransform)
        solver.assert(transformedExpr neq exprToTransform)

        // check assertions satisfiability with timeout
        val status = solver.check(timeout = 3.seconds)
        assertEquals(KSolverStatus.UNSAT, status)
    }
}
