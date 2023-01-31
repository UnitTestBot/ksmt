package org.ksmt.symfpu

import org.junit.jupiter.api.Test
import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KFp32Sort
import org.ksmt.utils.getValue
import kotlin.test.assertEquals
import kotlin.time.Duration.Companion.seconds

class FpToBvTransformerTest {

    @Test
    fun testFpToBvEqExpr() = with(KContext()) {
        testFpExpr { a, b -> mkFpEqualExpr(a, b) }
    }

    @Test
    fun testFpToBvLessExpr() = with(KContext()) {
        testFpExpr { a, b -> mkFpLessExpr(a, b) }
    }

    private fun KContext.testFpExpr(exprMaker: ExprMaker) {
        val transformer = FpToBvTransformer(this)

        KZ3Solver(this).use { solver ->
            checkTransformer(transformer, solver, exprMaker)
        }
    }


    private fun KContext.createTwoFpVariables(): Pair<KApp<KFp32Sort, *>, KApp<KFp32Sort, *>> {
        val a by mkFp32Sort()
        val b by mkFp32Sort()
        return Pair(a, b)
    }

    private fun KContext.checkTransformer(
        transformer: FpToBvTransformer,
        solver: KZ3Solver,
        exprMaker: ExprMaker
    ) {
        val (a, b) = createTwoFpVariables()
        val exprToTransform = exprMaker(a, b)

        val transformedExpr = transformer.apply(exprToTransform)
        solver.assert(transformedExpr neq exprToTransform)

        // check assertions satisfiability with timeout
        val status = solver.check(timeout = 3.seconds)
        assertEquals(KSolverStatus.UNSAT, status)
    }
}

private typealias ExprMaker = (KApp<KFp32Sort, *>, KApp<KFp32Sort, *>) -> KApp<KBoolSort, KExpr<KFp32Sort>>
