package org.ksmt.symfpu

import org.junit.jupiter.api.Test
import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KSort
import org.ksmt.utils.cast
import org.ksmt.utils.getValue
import kotlin.test.assertEquals
import kotlin.time.Duration.Companion.seconds

class FpToBvTransformerTest {

    @Test
    fun testFpToBvEqExpr() = with(KContext()) {
        val (a, b) = createTwoFpVariables()
        testFpExpr(mkFpEqualExpr(a, b))
    }

    @Test
    fun testFpToBvLessExpr() = with(KContext()) {
        val (a, b) = createTwoFpVariables()
        testFpExpr(mkFpLessExpr(a, b))
    }

    @Test
    fun testFpToBvMinExpr() = with(KContext()) {
        val transformer = FpToBvTransformer(this)

        KZ3Solver(this).use { solver ->
            val (a, b) = createTwoFpVariables()

            // both zero
            val zero = mkFpZero(false, KFp32Sort(this))
            val negativeZero = mkFpZero(true, KFp32Sort(this))
            val exprToTransform = mkFpMinExpr(a, b)

            val transformedExpr = (transformer.apply(exprToTransform) as UnpackedFp<KFp32Sort>).toFp()
            solver.assert((transformedExpr eq zero and (exprToTransform eq negativeZero)).not()) // min(-0, +0) = ±0
            solver.assert((transformedExpr eq negativeZero and (exprToTransform eq zero)).not())
            solver.assert(transformedExpr neq exprToTransform)


            // check assertions satisfiability with timeout
            val status = solver.check(timeout = 3.seconds)

            assertEquals(KSolverStatus.UNSAT, status)
        }
    }

    @Test
    fun testFpToBvRecursiveMinExpr() = with(KContext()) {
        val transformer = FpToBvTransformer(this)

        KZ3Solver(this).use { solver ->
            val (a, b) = createTwoFpVariables()

            // both zero
            val zero = mkFpZero(false, KFp32Sort(this))
            val negativeZero = mkFpZero(true, KFp32Sort(this))
            val exprToTransform1 = mkFpMinExpr(a, mkFpMinExpr(a, b))
            val exprToTransform2 = mkFpMinExpr(a, exprToTransform1)
            val exprToTransform = mkFpMinExpr(exprToTransform1, exprToTransform2)

            val transformedExpr = (transformer.apply(exprToTransform) as UnpackedFp<KFp32Sort>).toFp()

            solver.assert((transformedExpr eq zero and (exprToTransform eq negativeZero)).not()) // min(-0, +0) = ±0
            solver.assert((transformedExpr eq negativeZero and (exprToTransform eq zero)).not())
            solver.assert(transformedExpr neq exprToTransform)


            // check assertions satisfiability with timeout
            val status = solver.check(timeout = 3.seconds)

            assertEquals(KSolverStatus.UNSAT, status)
        }
    }

    private fun <T : KSort> KContext.testFpExpr(exprToTransform: KExpr<T>) {
        val transformer = FpToBvTransformer(this)

        KZ3Solver(this).use { solver ->
            checkTransformer(transformer, solver, exprToTransform)
        }
    }


    private fun KContext.createTwoFpVariables(): Pair<KApp<KFp32Sort, *>, KApp<KFp32Sort, *>> {
        val a by mkFp32Sort()
        val b by mkFp32Sort()
        return Pair(a, b)
    }

    private fun <T : KSort> KContext.checkTransformer(
        transformer: FpToBvTransformer,
        solver: KZ3Solver,
        exprToTransform: KExpr<T>
    ) {

        val transformedExpr = transformer.applyAndGetBvExpr(exprToTransform)
        solver.assert(transformedExpr neq exprToTransform.cast())

        // check assertions satisfiability with timeout
        val status = solver.check(timeout = 3.seconds)
        assertEquals(KSolverStatus.UNSAT, status)
    }
}

