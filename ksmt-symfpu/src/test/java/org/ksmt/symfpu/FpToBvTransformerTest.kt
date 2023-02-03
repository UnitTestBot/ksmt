package org.ksmt.symfpu

import org.junit.jupiter.api.Test
import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KSort
import org.ksmt.utils.cast
import org.ksmt.utils.getValue
import kotlin.test.assertEquals
import kotlin.time.Duration.Companion.seconds

typealias Fp = KFp32Sort

class FpToBvTransformerTest {
    private fun KContext.createTwoFpVariables(): Pair<KApp<Fp, *>, KApp<Fp, *>> {
        val a by mkFp32Sort()
        val b by mkFp32Sort()
        return Pair(a, b)
    }

    private fun KContext.zero() = mkFpZero(false, Fp(this))
    private fun KContext.negativeZero() = mkFpZero(true, Fp(this))
    private inline fun <R> withContextAndVariables(block: KContext.(KApp<Fp, *>, KApp<Fp, *>) -> R): R =
        with(KContext()) {
            val (a, b) = createTwoFpVariables()
            block(a, b)
        }


    @Test
    fun testFpToBvEqExpr() = withContextAndVariables { a, b ->
        testFpExpr(mkFpEqualExpr(a, b))
    }

    @Test
    fun testFpToBvLessExpr() = withContextAndVariables { a, b ->
        testFpExpr(mkFpLessExpr(a, b))
    }

    @Test
    fun testFpToBvLessOrEqualExpr() = withContextAndVariables { a, b ->
        testFpExpr(mkFpLessOrEqualExpr(a, b))
    }

    @Test
    fun testFpToBvGreaterExpr() = withContextAndVariables { a, b ->
        testFpExpr(mkFpGreaterExpr(a, b))
    }

    @Test
    fun testFpToBvGreaterOrEqualExpr() = withContextAndVariables { a, b ->
        testFpExpr(mkFpGreaterOrEqualExpr(a, b))
    }

    // filter results for min(a,b) = ±0 as it is not a failure
    private fun KContext.assertionForZeroResults() = { transformedExpr: KExpr<Fp>, exprToTransform: KExpr<Fp> ->
        (transformedExpr eq zero() and (exprToTransform eq negativeZero())).not() and
                (transformedExpr eq negativeZero() and (exprToTransform eq zero())).not()
    }

    @Test
    fun testFpToBvMinExpr() = withContextAndVariables { a, b ->
        testFpExpr(mkFpMinExpr(a, b), assertionForZeroResults())
    }

    @Test
    fun testFpToBvMaxExpr() = withContextAndVariables { a, b ->
        testFpExpr(mkFpMaxExpr(a, b), assertionForZeroResults())
    }

    @Test
    fun testFpToBvMinRecursiveExpr() = withContextAndVariables { a, b ->
        val exprToTransform1 = mkFpMinExpr(a, mkFpMinExpr(a, b))
        val exprToTransform2 = mkFpMinExpr(a, exprToTransform1)

        testFpExpr(mkFpMinExpr(exprToTransform1, exprToTransform2), assertionForZeroResults())
    }


    private fun <T : KSort> KContext.testFpExpr(
        exprToTransform: KExpr<T>,
        extraAssert: ((KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>) = { _, _ -> trueExpr }
    ) {
        val transformer = FpToBvTransformer(this)

        KZ3Solver(this).use { solver ->
            checkTransformer(transformer, solver, exprToTransform, extraAssert)
        }
    }


    private fun <T : KSort> KContext.checkTransformer(
        transformer: FpToBvTransformer,
        solver: KZ3Solver,
        exprToTransform: KExpr<T>,
        extraAssert: ((KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>)
    ) {

        val applied = transformer.apply(exprToTransform)
        val transformedExpr: KExpr<T> = ((applied as? UnpackedFp<*>)?.toFp() ?: applied).cast()
        solver.assert(extraAssert(transformedExpr, exprToTransform))
        solver.assert(transformedExpr neq exprToTransform.cast())

        // check assertions satisfiability with timeout
        val status = solver.check(timeout = 3.seconds)

        assertEquals(KSolverStatus.UNSAT, status)
    }
}

