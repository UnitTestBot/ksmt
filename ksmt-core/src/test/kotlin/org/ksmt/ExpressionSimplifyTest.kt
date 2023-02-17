package org.ksmt

import org.ksmt.expr.KExpr
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KSort
import kotlin.random.Random

open class ExpressionSimplifyTest {

    internal fun <S : KSort> runTest(mkSort: KContext.() -> S, test: KContext.(S, TestRunner) -> Unit) {
        val ctx = KContext(simplificationMode = KContext.SimplificationMode.SIMPLIFY)
        val sort: S = ctx.mkSort()
        val checker = TestRunner(ctx)
        ctx.test(sort, checker)
    }

    internal class TestRunner(private val ctx: KContext) {
        fun <T : KSort> check(
            unsimplifiedExpr: KExpr<T>,
            simplifiedExpr: KExpr<T>,
            printArgs: () -> String
        ) = KZ3Solver(ctx).use { solver ->
            val equivalenceCheck = ctx.mkEq(simplifiedExpr, unsimplifiedExpr)
            solver.assert(ctx.mkNot(equivalenceCheck))

            val status = solver.check()
            kotlin.test.assertEquals(KSolverStatus.UNSAT, status, printArgs())
        }
    }

    companion object {
        val random = Random(42)
    }
}
