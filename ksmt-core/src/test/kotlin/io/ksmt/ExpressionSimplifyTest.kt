package io.ksmt

import io.ksmt.expr.KExpr
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.sort.KSort
import kotlin.random.Random

open class ExpressionSimplifyTest {

    internal fun <S : KSort> runTest(mkSort: KContext.() -> S, test: KContext.(S, TestRunner) -> Unit) {
        val ctx = KContext(simplificationMode = KContext.SimplificationMode.SIMPLIFY)
        val sort: S = ctx.mkSort()
        TestRunner(ctx).use { checker ->
            ctx.test(sort, checker)
        }
    }

    internal class TestRunner(private val ctx: KContext): AutoCloseable {
        private val solver = KZ3Solver(ctx)

        override fun close() {
            solver.close()
        }

        fun <T : KSort> check(
            unsimplifiedExpr: KExpr<T>,
            simplifiedExpr: KExpr<T>,
            printArgs: () -> String
        ) = solverScope {
            val equivalenceCheck = ctx.mkEq(simplifiedExpr, unsimplifiedExpr)
            solver.assert(ctx.mkNot(equivalenceCheck))

            val status = solver.check()
            kotlin.test.assertEquals(KSolverStatus.UNSAT, status, printArgs())
        }

        private inline fun solverScope(body: () -> Unit) = try {
            solver.push()
            body()
        } finally {
            solver.pop()
        }
    }

    companion object {
        val random = Random(42)
    }
}
