package io.ksmt

import io.ksmt.KContext.SimplificationMode.NO_SIMPLIFY
import io.ksmt.expr.KAndExpr
import io.ksmt.expr.KApp
import io.ksmt.expr.KEqExpr
import io.ksmt.expr.KExpr
import io.ksmt.expr.transformer.KExprVisitResult
import io.ksmt.expr.transformer.KNonRecursiveVisitor
import io.ksmt.sort.KSort
import kotlin.test.Test
import kotlin.test.assertEquals

class ExpressionVisitorTest {

    @Test
    fun testExpressionVisitTrace(): Unit = with(KContext(simplificationMode = NO_SIMPLIFY)) {
        val x = mkConst("x", boolSort)
        val y = mkConst("y", boolSort)
        val e0 = mkAnd(y, y)
        val e1 = mkEq(x, x)
        val e2 = mkImplies(x, x)
        val expr = mkOr(e0, e1, e2)

        val trace = mutableListOf<VisitEvent>()
        val visitor = TracingVisitor(this, trace)

        val leafs = visitor.applyVisitor(expr)
        assertEquals(6, leafs.size)
        assertEquals(setOf(x, y), leafs.toSet())

        val expectedTrace = listOf(
            VisitEvent(VisitTraceKind.APP, expr),
            VisitEvent(VisitTraceKind.EXPR, expr),
            VisitEvent(VisitTraceKind.APP, e2),
            VisitEvent(VisitTraceKind.EXPR, e2),
            VisitEvent(VisitTraceKind.APP, x),
            VisitEvent(VisitTraceKind.EXPR, x),
            VisitEvent(VisitTraceKind.DEFAULT, x),
            VisitEvent(VisitTraceKind.APP, e2),
            VisitEvent(VisitTraceKind.EXPR, e2),
            VisitEvent(VisitTraceKind.EQ_ENTER, e1),
            VisitEvent(VisitTraceKind.EQ_VISIT, e1),
            VisitEvent(VisitTraceKind.APP, e0),
            VisitEvent(VisitTraceKind.APP, y),
            VisitEvent(VisitTraceKind.EXPR, y),
            VisitEvent(VisitTraceKind.DEFAULT, y),
            VisitEvent(VisitTraceKind.APP, expr),
            VisitEvent(VisitTraceKind.EXPR, expr),
        )

        assertEquals(expectedTrace, trace)
    }

    private class TracingVisitor(
        ctx: KContext,
        val trace: MutableList<VisitEvent>
    ) : KNonRecursiveVisitor<List<KExpr<*>>>(ctx) {

        override fun <T : KSort> defaultValue(expr: KExpr<T>): List<KExpr<*>> {
            trace += VisitEvent(VisitTraceKind.DEFAULT, expr)
            return listOf(expr)
        }

        override fun mergeResults(left: List<KExpr<*>>, right: List<KExpr<*>>): List<KExpr<*>> =
            left + right

        override fun <T : KSort> visitExpr(expr: KExpr<T>): KExprVisitResult<List<KExpr<*>>> {
            trace += VisitEvent(VisitTraceKind.EXPR, expr)
            return super.visitExpr(expr)
        }

        override fun <T : KSort, A : KSort> visitApp(expr: KApp<T, A>): KExprVisitResult<List<KExpr<*>>> {
            trace += VisitEvent(VisitTraceKind.APP, expr)
            return if (expr is KAndExpr) {
                val argsRecursive = expr.args.map {
                    applyVisitor(it)
                }
                val result = argsRecursive.reduce { res, arg -> res + arg }
                saveVisitResult(expr, result)
            } else {
                super.visitApp(expr)
            }
        }

        override fun <T : KSort> visit(expr: KEqExpr<T>): KExprVisitResult<List<KExpr<*>>> {
            trace += VisitEvent(VisitTraceKind.EQ_ENTER, expr)
            return visitExprAfterVisited(expr, expr.lhs, expr.rhs) { l, r ->
                trace += VisitEvent(VisitTraceKind.EQ_VISIT, expr)
                l + r
            }
        }
    }

    enum class VisitTraceKind {
        EXPR, APP, DEFAULT, EQ_ENTER, EQ_VISIT
    }

    data class VisitEvent(val kind: VisitTraceKind, val expr: KExpr<*>)
}
