package io.ksmt

import io.ksmt.expr.KExpr
import io.ksmt.sort.KBv32Sort
import io.ksmt.utils.getValue
import kotlin.test.Test
import kotlin.test.assertEquals

class ExpressionSortTest {

    @Test
    fun testDeepExpressionSort() = with(KContext()) {
        val a by mkBv32Sort()
        val b by mkBv32Sort()

        var expr: KExpr<KBv32Sort> = a
        repeat(100000) {
            expr = mkBvAddExpr(expr, b)
        }

        assertEquals(mkBv32Sort(), expr.sort)
    }

}
