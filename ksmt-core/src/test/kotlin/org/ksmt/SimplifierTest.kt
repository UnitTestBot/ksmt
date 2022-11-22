package org.ksmt

import org.ksmt.expr.rewrite.simplify.KExprSimplifier
import org.ksmt.utils.getValue
import kotlin.test.Test
import kotlin.test.assertEquals

class SimplifierTest {

    @Test
    fun testNestedRewrite(): Unit = with(KContext()) {
        val a by boolSort
        val expr = mkEq(falseExpr, mkNot(a))
        val result = KExprSimplifier(this).apply(expr)
        assertEquals(a, result)

        val arraySort = mkArraySort(intSort, boolSort)
        val expr1 = mkArrayConst(arraySort, falseExpr) eq mkArrayConst(arraySort, mkNot(a))
        val result1 = KExprSimplifier(this).apply(expr1)
        assertEquals(a, result1)
    }

}
