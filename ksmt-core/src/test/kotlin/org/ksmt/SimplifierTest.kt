package org.ksmt

import org.ksmt.expr.KAndExpr
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

    @Test
    fun testArrayEqSimplification(): Unit = with(KContext()) {
        val array by mkArraySort(intSort, intSort)
        val specialIdx by intSort
        val indices1 = (0..3).map { it.expr } + listOf(specialIdx) + (7..10).map { it.expr }
        val indices2 = (0..1).map { it.expr } + listOf(specialIdx) + ((2..3) + (7..10)).map { it.expr }
        val array1 = indices1.fold(array) { a, idx -> a.store(idx, idx) }
        val array2 = indices2.fold(array) { a, idx -> a.store(idx, idx) }
        val arrayEq = array1 eq array2
        val allSelectsEq = mkAnd((indices1 + indices2).map {
            array1.select(it) eq array2.select(it)
        })

        val simplifiedEq = KExprSimplifier(this).apply(arrayEq)
        val simplifiedSelects = KExprSimplifier(this).apply(allSelectsEq)

        val simplifiedEqParts = (simplifiedEq as KAndExpr).args
        val simplifiedAllSelectsParts = (simplifiedSelects as KAndExpr).args
        assertEquals(simplifiedAllSelectsParts.toSet(), simplifiedEqParts.toSet())
    }

}
