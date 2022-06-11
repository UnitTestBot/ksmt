package example

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBoolSort

fun main() = with(KContext()) {
    val a = mkBoolSort().mkConst("a")
    val b = mkIntSort().mkConst("b")
    val c = mkArraySort(mkIntSort(), mkIntSort()).mkConst("e")
    val x = mkArraySort(mkIntSort(), mkArraySort(mkIntSort(), mkIntSort())).mkConst("e")
    val e1 = (c.select(b) + (c.store(b, b).select(b) + b) eq c.select(b)) and a or !a
    val e2 = x.store(b, c).select(b).select(10.intExpr) ge 11.intExpr
    val z = 3
}

fun anyexpr(expr: KExpr<*>) = with(KContext()) {
    expr as KExpr<KBoolSort>
    expr.and(expr)
}
