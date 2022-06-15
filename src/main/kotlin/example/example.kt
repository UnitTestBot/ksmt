package example

import org.ksmt.decl.mkConst
import org.ksmt.expr.*
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KIntSort

fun main() {
    val a = KBoolSort.mkConst("a")
    val b = KIntSort.mkConst("b")
    val c = KArraySort(KIntSort, KIntSort).mkConst("e")
    val x = KArraySort(KIntSort, KArraySort(KIntSort, KIntSort)).mkConst("e")
    val e1 = (c.select(b) + (c.store(b, b).select(b) + b) eq c.select(b)) and a or !a
    val e2 = x.store(b, c).select(b).select(10.intExpr) ge 11.intExpr
    val z = 3
}

fun anyexpr(expr: KExpr<*>){
    expr as KExpr<KBoolSort>
    expr.and(expr)
}
