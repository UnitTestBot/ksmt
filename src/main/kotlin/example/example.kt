package example

import org.ksmt.expr.*
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.mkConst

fun main() {
    val a = KBoolSort.mkConst("a")
    val b = KArithSort.mkConst("b")
    val c = KArraySort(KArithSort, KArithSort).mkConst("e")
    val x = KArraySort(KArithSort, KArraySort(KArithSort, KArithSort)).mkConst("e")
    val e1 = (c.select(b) + (c.store(b, b).select(b) + b) eq c.select(b)) and a or !a
    val e2 = x.store(b, c).select(b).select(10.expr) ge 11.expr
    val z = 3
}