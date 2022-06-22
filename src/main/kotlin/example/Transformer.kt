package example

import org.ksmt.KContext
import org.ksmt.expr.*
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort

class ConstCollector(override val ctx: KContext) : KTransformer {
    val constants = hashSetOf<KConst<*>>()
    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> {
        constants += expr
        return expr
    }
}

fun main() = with(KContext()) {
    val e1 = ((KBoolSort.mkConst("e1") and KBoolSort.mkConst("e2"))
            or (KBoolSort.mkConst("e2") and KBoolSort.mkConst("e1")))
    val e2 = (KBoolSort.mkConst("e1")
            and (KBoolSort.mkConst("e1") and KBoolSort.mkConst("e2"))
            or (KBoolSort.mkConst("e2") and KBoolSort.mkConst("e1")))
    val e = e1 and e2
    val constants = ConstCollector(this).apply { e.accept(this) }.constants
    println(constants.size)
}
