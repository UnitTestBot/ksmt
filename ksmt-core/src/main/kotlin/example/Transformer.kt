package example

import org.ksmt.KContext
import org.ksmt.expr.KConst
import org.ksmt.expr.KExpr
import org.ksmt.expr.transformer.KTransformer
import org.ksmt.sort.KSort

class ConstCollector(override val ctx: KContext) : KTransformer {
    val constants = hashSetOf<KConst<*>>()
    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> {
        constants += expr
        return expr
    }
}

fun main() = with(KContext()) {
    val e1 = ((boolSort.mkConst("e1") and boolSort.mkConst("e2"))
            or (boolSort.mkConst("e2") and boolSort.mkConst("e1")))
    val e2 = (boolSort.mkConst("e1")
            and (boolSort.mkConst("e1") and boolSort.mkConst("e2"))
            or (boolSort.mkConst("e2") and boolSort.mkConst("e1")))
    val e = e1 and e2
    val constants = ConstCollector(this).apply { e.accept(this) }.constants
    println(constants.size)
}
