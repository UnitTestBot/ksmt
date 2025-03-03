import io.ksmt.KContext
import io.ksmt.utils.getValue

fun main() {
    val ctx = KContext(simplificationMode = KContext.SimplificationMode.SIMPLIFY)
    with(ctx) {
        println(mkStringFromInt( 2147483643337.expr))
//        val a by stringSort
//        val b by stringSort
//        println((a + "k".expr) + ("ok".expr + b))
//        println("aaaa".expr.len)
    }
}