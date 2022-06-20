package org.ksmt.expr

import org.ksmt.decl.KRealIsIntDecl
import org.ksmt.decl.KRealNumDecl
import org.ksmt.decl.KRealToIntDecl
import org.ksmt.expr.manager.ExprManager.intern
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort

class KToIntRealExpr internal constructor(
    val arg: KExpr<KRealSort>
) : KArithExpr<KIntSort, KExpr<KRealSort>>(listOf(arg)) {
    override val sort = KIntSort
    override val decl = KRealToIntDecl
    override fun accept(transformer: KTransformer): KExpr<KIntSort> {
        TODO("Not yet implemented")
    }
}

class KIsIntRealExpr internal constructor(
    val arg: KExpr<KRealSort>
) : KBoolExpr<KExpr<KRealSort>>(listOf(arg)) {
    override val decl = KRealIsIntDecl
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }
}

class KRealNumExpr internal constructor(
    val numerator: KIntNumExpr,
    val denominator: KIntNumExpr
) : KArithExpr<KRealSort, KExpr<*>>(emptyList()) {
    override val sort = KRealSort
    override val decl by lazy { KRealNumDecl("$numerator/$denominator") }
    override fun accept(transformer: KTransformer): KExpr<KRealSort> {
        TODO("Not yet implemented")
    }
}

fun mkRealToInt(arg: KExpr<KRealSort>) = KToIntRealExpr(arg).intern()
fun mkRealIsInt(arg: KExpr<KRealSort>) = KIsIntRealExpr(arg).intern()
fun mkRealNum(numerator: KIntNumExpr, denominator: KIntNumExpr) = KRealNumExpr(numerator, denominator).intern()
fun mkRealNum(numerator: KIntNumExpr) = mkRealNum(numerator, 1.intExpr)
fun mkRealNum(numerator: Int) = mkRealNum(mkIntNum(numerator))
fun mkRealNum(numerator: Int, denominator: Int) = mkRealNum(mkIntNum(numerator), mkIntNum(denominator))
fun mkRealNum(numerator: Long) = mkRealNum(mkIntNum(numerator))
fun mkRealNum(numerator: Long, denominator: Long) = mkRealNum(mkIntNum(numerator), mkIntNum(denominator))
fun mkRealNum(value: String): KRealNumExpr {
    val parts = value.split('/')
    return when (parts.size) {
        1 -> mkRealNum(mkIntNum(parts[0]))
        2 -> mkRealNum(mkIntNum(parts[0]), mkIntNum(parts[1]))
        else -> error("incorrect real num format")
    }
}

fun KExpr<KRealSort>.toIntExpr() = mkRealToInt(this)
fun KExpr<KRealSort>.isIntExpr() = mkRealIsInt(this)
