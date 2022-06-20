package org.ksmt.expr

import org.ksmt.decl.KIntModDecl
import org.ksmt.decl.KIntNumDecl
import org.ksmt.decl.KIntRemDecl
import org.ksmt.decl.KIntToRealDecl
import org.ksmt.expr.manager.ExprManager.intern
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import java.math.BigInteger

class KModIntExpr internal constructor(
    val lhs: KExpr<KIntSort>,
    val rhs: KExpr<KIntSort>
) : KArithExpr<KIntSort, KExpr<KIntSort>>(listOf(lhs, rhs)) {
    override val sort = KIntSort
    override val decl = KIntModDecl
    override fun accept(transformer: KTransformer): KExpr<KIntSort> {
        TODO("Not yet implemented")
    }
}

class KRemIntExpr internal constructor(
    val lhs: KExpr<KIntSort>,
    val rhs: KExpr<KIntSort>
) : KArithExpr<KIntSort, KExpr<KIntSort>>(listOf(lhs, rhs)) {
    override val sort = KIntSort
    override val decl = KIntRemDecl
    override fun accept(transformer: KTransformer): KExpr<KIntSort> {
        TODO("Not yet implemented")
    }
}

class KToRealIntExpr internal constructor(
    val arg: KExpr<KIntSort>
) : KArithExpr<KRealSort, KExpr<KIntSort>>(listOf(arg)) {
    override val sort = KRealSort
    override val decl = KIntToRealDecl
    override fun accept(transformer: KTransformer): KExpr<KRealSort> {
        TODO("Not yet implemented")
    }
}

abstract class KIntNumExpr(
    private val value: Number
) : KArithExpr<KIntSort, KExpr<*>>(emptyList()) {
    override val sort = KIntSort
    override val decl by lazy { KIntNumDecl("$value") }
    override fun equalTo(other: KExpr<*>): Boolean {
        if (!super.equalTo(other)) return false
        other as KIntNumExpr
        return value == other.value
    }
}

class KInt32NumExpr internal constructor(val value: Int) : KIntNumExpr(value) {
    override fun accept(transformer: KTransformer): KExpr<KIntSort> {
        TODO("Not yet implemented")
    }
}

class KInt64NumExpr internal constructor(val value: Long) : KIntNumExpr(value) {
    override fun accept(transformer: KTransformer): KExpr<KIntSort> {
        TODO("Not yet implemented")
    }
}

class KIntBigNumExpr internal constructor(val value: BigInteger) : KIntNumExpr(value) {
    override fun accept(transformer: KTransformer): KExpr<KIntSort> {
        TODO("Not yet implemented")
    }
}

fun mkIntMod(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>) = KModIntExpr(lhs, rhs).intern()
fun mkIntRem(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>) = KRemIntExpr(lhs, rhs).intern()
fun mkIntToReal(arg: KExpr<KIntSort>) = KToRealIntExpr(arg).intern()
fun mkIntNum(value: Int) = KInt32NumExpr(value).intern()
fun mkIntNum(value: Long) = KInt64NumExpr(value).intern()
fun mkIntNum(value: BigInteger) = KIntBigNumExpr(value).intern()
fun mkIntNum(value: String) =
    value.toIntOrNull()?.let { mkIntNum(it) }
        ?: value.toLongOrNull()?.let { mkIntNum(it) }
        ?: mkIntNum(value.toBigInteger())

infix fun KExpr<KIntSort>.mod(rhs: KExpr<KIntSort>) = mkIntMod(this, rhs)
infix fun KExpr<KIntSort>.rem(rhs: KExpr<KIntSort>) = mkIntRem(this, rhs)
fun KExpr<KIntSort>.toRealExpr() = mkIntToReal(this)
val Int.intExpr
    get() = mkIntNum(this)
val Long.intExpr
    get() = mkIntNum(this)
val BigInteger.intExpr
    get() = mkIntNum(this)

