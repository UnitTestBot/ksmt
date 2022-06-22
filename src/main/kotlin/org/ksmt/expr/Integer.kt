package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import java.math.BigInteger

class KModIntExpr internal constructor(
    val lhs: KExpr<KIntSort>,
    val rhs: KExpr<KIntSort>
) : KArithExpr<KIntSort, KExpr<KIntSort>>(listOf(lhs, rhs)) {
    override fun KContext.sort() = mkIntSort()
    override fun KContext.decl() = mkIntModDecl()
    override fun accept(transformer: KTransformer): KExpr<KIntSort> = transformer.transform(this)
}

class KRemIntExpr internal constructor(
    val lhs: KExpr<KIntSort>,
    val rhs: KExpr<KIntSort>
) : KArithExpr<KIntSort, KExpr<KIntSort>>(listOf(lhs, rhs)) {
    override fun KContext.sort() = mkIntSort()
    override fun KContext.decl() = mkIntRemDecl()
    override fun accept(transformer: KTransformer): KExpr<KIntSort> = transformer.transform(this)
}

class KToRealIntExpr internal constructor(
    val arg: KExpr<KIntSort>
) : KArithExpr<KRealSort, KExpr<KIntSort>>(listOf(arg)) {
    override fun KContext.sort() = mkRealSort()
    override fun KContext.decl() = mkIntToRealDecl()
    override fun accept(transformer: KTransformer): KExpr<KRealSort> = transformer.transform(this)
}

abstract class KIntNumExpr(
    private val value: Number
) : KArithExpr<KIntSort, KExpr<*>>(emptyList()) {
    override fun KContext.sort() = mkIntSort()
    override fun KContext.decl() = mkIntNumDecl("$value")
    override fun equalTo(other: KExpr<*>): Boolean {
        if (!super.equalTo(other)) return false
        other as KIntNumExpr
        return value == other.value
    }
}

class KInt32NumExpr internal constructor(val value: Int) : KIntNumExpr(value) {
    override fun accept(transformer: KTransformer): KExpr<KIntSort> = transformer.transform(this)
}

class KInt64NumExpr internal constructor(val value: Long) : KIntNumExpr(value) {
    override fun accept(transformer: KTransformer): KExpr<KIntSort> = transformer.transform(this)
}

class KIntBigNumExpr internal constructor(val value: BigInteger) : KIntNumExpr(value) {
    override fun accept(transformer: KTransformer): KExpr<KIntSort> = transformer.transform(this)
}
