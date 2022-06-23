package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KIntModDecl
import org.ksmt.decl.KIntNumDecl
import org.ksmt.decl.KIntRemDecl
import org.ksmt.decl.KIntToRealDecl
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import java.math.BigInteger

class KModIntExpr internal constructor(
    val lhs: KExpr<KIntSort>,
    val rhs: KExpr<KIntSort>
) : KApp<KIntSort, KExpr<KIntSort>> {
    override fun KContext.sort(): KIntSort = mkIntSort()
    override fun KContext.decl(): KIntModDecl = mkIntModDecl()
    override val args: List<KExpr<KIntSort>>
        get() = listOf(lhs, rhs)
    override fun accept(transformer: KTransformer): KExpr<KIntSort> = transformer.transform(this)
}

class KRemIntExpr internal constructor(
    val lhs: KExpr<KIntSort>,
    val rhs: KExpr<KIntSort>
) : KApp<KIntSort, KExpr<KIntSort>> {
    override fun KContext.sort(): KIntSort = mkIntSort()
    override fun KContext.decl(): KIntRemDecl = mkIntRemDecl()
    override val args: List<KExpr<KIntSort>>
        get() = listOf(lhs, rhs)
    override fun accept(transformer: KTransformer): KExpr<KIntSort> = transformer.transform(this)
}

class KToRealIntExpr internal constructor(
    val arg: KExpr<KIntSort>
) : KApp<KRealSort, KExpr<KIntSort>> {
    override fun KContext.sort(): KRealSort = mkRealSort()
    override fun KContext.decl(): KIntToRealDecl = mkIntToRealDecl()
    override val args: List<KExpr<KIntSort>>
        get() = listOf(arg)
    override fun accept(transformer: KTransformer): KExpr<KRealSort> = transformer.transform(this)
}

abstract class KIntNumExpr(
    private val value: Number
) : KApp<KIntSort, KExpr<*>> {
    override fun KContext.sort(): KIntSort = mkIntSort()
    override fun KContext.decl(): KIntNumDecl = mkIntNumDecl("$value")
    override val args = emptyList<KExpr<*>>()
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
