package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KIntModDecl
import org.ksmt.decl.KIntNumDecl
import org.ksmt.decl.KIntRemDecl
import org.ksmt.decl.KIntToRealDecl
import org.ksmt.expr.transformer.KTransformer
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import java.math.BigInteger

class KModIntExpr internal constructor(
    ctx: KContext,
    val lhs: KExpr<KIntSort>,
    val rhs: KExpr<KIntSort>
) : KApp<KIntSort, KExpr<KIntSort>>(ctx) {
    override fun sort(): KIntSort = ctx.mkIntSort()
    override fun decl(): KIntModDecl = ctx.mkIntModDecl()
    override val args: List<KExpr<KIntSort>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformer): KExpr<KIntSort> = transformer.transform(this)
}

class KRemIntExpr internal constructor(
    ctx: KContext,
    val lhs: KExpr<KIntSort>,
    val rhs: KExpr<KIntSort>
) : KApp<KIntSort, KExpr<KIntSort>>(ctx) {
    override fun sort(): KIntSort = ctx.mkIntSort()
    override fun decl(): KIntRemDecl = ctx.mkIntRemDecl()
    override val args: List<KExpr<KIntSort>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformer): KExpr<KIntSort> = transformer.transform(this)
}

class KToRealIntExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KIntSort>
) : KApp<KRealSort, KExpr<KIntSort>>(ctx) {
    override fun sort(): KRealSort = ctx.mkRealSort()
    override fun decl(): KIntToRealDecl = ctx.mkIntToRealDecl()
    override val args: List<KExpr<KIntSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformer): KExpr<KRealSort> = transformer.transform(this)
}

abstract class KIntNumExpr(
    ctx: KContext,
    private val value: Number
) : KApp<KIntSort, KExpr<*>>(ctx) {
    override fun sort(): KIntSort = ctx.mkIntSort()
    override fun decl(): KIntNumDecl = ctx.mkIntNumDecl("$value")
    override val args = emptyList<KExpr<*>>()
}

class KInt32NumExpr internal constructor(
    ctx: KContext,
    val value: Int
) : KIntNumExpr(ctx, value) {
    override fun accept(transformer: KTransformer): KExpr<KIntSort> = transformer.transform(this)
}

class KInt64NumExpr internal constructor(
    ctx: KContext,
    val value: Long
) : KIntNumExpr(ctx, value) {
    override fun accept(transformer: KTransformer): KExpr<KIntSort> = transformer.transform(this)
}

class KIntBigNumExpr internal constructor(
    ctx: KContext,
    val value: BigInteger
) : KIntNumExpr(ctx, value) {
    override fun accept(transformer: KTransformer): KExpr<KIntSort> = transformer.transform(this)
}
