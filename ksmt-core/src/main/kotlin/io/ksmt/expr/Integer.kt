package io.ksmt.expr

import io.ksmt.KContext
import io.ksmt.cache.hash
import io.ksmt.cache.structurallyEqual
import io.ksmt.decl.KIntModDecl
import io.ksmt.decl.KIntNumDecl
import io.ksmt.decl.KIntRemDecl
import io.ksmt.decl.KIntToRealDecl
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import java.math.BigInteger

class KModIntExpr internal constructor(
    ctx: KContext,
    val lhs: KExpr<KIntSort>,
    val rhs: KExpr<KIntSort>
) : KApp<KIntSort, KIntSort>(ctx) {
    override val sort: KIntSort
        get() = ctx.intSort

    override val decl: KIntModDecl
        get() = ctx.mkIntModDecl()

    override val args: List<KExpr<KIntSort>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformerBase): KExpr<KIntSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(lhs, rhs)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { lhs }, { rhs })
}

class KRemIntExpr internal constructor(
    ctx: KContext,
    val lhs: KExpr<KIntSort>,
    val rhs: KExpr<KIntSort>
) : KApp<KIntSort, KIntSort>(ctx) {
    override val sort: KIntSort
        get() = ctx.intSort

    override val decl: KIntRemDecl
        get() = ctx.mkIntRemDecl()

    override val args: List<KExpr<KIntSort>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformerBase): KExpr<KIntSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(lhs, rhs)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { lhs }, { rhs })
}

class KToRealIntExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KIntSort>
) : KApp<KRealSort, KIntSort>(ctx) {
    override val sort: KRealSort
        get() = ctx.realSort

    override val decl: KIntToRealDecl
        get() = ctx.mkIntToRealDecl()

    override val args: List<KExpr<KIntSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KRealSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

abstract class KIntNumExpr(
    ctx: KContext,
    private val value: Number
) : KInterpretedValue<KIntSort>(ctx) {
    override val sort: KIntSort
        get() = ctx.intSort

    override val decl: KIntNumDecl
        get() = ctx.mkIntNumDecl("$value")

    override fun internHashCode(): Int = hash(value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { value }
}

class KInt32NumExpr internal constructor(
    ctx: KContext,
    val value: Int
) : KIntNumExpr(ctx, value) {
    override fun accept(transformer: KTransformerBase): KExpr<KIntSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { value }
}

class KInt64NumExpr internal constructor(
    ctx: KContext,
    val value: Long
) : KIntNumExpr(ctx, value) {
    override fun accept(transformer: KTransformerBase): KExpr<KIntSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { value }
}

class KIntBigNumExpr internal constructor(
    ctx: KContext,
    val value: BigInteger
) : KIntNumExpr(ctx, value) {
    override fun accept(transformer: KTransformerBase): KExpr<KIntSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { value }
}
