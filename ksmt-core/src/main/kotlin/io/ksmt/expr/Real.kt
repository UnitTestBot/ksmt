package io.ksmt.expr

import io.ksmt.KContext
import io.ksmt.cache.hash
import io.ksmt.cache.structurallyEqual
import io.ksmt.decl.KRealIsIntDecl
import io.ksmt.decl.KRealNumDecl
import io.ksmt.decl.KRealToIntDecl
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort

class KToIntRealExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KRealSort>
) : KApp<KIntSort, KRealSort>(ctx) {
    override val sort: KIntSort
        get() = ctx.intSort

    override val decl: KRealToIntDecl
        get() = ctx.mkRealToIntDecl()

    override val args: List<KExpr<KRealSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KIntSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KIsIntRealExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KRealSort>
) : KApp<KBoolSort, KRealSort>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override val decl: KRealIsIntDecl
        get() = ctx.mkRealIsIntDecl()

    override val args: List<KExpr<KRealSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KRealNumExpr internal constructor(
    ctx: KContext,
    val numerator: KIntNumExpr,
    val denominator: KIntNumExpr
) : KInterpretedValue<KRealSort>(ctx) {
    override val sort: KRealSort
        get() = ctx.realSort

    override val decl: KRealNumDecl
        get() = ctx.mkRealNumDecl("$numerator/$denominator")

    override fun accept(transformer: KTransformerBase): KExpr<KRealSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(numerator, denominator)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { numerator }, { denominator })
}
