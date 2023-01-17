package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.cache.hash
import org.ksmt.cache.structurallyEqual
import org.ksmt.decl.KRealIsIntDecl
import org.ksmt.decl.KRealNumDecl
import org.ksmt.decl.KRealToIntDecl
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort

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

    override fun customHashCode(): Int = hash(arg)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { arg })
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

    override fun customHashCode(): Int = hash(arg)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { arg })
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

    override fun customHashCode(): Int = hash(numerator, denominator)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { numerator }, { denominator })
}
