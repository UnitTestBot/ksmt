package org.ksmt.expr

import org.ksmt.KContext
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
) : KApp<KIntSort, KExpr<KRealSort>>(ctx) {
    override val sort: KIntSort
        get() = ctx.intSort

    override val decl: KRealToIntDecl
        get() = ctx.mkRealToIntDecl()

    override val args: List<KExpr<KRealSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KIntSort> = transformer.transform(this)
}

class KIsIntRealExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KRealSort>
) : KApp<KBoolSort, KExpr<KRealSort>>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override val decl: KRealIsIntDecl
        get() = ctx.mkRealIsIntDecl()

    override val args: List<KExpr<KRealSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)
}

class KRealNumExpr internal constructor(
    ctx: KContext,
    val numerator: KIntNumExpr,
    val denominator: KIntNumExpr
) : KApp<KRealSort, KExpr<*>>(ctx) {
    override val sort: KRealSort
        get() = ctx.realSort

    override val decl: KRealNumDecl
        get() = ctx.mkRealNumDecl("$numerator/$denominator")

    override val args = emptyList<KExpr<*>>()

    override fun accept(transformer: KTransformerBase): KExpr<KRealSort> = transformer.transform(this)
}
