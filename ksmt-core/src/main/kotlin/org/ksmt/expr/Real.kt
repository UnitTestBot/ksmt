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
    override fun sort(): KIntSort = ctx.mkIntSort()

    override fun decl(): KRealToIntDecl = ctx.mkRealToIntDecl()

    override val args: List<KExpr<KRealSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KIntSort> = transformer.transform(this)
}

class KIsIntRealExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KRealSort>
) : KApp<KBoolSort, KExpr<KRealSort>>(ctx) {
    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun decl(): KRealIsIntDecl = ctx.mkRealIsIntDecl()

    override val args: List<KExpr<KRealSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)
}

class KRealNumExpr internal constructor(
    ctx: KContext,
    val numerator: KIntNumExpr,
    val denominator: KIntNumExpr
) : KApp<KRealSort, KExpr<*>>(ctx) {
    override fun sort(): KRealSort = ctx.mkRealSort()

    override fun decl(): KRealNumDecl = ctx.mkRealNumDecl("$numerator/$denominator")

    override val args = emptyList<KExpr<*>>()

    override fun accept(transformer: KTransformerBase): KExpr<KRealSort> = transformer.transform(this)
}
