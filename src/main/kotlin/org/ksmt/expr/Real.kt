package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort

class KToIntRealExpr internal constructor(
    val arg: KExpr<KRealSort>
) : KApp<KIntSort, KExpr<KRealSort>>() {
    override fun KContext.sort() = mkIntSort()
    override fun KContext.decl() = mkRealToIntDecl()
    override val args: List<KExpr<KRealSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformer): KExpr<KIntSort> = transformer.transform(this)
}

class KIsIntRealExpr internal constructor(
    val arg: KExpr<KRealSort>
) : KApp<KBoolSort, KExpr<KRealSort>>() {
    override fun KContext.sort() = mkBoolSort()
    override fun KContext.decl() = mkRealIsIntDecl()
    override val args: List<KExpr<KRealSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KRealNumExpr internal constructor(
    val numerator: KIntNumExpr,
    val denominator: KIntNumExpr
) : KApp<KRealSort, KExpr<*>>() {
    override fun KContext.sort() = mkRealSort()
    override fun KContext.decl() = mkRealNumDecl("$numerator/$denominator")
    override val args = emptyList<KExpr<*>>()
    override fun accept(transformer: KTransformer): KExpr<KRealSort> = transformer.transform(this)
}
