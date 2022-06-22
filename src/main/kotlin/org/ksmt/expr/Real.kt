package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort

class KToIntRealExpr internal constructor(
    val arg: KExpr<KRealSort>
) : KArithExpr<KIntSort, KExpr<KRealSort>>(listOf(arg)) {
    override fun KContext.sort() = mkIntSort()
    override fun KContext.decl() = mkRealToIntDecl()
    override fun accept(transformer: KTransformer): KExpr<KIntSort> = transformer.transform(this)
}

class KIsIntRealExpr internal constructor(
    val arg: KExpr<KRealSort>
) : KBoolExpr<KExpr<KRealSort>>(listOf(arg)) {
    override fun KContext.decl() = mkRealIsIntDecl()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KRealNumExpr internal constructor(
    val numerator: KIntNumExpr,
    val denominator: KIntNumExpr
) : KArithExpr<KRealSort, KExpr<*>>(emptyList()) {
    override fun KContext.sort() = mkRealSort()
    override fun KContext.decl() = mkRealNumDecl("$numerator/$denominator")
    override fun accept(transformer: KTransformer): KExpr<KRealSort> = transformer.transform(this)
}
