package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.sort.KBoolSort

abstract class KQuantifier(
    ctx: KContext,
    val body: KExpr<KBoolSort>,
    val bounds: List<KDecl<*>>
) : KExpr<KBoolSort>(ctx) {
    override fun sort(): KBoolSort = ctx.mkBoolSort()
}
