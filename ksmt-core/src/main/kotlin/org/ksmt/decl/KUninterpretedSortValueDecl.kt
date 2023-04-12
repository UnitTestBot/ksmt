package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KUninterpretedSort

class KUninterpretedSortValueDecl internal constructor(
    ctx: KContext,
    sort: KUninterpretedSort,
    val valueIdx: Int
) : KConstDecl<KUninterpretedSort>(ctx, "${sort.name}!val!${valueIdx}", sort) {
    override fun apply(args: List<KExpr<*>>): KApp<KUninterpretedSort, *> =
        ctx.mkUninterpretedSortValue(sort, valueIdx)
}
