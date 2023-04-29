package io.ksmt.decl

import io.ksmt.KContext
import io.ksmt.expr.KApp
import io.ksmt.expr.KExpr
import io.ksmt.sort.KUninterpretedSort

class KUninterpretedSortValueDecl internal constructor(
    ctx: KContext,
    sort: KUninterpretedSort,
    val valueIdx: Int
) : KConstDecl<KUninterpretedSort>(ctx, "${sort.name}!val!${valueIdx}", sort) {
    override fun apply(args: List<KExpr<*>>): KApp<KUninterpretedSort, *> =
        ctx.mkUninterpretedSortValue(sort, valueIdx)
}
