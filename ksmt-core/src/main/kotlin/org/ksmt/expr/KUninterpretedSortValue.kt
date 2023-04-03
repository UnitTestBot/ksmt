package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.cache.hash
import org.ksmt.cache.structurallyEqual
import org.ksmt.decl.KDecl
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KUninterpretedSort

class KUninterpretedSortValue internal constructor(
    ctx: KContext,
    override val sort: KUninterpretedSort,
    val valueIdx: Int
) : KInterpretedValue<KUninterpretedSort>(ctx) {
    override val decl: KDecl<KUninterpretedSort>
        get() = ctx.mkUninterpretedSortValueDecl(sort, valueIdx)

    override fun accept(transformer: KTransformerBase): KExpr<KUninterpretedSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(sort, valueIdx)

    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { sort }, { valueIdx })
}
