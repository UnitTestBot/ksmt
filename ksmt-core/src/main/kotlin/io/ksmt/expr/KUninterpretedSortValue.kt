package io.ksmt.expr

import io.ksmt.KContext
import io.ksmt.cache.hash
import io.ksmt.cache.structurallyEqual
import io.ksmt.decl.KDecl
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KUninterpretedSort

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
