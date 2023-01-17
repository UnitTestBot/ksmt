package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.cache.hash
import org.ksmt.cache.structurallyEqual
import org.ksmt.decl.KDecl
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KFpRoundingModeSort

enum class KFpRoundingMode(val modeName: String) {
    RoundNearestTiesToEven("RoundNearestTiesToEven"),
    RoundNearestTiesToAway("RoundNearestTiesToAway"),
    RoundTowardPositive("RoundTowardPositive"),
    RoundTowardNegative("RoundTowardNegative"),
    RoundTowardZero("RoundTowardZero")
}

class KFpRoundingModeExpr(
    ctx: KContext,
    val value: KFpRoundingMode
) : KInterpretedValue<KFpRoundingModeSort>(ctx) {
    override val decl: KDecl<KFpRoundingModeSort>
        get() = ctx.mkFpRoundingModeDecl(value)

    override val sort: KFpRoundingModeSort
        get() = ctx.mkFpRoundingModeSort()

    override fun accept(transformer: KTransformerBase): KExpr<KFpRoundingModeSort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(value)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { value })
}
