package io.ksmt.expr

import io.ksmt.KContext
import io.ksmt.cache.hash
import io.ksmt.cache.structurallyEqual
import io.ksmt.decl.KDecl
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KFpRoundingModeSort

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

    override fun internHashCode(): Int = hash(value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { value }
}
