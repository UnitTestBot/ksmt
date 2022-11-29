package org.ksmt.expr

import org.ksmt.KContext
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
) : KApp<KFpRoundingModeSort, KExpr<*>>(ctx) {
    override val args: List<KExpr<*>>
        get() = emptyList()

    override val decl: KDecl<KFpRoundingModeSort>
        get() = ctx.mkFpRoundingModeDecl(value)

    override val sort: KFpRoundingModeSort
        get() = ctx.mkFpRoundingModeSort()

    override fun accept(transformer: KTransformerBase): KExpr<KFpRoundingModeSort> = transformer.transform(this)
}
