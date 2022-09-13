package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.transformer.KTransformer
import org.ksmt.sort.KFpRoundingModeSort

enum class KFpRoundingMode(val modeName: String) {
    RoundNearestTiesToEven("RoundNearestTiesToEven"),
    RoundNearestTiesToAway("RoundNearestTiesToAway"),
    RoundTowardPositive("RoundTowardPositive"),
    RoundTowardNegative("RoundTowardNegative"),
    RoundTowardZero("RoundTowardZero")
}

class KFpRoundingModeExpr(ctx: KContext, val value: KFpRoundingMode): KApp<KFpRoundingModeSort, KExpr<*>>(ctx) {
    override val args: List<KExpr<*>>
        get() = emptyList()

    override fun decl(): KDecl<KFpRoundingModeSort> = ctx.mkFpRoundingModeDecl(value)

    override fun sort(): KFpRoundingModeSort = ctx.mkFpRoundingSortCache()

    override fun accept(transformer: KTransformer): KExpr<KFpRoundingModeSort> = transformer.transform(this)
}
