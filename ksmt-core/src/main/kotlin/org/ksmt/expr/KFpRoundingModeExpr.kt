package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.transformer.KTransformer
import org.ksmt.sort.KFpRoundNearestTiesToAwaySort
import org.ksmt.sort.KFpRoundNearestTiesToEvenSort
import org.ksmt.sort.KFpRoundTowardNegativeSort
import org.ksmt.sort.KFpRoundTowardPositiveSort
import org.ksmt.sort.KFpRoundTowardZeroSort
import org.ksmt.sort.KFpRoundingModeSort

sealed class KFpRoundingModeExpr<S : KFpRoundingModeSort>(ctx: KContext) : KApp<S, KExpr<*>>(ctx)

class KFpRoundNearestTiesToEvenExpr(ctx: KContext) :
    KFpRoundingModeExpr<KFpRoundNearestTiesToEvenSort>(ctx) {
    override val args: List<KExpr<*>>
        get() = emptyList()

    override fun decl(): KDecl<KFpRoundNearestTiesToEvenSort> = ctx.mkFpRoundNearestTiesToEvenDecl()

    override fun sort(): KFpRoundNearestTiesToEvenSort = ctx.mkFpRoundNearestTiesToEvenSort()

    override fun accept(transformer: KTransformer): KExpr<KFpRoundNearestTiesToEvenSort> = transformer.transform(this)
}

class KFpRoundNearestTiesToAwayExpr(ctx: KContext) :
    KFpRoundingModeExpr<KFpRoundNearestTiesToAwaySort>(ctx) {
    override val args: List<KExpr<*>>
        get() = emptyList()

    override fun decl(): KDecl<KFpRoundNearestTiesToAwaySort> = ctx.mkFpRoundNearestTiesToAwayDecl()

    override fun sort(): KFpRoundNearestTiesToAwaySort = ctx.mkFpRoundNearestTiesToAwaySort()

    override fun accept(transformer: KTransformer): KExpr<KFpRoundNearestTiesToAwaySort> = transformer.transform(this)
}

class KFpRoundTowardPositiveExpr(ctx: KContext) :
    KFpRoundingModeExpr<KFpRoundTowardPositiveSort>(ctx) {
    override val args: List<KExpr<*>>
        get() = emptyList()

    override fun decl(): KDecl<KFpRoundTowardPositiveSort> = ctx.mkFpRoundTowardPositiveDecl()

    override fun sort(): KFpRoundTowardPositiveSort = ctx.mkFpRoundTowardPositiveSort()

    override fun accept(transformer: KTransformer): KExpr<KFpRoundTowardPositiveSort> = transformer.transform(this)
}

class KFpRoundTowardNegativeExpr(ctx: KContext) :
    KFpRoundingModeExpr<KFpRoundTowardNegativeSort>(ctx) {
    override val args: List<KExpr<*>>
        get() = emptyList()

    override fun decl(): KDecl<KFpRoundTowardNegativeSort> = ctx.mkFpRoundTowardNegativeDecl()

    override fun sort(): KFpRoundTowardNegativeSort = ctx.mkFpRoundTowardNegativeSort()

    override fun accept(transformer: KTransformer): KExpr<KFpRoundTowardNegativeSort> = transformer.transform(this)
}

class KFpRoundTowardZeroExpr(ctx: KContext) :
    KFpRoundingModeExpr<KFpRoundTowardZeroSort>(ctx) {
    override val args: List<KExpr<*>>
        get() = emptyList()

    override fun decl(): KDecl<KFpRoundTowardZeroSort> = ctx.mkFpRoundTowardZeroDecl()

    override fun sort(): KFpRoundTowardZeroSort = ctx.mkFpRoundTowardZeroSort()

    override fun accept(transformer: KTransformer): KExpr<KFpRoundTowardZeroSort> = transformer.transform(this)
}
