package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KFpRoundNearestTiesToAwaySort
import org.ksmt.sort.KFpRoundNearestTiesToEvenSort
import org.ksmt.sort.KFpRoundTowardNegativeSort
import org.ksmt.sort.KFpRoundTowardPositiveSort
import org.ksmt.sort.KFpRoundTowardZeroSort
import org.ksmt.sort.KFpRoundingModeSort

sealed class KFpRoundingModeDecl<S : KFpRoundingModeSort>(
    ctx: KContext,
    name: String,
    sort: S
) : KConstDecl<S>(ctx, name, sort)

class KFpRoundNearestTiesToEvenDecl(ctx: KContext) :
    KFpRoundingModeDecl<KFpRoundNearestTiesToEvenSort>(
        ctx,
        "roundNearestTiesToEven",
        ctx.mkFpRoundNearestTiesToEvenSort()
    ) {
    override fun apply(args: List<KExpr<*>>): KApp<KFpRoundNearestTiesToEvenSort, *> =
        ctx.mkFpRoundNearestTiesToEvenExpr()

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFpRoundNearestTiesToAwayDecl(ctx: KContext) :
    KFpRoundingModeDecl<KFpRoundNearestTiesToAwaySort>(
        ctx,
        "roundNearestTiesToAway",
        ctx.mkFpRoundNearestTiesToAwaySort()
    ) {
    override fun apply(args: List<KExpr<*>>): KApp<KFpRoundNearestTiesToAwaySort, *> =
        ctx.mkFpRoundNearestTiesToAwayExpr()

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFpRoundTowardPositiveDecl(ctx: KContext) :
    KFpRoundingModeDecl<KFpRoundTowardPositiveSort>(ctx, "roundTowardPositive", ctx.mkFpRoundTowardPositiveSort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KFpRoundTowardPositiveSort, *> =
        ctx.mkFpRoundTowardPositiveExpr()

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFpRoundTowardNegativeDecl(ctx: KContext) :
    KFpRoundingModeDecl<KFpRoundTowardNegativeSort>(ctx, "roundTowardNegative", ctx.mkFpRoundTowardNegativeSort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KFpRoundTowardNegativeSort, *> =
        ctx.mkFpRoundTowardNegativeExpr()

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFpRoundTowardZeroDecl(ctx: KContext) :
    KFpRoundingModeDecl<KFpRoundTowardZeroSort>(ctx, "roundTowardZero", ctx.mkFpRoundTowardZeroSort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KFpRoundTowardZeroSort, *> =
        ctx.mkFpRoundTowardZeroExpr()

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
