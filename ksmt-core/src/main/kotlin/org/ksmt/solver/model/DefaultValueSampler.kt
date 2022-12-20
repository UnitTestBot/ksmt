package org.ksmt.solver.model

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KSortVisitor
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.utils.asExpr

class DefaultValueSampler<T : KSort> private constructor(
    val ctx: KContext,
    val sort: T
) : KSortVisitor<KExpr<T>> {
    override fun visit(sort: KBoolSort): KExpr<T> = with(ctx) {
        trueExpr.asExpr(this@DefaultValueSampler.sort)
    }

    override fun visit(sort: KIntSort): KExpr<T> = with(ctx) {
        0.expr.asExpr(this@DefaultValueSampler.sort)
    }

    override fun visit(sort: KRealSort): KExpr<T> = with(ctx) {
        mkRealNum(0).asExpr(this@DefaultValueSampler.sort)
    }

    override fun <S : KBvSort> visit(sort: S): KExpr<T> = with(ctx) {
        mkBv("0", sort.sizeBits).asExpr(this@DefaultValueSampler.sort)
    }

    override fun <S : KFpSort> visit(sort: S): KExpr<T> = with(ctx) {
        mkFp(0f, sort).asExpr(this@DefaultValueSampler.sort)
    }

    override fun visit(sort: KFpRoundingModeSort): KExpr<T> = with(ctx) {
        val defaultRm = KFpRoundingMode.RoundNearestTiesToEven
        mkFpRoundingModeExpr(defaultRm).asExpr(this@DefaultValueSampler.sort)
    }

    override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): KExpr<T> = with(ctx) {
        mkArrayConst(sort, sort.range.sampleValue()).asExpr(this@DefaultValueSampler.sort)
    }

    override fun visit(sort: KUninterpretedSort): KExpr<T> =
        ctx.uninterpretedSortDefaultValue(sort).asExpr(this@DefaultValueSampler.sort)

    companion object{
        fun <T : KSort> T.sampleValue(): KExpr<T> =
            accept(DefaultValueSampler(ctx, this))
    }
}
