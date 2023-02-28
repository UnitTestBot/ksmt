package org.ksmt.solver.model

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
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

open class DefaultValueSampler<T : KSort> (
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
        mkBv(0, sort.sizeBits).asExpr(this@DefaultValueSampler.sort)
    }

    override fun <S : KFpSort> visit(sort: S): KExpr<T> = with(ctx) {
        mkFp(0f, sort).asExpr(this@DefaultValueSampler.sort)
    }

    override fun visit(sort: KFpRoundingModeSort): KExpr<T> = with(ctx) {
        val defaultRm = KFpRoundingMode.RoundNearestTiesToEven
        mkFpRoundingModeExpr(defaultRm).asExpr(this@DefaultValueSampler.sort)
    }

    private fun <T : KArraySortBase<R>, R: KSort> sampleArrayValue(sort: T): KExpr<T> =
        ctx.mkArrayConst(sort, sort.range.sampleValue())

    override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): KExpr<T> =
        sampleArrayValue(sort).asExpr(this@DefaultValueSampler.sort)

    override fun <D0 : KSort, D1 : KSort, R : KSort> visit(sort: KArray2Sort<D0, D1, R>): KExpr<T> =
        sampleArrayValue(sort).asExpr(this@DefaultValueSampler.sort)

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(sort: KArray3Sort<D0, D1, D2, R>): KExpr<T> =
        sampleArrayValue(sort).asExpr(this@DefaultValueSampler.sort)

    override fun <R : KSort> visit(sort: KArrayNSort<R>): KExpr<T> =
        sampleArrayValue(sort).asExpr(this@DefaultValueSampler.sort)

    override fun visit(sort: KUninterpretedSort): KExpr<T> =
        ctx.uninterpretedSortDefaultValue(sort).asExpr(this@DefaultValueSampler.sort)

    companion object{
        fun <T : KSort> T.sampleValue(): KExpr<T> =
            accept(DefaultValueSampler(ctx, this))
    }
}
