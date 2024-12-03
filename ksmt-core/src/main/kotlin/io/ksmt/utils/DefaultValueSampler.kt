package io.ksmt.utils

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.sort.*

open class DefaultValueSampler(val ctx: KContext) : KSortVisitor<KExpr<*>> {
    override fun visit(sort: KBoolSort): KExpr<*> =
        ctx.boolSortDefaultValue()

    override fun visit(sort: KIntSort): KExpr<*> =
        ctx.intSortDefaultValue()

    override fun visit(sort: KRealSort): KExpr<*> =
        ctx.realSortDefaultValue()

    override fun visit(sort: KStringSort): KExpr<*> {
        TODO("Not yet implemented")
    }

    override fun visit(sort: KRegexSort): KExpr<*> {
        TODO("Not yet implemented")
    }

    override fun <S : KBvSort> visit(sort: S): KExpr<*> =
        ctx.bvSortDefaultValue(sort)

    override fun <S : KFpSort> visit(sort: S): KExpr<*> =
        ctx.fpSortDefaultValue(sort)

    override fun visit(sort: KFpRoundingModeSort): KExpr<*> =
        ctx.fpRoundingModeSortDefaultValue()

    override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): KExpr<*> =
        ctx.arraySortDefaultValue(sort)

    override fun <D0 : KSort, D1 : KSort, R : KSort> visit(sort: KArray2Sort<D0, D1, R>): KExpr<*> =
        ctx.arraySortDefaultValue(sort)

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(sort: KArray3Sort<D0, D1, D2, R>): KExpr<*> =
        ctx.arraySortDefaultValue(sort)

    override fun <R : KSort> visit(sort: KArrayNSort<R>): KExpr<*> =
        ctx.arraySortDefaultValue(sort)

    override fun visit(sort: KUninterpretedSort): KExpr<*> =
        ctx.uninterpretedSortDefaultValue(sort)
}
