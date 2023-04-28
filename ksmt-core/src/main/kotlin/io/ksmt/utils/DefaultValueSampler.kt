package io.ksmt.utils

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KSortVisitor
import io.ksmt.sort.KUninterpretedSort

open class DefaultValueSampler(val ctx: KContext) : KSortVisitor<KExpr<*>> {
    override fun visit(sort: KBoolSort): KExpr<*> =
        ctx.boolSortDefaultValue()

    override fun visit(sort: KIntSort): KExpr<*> =
        ctx.intSortDefaultValue()

    override fun visit(sort: KRealSort): KExpr<*> =
        ctx.realSortDefaultValue()

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
