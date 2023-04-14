package org.ksmt.utils

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArrayNSort
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
