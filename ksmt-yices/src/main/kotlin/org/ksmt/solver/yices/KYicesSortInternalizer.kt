package org.ksmt.solver.yices

import org.ksmt.solver.KSolverUnsupportedFeatureException
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

open class KYicesSortInternalizer(
    private val yicesCtx: KYicesContext
) : KSortVisitor<YicesSort> {
    override fun visit(sort: KBoolSort): YicesSort = yicesCtx.internalizeSort(sort) {
        yicesCtx.bool
    }

    override fun visit(sort: KIntSort): YicesSort = yicesCtx.internalizeSort(sort) {
        yicesCtx.int
    }

    override fun visit(sort: KRealSort): YicesSort = yicesCtx.internalizeSort(sort) {
        yicesCtx.real
    }

    override fun <S : KBvSort> visit(sort: S): YicesSort = yicesCtx.internalizeSort(sort) {
        yicesCtx.bvType(sort.sizeBits)
    }

    override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): YicesSort =
        yicesCtx.internalizeSort(sort) {
            yicesCtx.functionType(sort.domain.internalizeYicesSort(), sort.range.internalizeYicesSort())
        }

    override fun visit(sort: KUninterpretedSort): YicesSort = yicesCtx.internalizeSort(sort) {
        yicesCtx.newUninterpretedType(sort.name)
    }

    override fun <S : KFpSort> visit(sort: S): YicesSort {
        throw KSolverUnsupportedFeatureException("Unsupported sort $sort")
    }

    override fun visit(sort: KFpRoundingModeSort): YicesSort {
        throw KSolverUnsupportedFeatureException("Unsupported sort $sort")
    }

    private fun KSort.internalizeYicesSort() = accept(this@KYicesSortInternalizer)
}
