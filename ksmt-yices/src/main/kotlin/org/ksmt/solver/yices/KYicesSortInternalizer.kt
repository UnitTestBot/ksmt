package org.ksmt.solver.yices

import org.ksmt.solver.KSolverUnsupportedFeatureException
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

    override fun <D0 : KSort, D1 : KSort, R : KSort> visit(
        sort: KArray2Sort<D0, D1, R>
    ): YicesSort = yicesCtx.internalizeSort(sort) {
        val d0 = sort.domain0.internalizeYicesSort()
        val d1 = sort.domain1.internalizeYicesSort()
        val range = sort.range.internalizeYicesSort()

        yicesCtx.functionType(intArrayOf(d0, d1), range)
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(
        sort: KArray3Sort<D0, D1, D2, R>
    ): YicesSort = yicesCtx.internalizeSort(sort) {
        val d0 = sort.domain0.internalizeYicesSort()
        val d1 = sort.domain1.internalizeYicesSort()
        val d2 = sort.domain2.internalizeYicesSort()
        val range = sort.range.internalizeYicesSort()

        yicesCtx.functionType(intArrayOf(d0, d1, d2), range)
    }

    override fun <R : KSort> visit(sort: KArrayNSort<R>): YicesSort = yicesCtx.internalizeSort(sort) {
        val domain = sort.domainSorts.let { domain ->
            IntArray(domain.size) { domain[it].internalizeYicesSort() }
        }
        val range = sort.range.internalizeYicesSort()

        yicesCtx.functionType(domain, range)
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
