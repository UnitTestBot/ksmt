package org.ksmt.solver.yices

import org.ksmt.solver.KSolverUnsupportedFeatureException
import org.ksmt.solver.util.KExprIntInternalizerBase.Companion.NOT_INTERNALIZED
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
) : KSortVisitor<Unit> {
    private var internalizedSort: YicesSort = NOT_INTERNALIZED

    override fun visit(sort: KBoolSort) {
        internalizedSort = yicesCtx.bool
    }

    override fun visit(sort: KIntSort) {
        internalizedSort = yicesCtx.int
    }

    override fun visit(sort: KRealSort) {
        internalizedSort = yicesCtx.real
    }

    override fun <S : KBvSort> visit(sort: S) {
        internalizedSort = yicesCtx.bvType(sort.sizeBits)
    }

    override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>) {
        val domain = internalizeYicesSort(sort.domain)
        val range = internalizeYicesSort(sort.range)

        internalizedSort = yicesCtx.functionType(domain, range)
    }

    override fun <D0 : KSort, D1 : KSort, R : KSort> visit(sort: KArray2Sort<D0, D1, R>) {
        val d0 = internalizeYicesSort(sort.domain0)
        val d1 = internalizeYicesSort(sort.domain1)
        val range = internalizeYicesSort(sort.range)

        internalizedSort = yicesCtx.functionType(intArrayOf(d0, d1), range)
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(sort: KArray3Sort<D0, D1, D2, R>) {
        val d0 = internalizeYicesSort(sort.domain0)
        val d1 = internalizeYicesSort(sort.domain1)
        val d2 = internalizeYicesSort(sort.domain2)
        val range = internalizeYicesSort(sort.range)

        internalizedSort = yicesCtx.functionType(intArrayOf(d0, d1, d2), range)
    }

    override fun <R : KSort> visit(sort: KArrayNSort<R>) {
        val domain = sort.domainSorts.let { domain ->
            IntArray(domain.size) { internalizeYicesSort(domain[it]) }
        }
        val range = internalizeYicesSort(sort.range)

        internalizedSort = yicesCtx.functionType(domain, range)
    }

    override fun visit(sort: KUninterpretedSort) {
        internalizedSort = yicesCtx.newUninterpretedType(sort.name)
    }

    override fun <S : KFpSort> visit(sort: S) {
        throw KSolverUnsupportedFeatureException("Unsupported sort $sort")
    }

    override fun visit(sort: KFpRoundingModeSort) {
        throw KSolverUnsupportedFeatureException("Unsupported sort $sort")
    }

    fun internalizeYicesSort(sort: KSort): YicesSort = yicesCtx.internalizeSort(sort) {
        sort.accept(this)
        internalizedSort
    }
}
