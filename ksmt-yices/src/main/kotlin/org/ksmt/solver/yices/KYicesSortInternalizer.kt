package org.ksmt.solver.yices

import com.sri.yices.Types
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
        Types.BOOL
    }

    override fun visit(sort: KIntSort): YicesSort = yicesCtx.internalizeSort(sort) {
        Types.INT
    }

    override fun visit(sort: KRealSort): YicesSort = yicesCtx.internalizeSort(sort) {
        Types.REAL
    }

    override fun <S : KBvSort> visit(sort: S): YicesSort = yicesCtx.internalizeSort(sort) {
        when (sort.sizeBits) {
            8u -> Types.BV8
            16u -> Types.BV16
            32u -> Types.BV32
            64u -> Types.BV64
            else -> Types.bvType(sort.sizeBits.toInt())
        }
    }

    override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): YicesSort =
        yicesCtx.internalizeSort(sort) {
            Types.functionType(sort.domain.internalizeYicesSort(), sort.range.internalizeYicesSort())
        }

    override fun visit(sort: KUninterpretedSort): YicesSort = yicesCtx.internalizeSort(sort) {
        Types.newUninterpretedType(sort.name)
    }

    override fun <S : KFpSort> visit(sort: S): YicesSort {
        throw KSolverUnsupportedFeatureException("Unsupported sort $sort")
    }

    override fun visit(sort: KFpRoundingModeSort): YicesSort {
        throw KSolverUnsupportedFeatureException("Unsupported sort $sort")
    }

    private fun <T : KSort> T.internalizeYicesSort() = accept(this@KYicesSortInternalizer)
}
