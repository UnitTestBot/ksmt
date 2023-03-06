package org.ksmt.solver.z3

import com.microsoft.z3.Native
import org.ksmt.solver.util.KExprLongInternalizerBase.Companion.NOT_INTERNALIZED
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFp128Sort
import org.ksmt.sort.KFp16Sort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpCustomSizeSort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KSortVisitor
import org.ksmt.sort.KUninterpretedSort


open class KZ3SortInternalizer(
    private val z3Ctx: KZ3Context
) : KSortVisitor<Unit> {
    private var internalizedSort: Long = NOT_INTERNALIZED
    private val nCtx: Long = z3Ctx.nCtx

    override fun visit(sort: KBoolSort) {
        internalizedSort = Native.mkBoolSort(nCtx)
    }

    override fun visit(sort: KIntSort) {
        internalizedSort = Native.mkIntSort(nCtx)
    }

    override fun visit(sort: KRealSort) {
        internalizedSort = Native.mkRealSort(nCtx)
    }

    override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>) {
        val domain = internalizeZ3Sort(sort.domain)
        val range = internalizeZ3Sort(sort.range)
        internalizedSort = Native.mkArraySort(nCtx, domain, range)
    }

    override fun <D0 : KSort, D1 : KSort, R : KSort> visit(sort: KArray2Sort<D0, D1, R>) {
        val domain = longArrayOf(
            internalizeZ3Sort(sort.domain0),
            internalizeZ3Sort(sort.domain1)
        )
        val range = internalizeZ3Sort(sort.range)
        internalizedSort = Native.mkArraySortN(nCtx, domain.size, domain, range)
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(sort: KArray3Sort<D0, D1, D2, R>) {
        val domain = longArrayOf(
            internalizeZ3Sort(sort.domain0),
            internalizeZ3Sort(sort.domain1),
            internalizeZ3Sort(sort.domain2)
        )
        val range = internalizeZ3Sort(sort.range)
        internalizedSort = Native.mkArraySortN(nCtx, domain.size, domain, range)
    }

    override fun <R : KSort> visit(sort: KArrayNSort<R>) {
        val domain = sort.domainSorts.let { sorts ->
            LongArray(sorts.size) { internalizeZ3Sort(sorts[it]) }
        }
        val range = internalizeZ3Sort(sort.range)
        internalizedSort = Native.mkArraySortN(nCtx, domain.size, domain, range)
    }

    override fun visit(sort: KFpRoundingModeSort) {
        internalizedSort = Native.mkFpaRoundingModeSort(nCtx)
    }

    override fun <T : KBvSort> visit(sort: T) {
        val size = sort.sizeBits.toInt()
        internalizedSort = Native.mkBvSort(nCtx, size)
    }

    override fun <S : KFpSort> visit(sort: S) {
        internalizedSort = when (sort) {
            is KFp16Sort -> Native.mkFpaSort16(nCtx)
            is KFp32Sort -> Native.mkFpaSort32(nCtx)
            is KFp64Sort -> Native.mkFpaSort64(nCtx)
            is KFp128Sort -> Native.mkFpaSort128(nCtx)
            is KFpCustomSizeSort -> {
                val exp = sort.exponentBits.toInt()
                val significand = sort.significandBits.toInt()
                Native.mkFpaSort(nCtx, exp, significand)
            }

            else -> error("Unsupported sort: $sort")
        }
    }

    override fun visit(sort: KUninterpretedSort) {
        val sortName = Native.mkStringSymbol(nCtx, sort.name)
        internalizedSort = Native.mkUninterpretedSort(nCtx, sortName)
    }

    fun internalizeZ3Sort(sort: KSort): Long = z3Ctx.internalizeSort(sort) {
        sort.accept(this@KZ3SortInternalizer)
        internalizedSort
    }
}
