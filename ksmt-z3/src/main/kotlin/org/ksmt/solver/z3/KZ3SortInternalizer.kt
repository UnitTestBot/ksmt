package org.ksmt.solver.z3

import com.microsoft.z3.Native
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
) : KSortVisitor<Long> {
    private val nCtx: Long = z3Ctx.nCtx

    override fun visit(sort: KBoolSort): Long = z3Ctx.internalizeSort(sort) {
        Native.mkBoolSort(nCtx)
    }

    override fun visit(sort: KIntSort): Long = z3Ctx.internalizeSort(sort) {
        Native.mkIntSort(nCtx)
    }

    override fun visit(sort: KRealSort): Long = z3Ctx.internalizeSort(sort) {
        Native.mkRealSort(nCtx)
    }

    override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): Long =
        z3Ctx.internalizeSort(sort) {
            val domain = sort.domain.internalizeZ3Sort()
            val range = sort.range.internalizeZ3Sort()
            Native.mkArraySort(nCtx, domain, range)
        }

    override fun visit(sort: KFpRoundingModeSort): Long = z3Ctx.internalizeSort(sort) {
        Native.mkFpaRoundingModeSort(nCtx)
    }

    override fun <T : KBvSort> visit(sort: T): Long = z3Ctx.internalizeSort(sort) {
        val size = sort.sizeBits.toInt()
        Native.mkBvSort(nCtx, size)
    }

    override fun <S : KFpSort> visit(sort: S): Long = z3Ctx.internalizeSort(sort) {
        when (sort) {
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

    override fun visit(sort: KUninterpretedSort): Long = z3Ctx.internalizeSort(sort) {
        val sortName = Native.mkStringSymbol(nCtx, sort.name)
        Native.mkUninterpretedSort(nCtx, sortName)
    }

    @Suppress("MemberVisibilityCanBePrivate")
    fun <T : KSort> T.internalizeZ3Sort() = accept(this@KZ3SortInternalizer)
}
