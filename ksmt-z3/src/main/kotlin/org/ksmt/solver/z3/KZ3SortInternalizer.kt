package org.ksmt.solver.z3

import com.microsoft.z3.Context
import com.microsoft.z3.Sort
import org.ksmt.sort.KSortVisitor
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KFp128Sort
import org.ksmt.sort.KFp16Sort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpCustomSizeSort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort


open class KZ3SortInternalizer(
    private val z3Ctx: Context,
    private val z3InternCtx: KZ3InternalizationContext
) : KSortVisitor<Sort> {
    override fun visit(sort: KBoolSort): Sort = z3InternCtx.internalizeSort(sort) {
        z3Ctx.boolSort
    }

    override fun visit(sort: KIntSort): Sort = z3InternCtx.internalizeSort(sort) {
        z3Ctx.intSort
    }

    override fun visit(sort: KRealSort): Sort = z3InternCtx.internalizeSort(sort) {
        z3Ctx.realSort
    }

    override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): Sort =
        z3InternCtx.internalizeSort(sort) {
            z3Ctx.mkArraySort(sort.domain.internalizeZ3Sort(), sort.range.internalizeZ3Sort())
        }

    override fun visit(sort: KFpRoundingModeSort): Sort = z3InternCtx.internalizeSort(sort) {
        z3Ctx.mkFPRoundingModeSort()
    }

    override fun <T : KBvSort> visit(sort: T): Sort = z3InternCtx.internalizeSort(sort) {
        z3Ctx.mkBitVecSort(sort.sizeBits.toInt())
    }

    override fun <S : KFpSort> visit(sort: S): Sort = z3InternCtx.internalizeSort(sort) {
        when (sort) {
            is KFp16Sort -> z3Ctx.mkFPSort16()
            is KFp32Sort -> z3Ctx.mkFPSort32()
            is KFp64Sort -> z3Ctx.mkFPSort64()
            is KFp128Sort -> z3Ctx.mkFPSort128()
            is KFpCustomSizeSort -> z3Ctx.mkFPSort(sort.exponentBits.toInt(), sort.significandBits.toInt())
            else -> error("Unsupported sort: $sort")
        }
    }

    override fun visit(sort: KUninterpretedSort): Sort = z3InternCtx.internalizeSort(sort) {
        z3Ctx.mkUninterpretedSort(sort.name)
    }

    @Suppress("MemberVisibilityCanBePrivate")
    fun <T : KSort> T.internalizeZ3Sort() = accept(this@KZ3SortInternalizer)
}
