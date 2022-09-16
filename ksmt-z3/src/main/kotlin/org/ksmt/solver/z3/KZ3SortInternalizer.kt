package org.ksmt.solver.z3

import com.microsoft.z3.Context
import com.microsoft.z3.Sort
import org.ksmt.expr.KBVSize
import org.ksmt.sort.KSortVisitor
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KBVSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort


open class KZ3SortInternalizer(
    val z3Ctx: Context,
    val z3InternCtx: KZ3InternalizationContext
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

    override fun <S : KBVSize> visit(sort: KBVSort<S>): Sort {
        TODO("Not yet implemented")
    }

    fun <T : KSort> T.internalizeZ3Sort() = accept(this@KZ3SortInternalizer)
}
