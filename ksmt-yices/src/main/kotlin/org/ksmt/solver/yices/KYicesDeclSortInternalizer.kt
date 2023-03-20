package org.ksmt.solver.yices

import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDeclVisitor
import org.ksmt.decl.KFuncDecl
import org.ksmt.sort.KSort

open class KYicesDeclSortInternalizer(
    private val yicesCtx: KYicesContext,
    private val sortInternalizer: KYicesSortInternalizer
) : KDeclVisitor<YicesSort> {
    override fun <S : KSort> visit(decl: KFuncDecl<S>): YicesSort {
        val argSorts = decl.argSorts.let { domain ->
            IntArray(domain.size) { domain[it].accept(sortInternalizer) }
        }
        val rangeSort = decl.sort.accept(sortInternalizer)

        return yicesCtx.functionType(argSorts, rangeSort)
    }

    override fun <S : KSort> visit(decl: KConstDecl<S>): YicesSort =
        decl.sort.accept(sortInternalizer)
}
