package org.ksmt.solver.yices

import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.decl.KDeclVisitor
import org.ksmt.decl.KFuncDecl
import org.ksmt.solver.util.KExprIntInternalizerBase.Companion.NOT_INTERNALIZED
import org.ksmt.sort.KSort

open class KYicesDeclSortInternalizer(
    private val yicesCtx: KYicesContext,
    private val sortInternalizer: KYicesSortInternalizer
) : KDeclVisitor<Unit> {
    private var internalizedDeclSort: YicesSort = NOT_INTERNALIZED

    override fun <S : KSort> visit(decl: KFuncDecl<S>) {
        val argSorts = decl.argSorts.let { domain ->
            IntArray(domain.size) { sortInternalizer.internalizeYicesSort(domain[it]) }
        }
        val rangeSort = sortInternalizer.internalizeYicesSort(decl.sort)

        internalizedDeclSort = yicesCtx.functionType(argSorts, rangeSort)
    }

    override fun <S : KSort> visit(decl: KConstDecl<S>) {
        internalizedDeclSort = sortInternalizer.internalizeYicesSort(decl.sort)
    }

    fun internalizeYicesDeclSort(decl: KDecl<*>): YicesSort {
        decl.accept(this)
        return internalizedDeclSort
    }
}
