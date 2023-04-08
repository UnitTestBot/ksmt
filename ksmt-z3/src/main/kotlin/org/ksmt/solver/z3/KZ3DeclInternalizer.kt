package org.ksmt.solver.z3

import com.microsoft.z3.Native
import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.decl.KDeclVisitor
import org.ksmt.decl.KFuncDecl
import org.ksmt.solver.util.KExprLongInternalizerBase.Companion.NOT_INTERNALIZED
import org.ksmt.sort.KSort

open class KZ3DeclInternalizer(
    private val z3Ctx: KZ3Context,
    private val sortInternalizer: KZ3SortInternalizer
) : KDeclVisitor<Unit> {
    private val nCtx = z3Ctx.nCtx
    private var lastInternalizedDecl: Long = NOT_INTERNALIZED

    override fun <S : KSort> visit(decl: KFuncDecl<S>) {
        val domainSorts = decl.argSorts.let { argSorts ->
            LongArray(argSorts.size) { sortInternalizer.internalizeZ3Sort(argSorts[it]) }
        }

        val rangeSort = sortInternalizer.internalizeZ3Sort(decl.sort)
        val nameSymbol = Native.mkStringSymbol(nCtx, decl.name)

        lastInternalizedDecl = Native.mkFuncDecl(
            nCtx,
            nameSymbol,
            domainSorts.size,
            domainSorts,
            rangeSort
        )
    }

    override fun <S : KSort> visit(decl: KConstDecl<S>) {
        val declSort = sortInternalizer.internalizeZ3Sort(decl.sort)
        val nameSymbol = Native.mkStringSymbol(nCtx, decl.name)

        lastInternalizedDecl = Native.mkFuncDecl(
            nCtx,
            nameSymbol,
            0,
            null,
            declSort
        )
    }

    fun internalizeZ3Decl(decl: KDecl<*>): Long = z3Ctx.internalizeDecl(decl) {
        decl.accept(this@KZ3DeclInternalizer)
        lastInternalizedDecl
    }
}
