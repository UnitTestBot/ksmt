package org.ksmt.solver.z3

import com.microsoft.z3.Native
import org.ksmt.decl.KDeclVisitor
import org.ksmt.decl.KFuncDecl
import org.ksmt.sort.KSort

open class KZ3DeclInternalizer(
    private val z3Ctx: KZ3Context,
    private val sortInternalizer: KZ3SortInternalizer
) : KDeclVisitor<Long> {
    override fun <S : KSort> visit(
        decl: KFuncDecl<S>
    ): Long = z3Ctx.internalizeDecl(decl) {
        val domainSorts = decl.argSorts.map { it.accept(sortInternalizer) }
        val rangeSort = decl.sort.accept(sortInternalizer)
        val nameSymbol = Native.mkStringSymbol(z3Ctx.nCtx, decl.name)
        Native.mkFuncDecl(
            z3Ctx.nCtx,
            nameSymbol,
            domainSorts.size,
            domainSorts.toLongArray(),
            rangeSort
        )
    }
}
