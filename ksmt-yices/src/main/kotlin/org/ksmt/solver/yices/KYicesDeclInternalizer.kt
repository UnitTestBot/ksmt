package org.ksmt.solver.yices

import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDeclVisitor
import org.ksmt.decl.KFuncDecl
import org.ksmt.sort.KSort

open class KYicesDeclInternalizer (
    private val yicesCtx: KYicesContext,
    private val sortInternalizer: KYicesSortInternalizer
) : KDeclVisitor<YicesTerm> {
    override fun <S : KSort> visit(decl: KFuncDecl<S>): YicesTerm =
        yicesCtx.internalizeDecl(decl) {
            val argSorts = decl.argSorts.map { it.accept(sortInternalizer) }
            val sort = yicesCtx.functionType(argSorts + decl.sort.accept(sortInternalizer))

            yicesCtx.newUninterpretedTerm(decl.name, sort)
        }

    override fun <S : KSort> visit(decl: KConstDecl<S>): YicesTerm =
        yicesCtx.internalizeDecl(decl) {
            yicesCtx.newUninterpretedTerm(decl.name, decl.sort.accept(sortInternalizer))
        }
}
