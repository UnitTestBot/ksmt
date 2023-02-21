package org.ksmt.solver.yices

import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDeclVisitor
import org.ksmt.decl.KFuncDecl
import org.ksmt.sort.KSort

open class KYicesVariableInternalizer (
    private val yicesCtx: KYicesContext,
    private val sortInternalizer: KYicesSortInternalizer
) : KDeclVisitor<YicesTerm> {
    override fun <S : KSort> visit(decl: KFuncDecl<S>): YicesTerm =
        yicesCtx.internalizeDecl(decl) {
            val argSorts = decl.argSorts.map { it.accept(sortInternalizer) }
            val variableType = yicesCtx.functionType(argSorts + decl.sort.accept(sortInternalizer))

            yicesCtx.newVariable(decl.name, variableType)
        }

    override fun <S : KSort> visit(decl: KConstDecl<S>): YicesTerm =
        yicesCtx.internalizeDecl(decl) {
            yicesCtx.newVariable(decl.name, decl.sort.accept(sortInternalizer))
        }
}
