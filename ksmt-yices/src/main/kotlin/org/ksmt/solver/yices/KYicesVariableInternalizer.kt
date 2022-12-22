package org.ksmt.solver.yices

import com.sri.yices.Terms
import com.sri.yices.Types
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
            val variableType = Types.functionType(argSorts + decl.sort.accept(sortInternalizer))

            Terms.newVariable(decl.name, variableType)
        }

    override fun <S : KSort> visit(decl: KConstDecl<S>): YicesTerm =
        yicesCtx.internalizeDecl(decl) {
            Terms.newVariable(decl.name, decl.sort.accept(sortInternalizer))
        }
}
