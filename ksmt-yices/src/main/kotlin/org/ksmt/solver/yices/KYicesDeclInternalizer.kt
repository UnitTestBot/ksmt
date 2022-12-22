package org.ksmt.solver.yices

import com.sri.yices.Terms
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

            Terms.newUninterpretedFunction(decl.name, argSorts + decl.sort.accept(sortInternalizer))
        }

    override fun <S : KSort> visit(decl: KConstDecl<S>): YicesTerm =
        yicesCtx.internalizeDecl(decl) {
            Terms.newUninterpretedTerm(decl.name, decl.sort.accept(sortInternalizer))
        }
}
