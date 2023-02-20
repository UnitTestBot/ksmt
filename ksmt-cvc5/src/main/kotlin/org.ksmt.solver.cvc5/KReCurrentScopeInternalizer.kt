package org.ksmt.solver.cvc5

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KApp
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KConst
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFunctionApp
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.rewrite.KExprUninterpretedDeclCollector
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort


private class KReCurrentScopeInternalizer(
    private val cvc5Ctx: KCvc5Context, ctx: KContext
) : KNonRecursiveTransformer(ctx) {
    override fun <T : KSort, A : KSort> transformApp(expr: KApp<T, A>): KExpr<T> {
        if (cvc5Ctx.findCurrentScopeInternalizedExpr(expr) != null)
            return expr

        cvc5Ctx.savePreviouslyInternalizedExpr(expr)
        return super.transformApp(expr)
    }

    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> {
        cvc5Ctx.addDeclaration(expr.decl)
        cvc5Ctx.collectUninterpretedSorts(expr.decl)
        cvc5Ctx.savePreviouslyInternalizedExpr(expr)
        return super.transform(expr)
    }

    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> {
        cvc5Ctx.addDeclaration(expr.decl)
        cvc5Ctx.collectUninterpretedSorts(expr.decl)
        cvc5Ctx.savePreviouslyInternalizedExpr(expr)
        return super.transform(expr)
    }

    override fun <D : KSort, R : KSort> transform(expr: KFunctionAsArray<D, R>): KExpr<KArraySort<D, R>> {
        cvc5Ctx.addDeclaration(expr.function)
        cvc5Ctx.collectUninterpretedSorts(expr.function)
        cvc5Ctx.savePreviouslyInternalizedExpr(expr)
        return super.transform(expr)
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>> {
        transformQuantifier(setOf(expr.indexVarDecl), expr.body)
        cvc5Ctx.savePreviouslyInternalizedExpr(expr)
        return expr
    }

    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> {
        transformQuantifier(expr.bounds.toSet(), expr.body)
        cvc5Ctx.savePreviouslyInternalizedExpr(expr)
        return expr
    }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> {
        transformQuantifier(expr.bounds.toSet(), expr.body)
        cvc5Ctx.savePreviouslyInternalizedExpr(expr)
        return expr
    }

    private fun transformQuantifier(bounds: Set<KDecl<*>>, body: KExpr<*>) {
        val usedDecls = KExprUninterpretedDeclCollector.collectUninterpretedDeclarations(body) - bounds
        usedDecls.forEach(cvc5Ctx::addDeclaration)
    }
}

internal fun KCvc5Context.collectCurrentLevelInternalizations(expr: KExpr<*>) {
    KReCurrentScopeInternalizer(this, expr.ctx).apply(expr)
}
