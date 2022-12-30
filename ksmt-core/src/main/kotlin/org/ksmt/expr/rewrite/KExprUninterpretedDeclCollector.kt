package org.ksmt.expr.rewrite

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KConst
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFunctionApp
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort

/**
 * Collect declarations of used uninterpreted constants and functions.
 * */
open class KExprUninterpretedDeclCollector(ctx: KContext) : KNonRecursiveTransformer(ctx) {
    private val declarations = hashSetOf<KDecl<*>>()

    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> {
        declarations += expr.decl
        return super.transform(expr)
    }

    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> {
        declarations += expr.decl
        return super.transform(expr)
    }

    override fun <D : KSort, R : KSort> transform(expr: KFunctionAsArray<D, R>): KExpr<KArraySort<D, R>> {
        declarations += expr.function
        return super.transform(expr)
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>> {
        transformQuantifier(setOf(expr.indexVarDecl), expr.body)
        return expr
    }

    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> {
        transformQuantifier(expr.bounds.toSet(), expr.body)
        return expr
    }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> {
        transformQuantifier(expr.bounds.toSet(), expr.body)
        return expr
    }

    private fun transformQuantifier(bounds: Set<KDecl<*>>, body: KExpr<*>) {
        val usedDecls = collectUninterpretedDeclarations(body) - bounds
        declarations += usedDecls
    }
    
    companion object{
        fun collectUninterpretedDeclarations(expr: KExpr<*>): Set<KDecl<*>> =
            KExprUninterpretedDeclCollector(expr.ctx)
                .apply { apply(expr) }
                .declarations
    }
}
