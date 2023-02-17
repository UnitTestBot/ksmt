package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFpToIEEEBvExpr
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.utils.uncheckedCast

class KBitwuzlaInternalizationAxioms(ctx: KContext) : KNonRecursiveTransformer(ctx) {
    private val axioms = arrayListOf<KExpr<KBoolSort>>()
    private val stubs = arrayListOf<KDecl<*>>()

    class ExpressionWithAxioms(
        val expr: KExpr<KBoolSort>,
        val axioms: List<KExpr<KBoolSort>>,
        val stubs: List<KDecl<*>>
    )

    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> =
        transformAppAfterArgsTransformed(expr) { (value) ->
            with(ctx) {
                val stub = mkFreshConst("fpToIEEEBv", expr.sort)
                stubs += stub.decl

                val size = stub.sort.sizeBits.toInt()
                val exponentBits = value.sort.exponentBits.toInt()

                val signBitIdx = size - 1
                val exponentFirstBitIdx = signBitIdx - 1
                val exponentLastBitIdx = size - exponentBits - 1
                val significandFirstBitIdx = exponentLastBitIdx - 1

                val sign = mkBvExtractExpr(high = signBitIdx, low = signBitIdx, value = stub)
                val exponent = mkBvExtractExpr(high = exponentFirstBitIdx, low = exponentLastBitIdx, value = stub)
                val significand = mkBvExtractExpr(high = significandFirstBitIdx, low = 0, value = stub)

                val inverseOperation = mkFpFromBvExpr<T>(sign.uncheckedCast(), exponent, significand)
                axioms += value eq inverseOperation

                stub
            }
        }

    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>> = with(ctx) {
        val stub = mkFreshConst("arrayLambda", expr.sort)
        stubs += stub.decl

        val indexVar = mkConstApp(expr.indexVarDecl)
        val quantifierBody = stub.select(indexVar) eq expr.body

        val nestedRewriter = KBitwuzlaInternalizationAxioms(ctx)
        val exprWithAxioms = nestedRewriter.rewriteWithAxioms(quantifierBody)
        val rewrittenBody = ctx.mkAnd(listOf(exprWithAxioms.expr) + exprWithAxioms.axioms)
        val newBounds = listOf(expr.indexVarDecl) + exprWithAxioms.stubs
        val axiomQuantifier = ctx.mkUniversalQuantifier(rewrittenBody, newBounds)
        axioms += axiomQuantifier

        stub
    }

    // (exists (a, b) (= a (f b))) ==> (exists (a, b, c) (and (= a c) (= b (f^-1 c))))
    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> {
        val nestedRewriter = KBitwuzlaInternalizationAxioms(ctx)
        val exprWithAxioms = nestedRewriter.rewriteWithAxioms(expr.body)
        val rewrittenBody = ctx.mkAnd(listOf(exprWithAxioms.expr) + exprWithAxioms.axioms)
        return ctx.mkExistentialQuantifier(rewrittenBody, expr.bounds + exprWithAxioms.stubs)
    }

    // (forall (a, b) (= a (f b))) ==> (forall (a, b, c) (and (= a c) (= b (f^-1 c))))
    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> {
        val nestedRewriter = KBitwuzlaInternalizationAxioms(ctx)
        val exprWithAxioms = nestedRewriter.rewriteWithAxioms(expr.body)
        val rewrittenBody = ctx.mkAnd(listOf(exprWithAxioms.expr) + exprWithAxioms.axioms)
        return ctx.mkUniversalQuantifier(rewrittenBody, expr.bounds + exprWithAxioms.stubs)
    }

    fun rewriteWithAxioms(expr: KExpr<KBoolSort>): ExpressionWithAxioms {
        val result = apply(expr)
        return ExpressionWithAxioms(result, axioms, stubs)
    }
}
