package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KArray2Lambda
import org.ksmt.expr.KArray3Lambda
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KArrayLambdaBase
import org.ksmt.expr.KArrayNLambda
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFpToIEEEBvExpr
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
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
        transformExprAfterTransformed(expr, expr.value) { value ->
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

    // (exists (a, b) (= a (f b))) ==> (exists (a, b, c) (and (= a c) (= b (f^-1 c))))
    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> {
        val nestedRewriter = KBitwuzlaInternalizationAxioms(ctx)
        val exprWithAxioms = nestedRewriter.rewriteWithAxioms(expr.body)
        val rewrittenBody = ctx.mkAnd(listOf(exprWithAxioms.expr) + exprWithAxioms.axioms)
        return ctx.mkExistentialQuantifier(rewrittenBody, expr.bounds + exprWithAxioms.stubs)
    }

    // (forall (a, b) (= a (f b))) ==> (forall (a, b) (exists (c) (and (= a c) (= b (f^-1 c)))))
    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> {
        val nestedRewriter = KBitwuzlaInternalizationAxioms(ctx)
        val exprWithAxioms = nestedRewriter.rewriteWithAxioms(expr.body)

        val body = if (exprWithAxioms.axioms.isEmpty()) {
            exprWithAxioms.expr
        } else {
            val bodyWithAxioms = ctx.mkAnd(listOf(exprWithAxioms.expr) + exprWithAxioms.axioms)
            ctx.mkExistentialQuantifier(bodyWithAxioms, exprWithAxioms.stubs)
        }

        return ctx.mkUniversalQuantifier(body, expr.bounds)
    }

    override fun <D : KSort, R : KSort> transform(
        expr: KArrayLambda<D, R>
    ): KExpr<KArraySort<D, R>> = rewriteArrayLambdaWithAxiom(expr, expr.body) { array ->
        mkArraySelect(array, mkConstApp(expr.indexVarDecl))
    }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Lambda<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = rewriteArrayLambdaWithAxiom(expr, expr.body) { array ->
        mkArraySelect(array, mkConstApp(expr.indexVar0Decl), mkConstApp(expr.indexVar1Decl))
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Lambda<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = rewriteArrayLambdaWithAxiom(expr, expr.body) { array ->
        mkArraySelect(
            array,
            mkConstApp(expr.indexVar0Decl),
            mkConstApp(expr.indexVar1Decl),
            mkConstApp(expr.indexVar2Decl)
        )
    }

    override fun <R : KSort> transform(
        expr: KArrayNLambda<R>
    ): KExpr<KArrayNSort<R>> = rewriteArrayLambdaWithAxiom(expr, expr.body) { array ->
        mkArrayNSelect(array, expr.indexVarDeclarations.map { mkConstApp(it) })
    }

    // (lambda (i) (f i)) ==> array | (forall (i) (= (select array i) (f i)))
    private inline fun <A : KArraySortBase<R>, R : KSort> rewriteArrayLambdaWithAxiom(
        lambda: KArrayLambdaBase<A, R>,
        body: KExpr<R>,
        mkSelect: KContext.(KExpr<A>) -> KExpr<R>,
    ): KExpr<A> = with(ctx) {
        val arrayStub = mkFreshConst("lambda", lambda.sort)
        val arrayValue = mkSelect(arrayStub)
        val bodyExprStub = arrayValue eq body

        val nestedRewriter = KBitwuzlaInternalizationAxioms(ctx)
        val exprWithAxioms = nestedRewriter.rewriteWithAxioms(bodyExprStub)

        // Lambda expression body has no rewritten expressions
        if (exprWithAxioms.axioms.isEmpty()) return lambda

        val bodyWithAxioms = mkAnd(listOf(exprWithAxioms.expr) + exprWithAxioms.axioms)
        val lambdaAxiomBody = mkExistentialQuantifier(bodyWithAxioms, exprWithAxioms.stubs)
        val lambdaAxiom = mkUniversalQuantifier(lambdaAxiomBody, lambda.indexVarDeclarations)

        axioms += lambdaAxiom
        stubs += arrayStub.decl
        arrayStub
    }

    fun rewriteWithAxioms(expr: KExpr<KBoolSort>): ExpressionWithAxioms {
        val result = apply(expr)
        return ExpressionWithAxioms(result, axioms, stubs)
    }
}
