package io.ksmt.expr.rewrite

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.*
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KSort
import io.ksmt.utils.uncheckedCast

@Suppress("UNCHECKED_CAST")
class KBvQETransformer(ctx: KContext, bound: KDecl<KBvSort>) : KNonRecursiveTransformer(ctx) {
    private var bounds = mutableListOf<KDecl<*>>()
    private var newBody: KExpr<KBoolSort> = ctx.mkTrue()

    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>):
            KExpr<T>  = with(ctx) {
        return mkBvMulNoOverflowExpr(expr.arg0, expr.arg1, isSigned = false).uncheckedCast()
    }



    override fun <T : KSort> transform(expr: KEqExpr<T>):
            KExpr<KBoolSort> = with(ctx) {
        fun eqToAndNoSimplify(l: KExpr<KBvSort>, r: KExpr<KBvSort>): KExpr<KBoolSort> =
            mkAndNoSimplify(
                mkBvUnsignedLessOrEqualExprNoSimplify(l, r),
                mkBvUnsignedLessOrEqualExprNoSimplify(r, l)
            )

        expr as KEqExpr<KBvSort>
        return transformExprAfterTransformedDefault(
            expr, expr.lhs, expr.rhs, { eq -> eqToAndNoSimplify(eq.lhs, eq.rhs)})
        { l, r -> eqToAndNoSimplify(l, r) }
    }

    companion object {
        fun transformBody(body: KExpr<KBoolSort>, bound: KDecl<KBvSort>): Pair<KExpr<KBoolSort>, List<KDecl<*>>> =
            with(KBvQETransformer(body.ctx, bound))
            {
                newBody = apply(body)
                return newBody to bounds
            }
    }

}