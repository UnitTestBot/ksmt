package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KDistinctExpr
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExpr
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.utils.asExpr

class KExprSimplifier(ctx: KContext) :
    KNonRecursiveTransformer(ctx),
    KExprSimplifierBase,
    KBoolExprSimplifier,
    KArithExprSimplifier,
    KBvExprSimplifier,
    KFpExprSimplifier,
    KArrayExprSimplifier {

    private val rewrittenExpressions = hashMapOf<KExpr<*>, KExpr<*>>()

    override fun <T : KSort> transform(expr: KEqExpr<T>) = simplifyApp(expr) { (lhs, rhs) ->
        with(ctx) {
            when (val sort = lhs.sort) {
                boolSort -> simplifyEqBool(lhs.asExpr(boolSort), rhs.asExpr(boolSort))
                intSort -> simplifyEqInt(lhs.asExpr(intSort), rhs.asExpr(intSort))
                realSort -> simplifyEqReal(lhs.asExpr(realSort), rhs.asExpr(realSort))
                is KBvSort -> simplifyEqBv(lhs.asExpr(sort), rhs.asExpr(sort))
                is KFpSort -> simplifyEqFp(lhs.asExpr(sort), rhs.asExpr(sort))
                is KArraySort<*, *> -> simplifyEqArray(lhs.asExpr(sort), rhs.asExpr(sort))
                else -> mkEq(lhs, rhs)
            }
        }
    }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>): KExpr<KBoolSort> {
        if (expr.args.size <= 1) return ctx.trueExpr

        return simplifyApp(expr) { args ->
            with(ctx) {
                if (args.size == 2) {
                    return@simplifyApp mkNot(mkEq(args[0], args[1]))
                }

                // todo: bool_rewriter.cpp:786
                mkDistinct(args)
            }
        }
    }

    override fun <T : KSort> areDefinitelyDistinct(lhs: KExpr<T>, rhs: KExpr<T>): Boolean = with(ctx){
        if (lhs == rhs) return false
        return when (val sort = lhs.sort) {
            boolSort -> areDefinitelyDistinctBool(lhs.asExpr(boolSort), rhs.asExpr(boolSort))
            intSort -> areDefinitelyDistinctInt(lhs.asExpr(intSort), rhs.asExpr(intSort))
            realSort -> areDefinitelyDistinctReal(lhs.asExpr(realSort), rhs.asExpr(realSort))
            is KBvSort -> areDefinitelyDistinctBv(lhs.asExpr(sort), rhs.asExpr(sort))
            is KFpSort -> areDefinitelyDistinctFp(lhs.asExpr(sort), rhs.asExpr(sort))
            else -> false
        }
    }

    fun <T : KSort> rewrittenOrNull(expr: KExpr<T>): KExpr<T>? {
        val rewritten = rewrittenExpressions.remove(expr) ?: return null
        val result = transformedExpr(rewritten)
            ?: error("Nested rewrite failed")
        return result.asExpr(expr.sort)
    }

    fun postRewrite(original: KExpr<*>, rewritten: KExpr<*>) {
        rewrittenExpressions[original] = rewritten
        original.transformAfter(listOf(rewritten))
        markExpressionAsNotTransformed()
    }

}

inline fun <T : KSort, A : KSort> KExprSimplifierBase.simplifyApp(
    expr: KApp<T, KExpr<A>>,
    crossinline simplifier: KContext.(List<KExpr<A>>) -> KExpr<T>
): KExpr<T> {
    this as KExprSimplifier

    val rewritten = rewrittenOrNull(expr)

    if (rewritten != null) {
        return rewritten
    }

    val transformed = transformAppAfterArgsTransformed(expr) { args -> ctx.simplifier(args) }

    if (transformed != expr) {
        postRewrite(expr, transformed)
        return expr
    }

    return transformed
}
