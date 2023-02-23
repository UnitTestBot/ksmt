package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.expr.*
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.utils.asExpr
import org.ksmt.utils.cast

class FpToBvTransformer(ctx: KContext) : KNonRecursiveTransformer(ctx) {
    // use this function instead of apply as it may return UnpackedFp wrapper

//    fun <T : KSort> applyAndGetBvExpr(expr: KExpr<T>): KExpr<KBvSort> {
//        val transformed = apply(expr)
//        val unpacked = (transformed as? UnpackedFp)
//        return unpacked ?: transformed.cast()
//    }

    override fun <Fp : KFpSort> transform(expr: KFpEqualExpr<Fp>): KExpr<KBoolSort> = with(ctx) {
        transformHelper(expr, ::equal)
    }

    override fun <Fp : KFpSort> transform(expr: KFpLessExpr<Fp>): KExpr<KBoolSort> = with(ctx) {
        transformHelper(expr, ::less)
    }

    override fun <Fp : KFpSort> transform(expr: KFpMulExpr<Fp>): KExpr<Fp> = with(ctx) {
        val args1: List<KExpr<Fp>> = expr.args.cast()
        transformExprAfterTransformed(expr, args1) { args ->
            val (left, right) = argsToTypedPair(args.drop(1))
            multiply(left, right, expr.roundingMode)
        }
    }

    override fun <Fp : KFpSort> transform(expr: KFpLessOrEqualExpr<Fp>): KExpr<KBoolSort> = with(ctx) {
        transformHelper(expr, ::lessOrEqual)
    }

    override fun <Fp : KFpSort> transform(expr: KFpGreaterExpr<Fp>): KExpr<KBoolSort> = with(ctx) {
        transformHelper(expr, ::greater)
    }

    override fun <Fp : KFpSort> transform(expr: KFpGreaterOrEqualExpr<Fp>): KExpr<KBoolSort> = with(ctx) {
        transformHelper(expr, ::greaterOrEqual)
    }

    override fun <Fp : KFpSort> transform(expr: KFpMinExpr<Fp>): KExpr<Fp> = with(ctx) {
        transformHelper(expr, ::min)
    }

    override fun <Fp : KFpSort> transform(expr: KFpMaxExpr<Fp>): KExpr<Fp> = with(ctx) {
        transformHelper(expr, ::max)
    }

    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = with(ctx) {
        return if (expr.sort is KFpSort) {
            val asFp: KConst<KFpSort> = expr.cast()
            unpack(asFp.sort, mkFpToIEEEBvExpr(asFp)).cast()
        } else expr
    }

    override fun <Fp : KFpSort> transformFpValue(expr: KFpValue<Fp>): KExpr<Fp> = with(ctx) {
        return unpack(
            expr.sort,
            expr.signBit.expr,
            expr.biasedExponent.asExpr(mkBvSort(expr.sort.exponentBits)),
            expr.significand.asExpr(mkBvSort(expr.sort.significandBits - 1u)),
            mkFpToIEEEBvExpr(expr)
        )
    }


    private fun <Fp : KFpSort> argsToTypedPair(args: List<KExpr<Fp>>): Pair<UnpackedFp<Fp>, UnpackedFp<Fp>> {
        val left = args[0] as UnpackedFp<Fp>
        val right = args[1] as UnpackedFp<Fp>
        return Pair(left, right)
    }

    private fun <Fp : KFpSort, R : KSort> transformHelper(
        expr: KApp<R, Fp>, f: (UnpackedFp<Fp>, UnpackedFp<Fp>) -> KExpr<R>
    ): KExpr<R> =
        transformExprAfterTransformed(expr, expr.args) { args ->
            val (left, right) = argsToTypedPair(args)
            f(left, right)
        }
}
