package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.expr.*
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.symfpu.UnpackedFp.Companion.unpackedFp
import org.ksmt.utils.cast

class FpToBvTransformer(ctx: KContext) : KNonRecursiveTransformer(ctx) {
    override fun <Fp : KFpSort> transform(expr: KFpEqualExpr<Fp>): KExpr<KBoolSort> = with(ctx) {
        transformExprAfterTransformed(expr, expr.args) { args ->
            val left = args[0] as UnpackedFp<Fp>
            val right = args[1] as UnpackedFp<Fp>

            // All comparison with NaN are false
            val neitherNan = left.isNaN.not() and right.isNaN.not()

            val bothZero = left.isZero and right.isZero
            val neitherZero = left.isZero.not() and right.isZero.not()
            val bitEq = left.bv eq right.bv

            return neitherNan and (bothZero or (neitherZero and (left.isInfinite eq right.isInfinite and bitEq)))
        }
    }

    override fun <Fp : KFpSort> transform(expr: KFpLessExpr<Fp>): KExpr<KBoolSort> = with(ctx) {
        transformExprAfterTransformed(expr, expr.args) { args ->
            val left = args[0] as UnpackedFp<Fp>
            val right = args[1] as UnpackedFp<Fp>

            // All comparison with NaN are false
            val neitherNan = left.isNaN.not() and right.isNaN.not()


            // Infinities are bigger than everything but themselves
            val eitherInf = left.isInfinite or right.isInfinite
            val infCase =
                (left.isNegativeInfinity and right.isNegativeInfinity.not()) or (left.isPositiveInfinity.not() and right.isPositiveInfinity)


            // Both zero are equal
            val eitherZero = left.isZero or right.isZero


            val zeroCase =
                (left.isZero and right.isZero.not() and right.isNegative.not()) or (left.isZero.not() and left.isNegative and right.isZero)


            // Normal and subnormal
            val negativeLessThanPositive = left.isNegative and right.isNegative.not()

            val positiveCase = left.isNegative.not() and right.isNegative.not() and (mkBvUnsignedLessExpr(
                left.exponent,
                right.exponent
            ) or (left.exponent eq right.exponent and mkBvUnsignedLessExpr(left.significand, right.significand)))


            val negativeCase = left.isNegative and right.isNegative and (mkBvUnsignedLessExpr(
                right.exponent,
                left.exponent
            ) or (left.exponent eq right.exponent and mkBvUnsignedLessExpr(right.significand, left.significand)))


            return neitherNan and mkIte(
                eitherInf, infCase, mkIte(
                    eitherZero, zeroCase, negativeLessThanPositive or positiveCase or negativeCase
                )
            )
        }
    }

    override fun <Fp : KFpSort> transform(expr: KFpMinExpr<Fp>): KExpr<Fp> = with(ctx) {
        transformExprAfterTransformed(expr, expr.args) { args ->
            val right = args[1] as UnpackedFp<Fp>

            return mkIte(
                right.isNaN or transform(mkFpLessExpr(expr.arg0, expr.arg1)), expr.arg0, expr.arg1
            )
        }
    }


    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = with(ctx) {
        if (expr.sort is KFpSort) {
            val asFp: KExpr<KFpSort> = expr.cast()
            return unpackedFp(asFp).cast()
        }
        return expr
    }

    override fun <Fp : KFpSort> transformFpValue(expr: KFpValue<Fp>): KExpr<Fp> = with(ctx) {
        return unpackedFp(expr)
    }
}
