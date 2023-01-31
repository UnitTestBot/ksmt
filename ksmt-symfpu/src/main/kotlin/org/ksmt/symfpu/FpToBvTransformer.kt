package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFpEqualExpr
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KFpSort
import org.ksmt.symfpu.UnpackedFp.Companion.unpackedFp

class FpToBvTransformer(ctx: KContext) : KNonRecursiveTransformer(ctx) {
    override fun <Fp : KFpSort> transform(expr: KFpEqualExpr<Fp>): KExpr<KBoolSort> = with(ctx) {
        val left = unpackedFp(expr.arg0)
        val right = unpackedFp(expr.arg1)

        // All comparison with NaN are false
        val neitherNan = left.isNaN.not() and right.isNaN.not()

        val bothZero = left.isZero and right.isZero
        val neitherZero = left.isZero.not() and right.isZero.not()
        val bitEq = left.bv eq right.bv

        return neitherNan and (bothZero or (neitherZero and (left.isInfinite eq right.isInfinite and bitEq)))
    }
}