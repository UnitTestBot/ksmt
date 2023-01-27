package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.expr.KAndExpr
import org.ksmt.expr.KEqExpr
import org.ksmt.sort.KFpSort

class Compare {
    companion object {
        fun <F : KFpSort> KContext.fpToBvExpr(expr: KEqExpr<F>): KAndExpr {
            val left = expr.lhs
            val right = expr.rhs

            // All comparison with NaN are false
            val neitherNan = mkFpIsNaNExpr(left).not() and mkFpIsNaNExpr(right).not()

            val leftIsZero = mkFpIsZeroExpr(left)
            val rightIsZero = mkFpIsZeroExpr(right)
            val bothZero = leftIsZero and rightIsZero
            val neitherZero = leftIsZero.not() and rightIsZero.not()

            val bitEq = mkFpToIEEEBvExpr(left) eq mkFpToIEEEBvExpr(right)

            return neitherNan and (bothZero or (neitherZero and (mkFpIsInfiniteExpr(left) eq mkFpIsInfiniteExpr(right) and bitEq)))
        }


    }

}