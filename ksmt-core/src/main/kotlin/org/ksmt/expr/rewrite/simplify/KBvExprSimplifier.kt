package org.ksmt.expr.rewrite.simplify

import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KBvConcatExpr
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort

interface KBvExprSimplifier : KExprSimplifierBase {

    fun <T : KBvSort> simplifyEqBv(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> = with(ctx) {
        if (lhs == rhs) return trueExpr

        if (lhs is KBitVecValue<*> && rhs is KBitVecValue<*>) {
            return falseExpr
        }

        if (lhs is KBvConcatExpr || rhs is KBvConcatExpr) {
            return simplifyBvConcatEq(lhs, rhs)
        }

        // todo: bv_rewriter.cpp:2681

        mkEq(lhs, rhs)
    }

    fun <T : KBvSort> areDefinitelyDistinctBv(lhs: KExpr<T>, rhs: KExpr<T>): Boolean {
        if (lhs is KBitVecValue<*> && rhs is KBitVecValue<*>) {
            return lhs != rhs
        }
        return false
    }

    /**
     * (= (concat a b) c) ==>
     *  (and
     *      (= a (extract (0, <a_size>) c))
     *      (= b (extract (<a_size>, <a_size> + <b_size>) c))
     *  )
     * */
    fun <T : KBvSort> simplifyBvConcatEq(l: KExpr<T>, r: KExpr<T>): KExpr<KBoolSort> = with(ctx) {
        val lArgs = if (l is KBvConcatExpr) flatConcat(l) else listOf(l)
        val rArgs = if (r is KBvConcatExpr) flatConcat(r) else listOf(r)
        val newEqualities = arrayListOf<KExpr<KBoolSort>>()
        var lowL = 0
        var lowR = 0
        var lIdx = lArgs.size
        var rIdx = rArgs.size
        while (lIdx > 0 && rIdx > 0) {
            val lArg = lArgs[lIdx - 1]
            val rArg = rArgs[rIdx - 1]
            val lSize = lArg.sort.sizeBits.toInt()
            val rSize = rArg.sort.sizeBits.toInt()
            val remainSizeL = lSize - lowL
            val remainSizeR = rSize - lowR
            when {
                remainSizeL == remainSizeR -> {
                    val newL = mkBvExtractExpr(high = lSize - 1, low = lowL, value = lArg)
                    val newR = mkBvExtractExpr(high = rSize - 1, low = lowR, value = rArg)
                    newEqualities += newL eq newR
                    lowL = 0
                    lowR = 0
                    lIdx--
                    rIdx--
                }

                remainSizeL < remainSizeR -> {
                    val newL = mkBvExtractExpr(high = lSize - 1, low = lowL, value = lArg)
                    val newR = mkBvExtractExpr(high = remainSizeL + lowR - 1, low = lowR, value = rArg)
                    newEqualities += newL eq newR
                    lowL = 0
                    lowR += remainSizeL
                    lIdx--
                }

                else -> {
                    val newL = mkBvExtractExpr(high = remainSizeR + lowL - 1, low = lowL, value = lArg)
                    val newR = mkBvExtractExpr(high = rSize - 1, low = lowR, value = rArg)
                    newEqualities += newL eq newR
                    lowL += remainSizeR
                    lowR = 0
                    rIdx--
                }
            }
        }
        return mkAnd(newEqualities)
    }

    private fun flatConcat(expr: KBvConcatExpr): List<KExpr<KBvSort>> {
        val flatten = arrayListOf<KExpr<KBvSort>>()
        val unprocessed = arrayListOf<KExpr<KBvSort>>()
        unprocessed += expr
        while (unprocessed.isNotEmpty()) {
            val e = unprocessed.removeLast()
            if (e !is KBvConcatExpr) {
                flatten += e
                continue
            }
            unprocessed += e.arg1
            unprocessed += e.arg0
        }
        return flatten
    }
}
