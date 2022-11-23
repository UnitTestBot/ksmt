package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.expr.KBitVec16Value
import org.ksmt.expr.KBitVec1Value
import org.ksmt.expr.KBitVec32Value
import org.ksmt.expr.KBitVec64Value
import org.ksmt.expr.KBitVec8Value
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KBv2IntExpr
import org.ksmt.expr.KBvAddExpr
import org.ksmt.expr.KBvAddNoOverflowExpr
import org.ksmt.expr.KBvAddNoUnderflowExpr
import org.ksmt.expr.KBvAndExpr
import org.ksmt.expr.KBvArithShiftRightExpr
import org.ksmt.expr.KBvConcatExpr
import org.ksmt.expr.KBvDivNoOverflowExpr
import org.ksmt.expr.KBvExtractExpr
import org.ksmt.expr.KBvLogicalShiftRightExpr
import org.ksmt.expr.KBvMulExpr
import org.ksmt.expr.KBvMulNoOverflowExpr
import org.ksmt.expr.KBvMulNoUnderflowExpr
import org.ksmt.expr.KBvNAndExpr
import org.ksmt.expr.KBvNegNoOverflowExpr
import org.ksmt.expr.KBvNegationExpr
import org.ksmt.expr.KBvNorExpr
import org.ksmt.expr.KBvNotExpr
import org.ksmt.expr.KBvOrExpr
import org.ksmt.expr.KBvReductionAndExpr
import org.ksmt.expr.KBvReductionOrExpr
import org.ksmt.expr.KBvRepeatExpr
import org.ksmt.expr.KBvRotateLeftExpr
import org.ksmt.expr.KBvRotateLeftIndexedExpr
import org.ksmt.expr.KBvRotateRightExpr
import org.ksmt.expr.KBvRotateRightIndexedExpr
import org.ksmt.expr.KBvShiftLeftExpr
import org.ksmt.expr.KBvSignExtensionExpr
import org.ksmt.expr.KBvSignedDivExpr
import org.ksmt.expr.KBvSignedGreaterExpr
import org.ksmt.expr.KBvSignedGreaterOrEqualExpr
import org.ksmt.expr.KBvSignedLessExpr
import org.ksmt.expr.KBvSignedLessOrEqualExpr
import org.ksmt.expr.KBvSignedModExpr
import org.ksmt.expr.KBvSignedRemExpr
import org.ksmt.expr.KBvSubExpr
import org.ksmt.expr.KBvSubNoOverflowExpr
import org.ksmt.expr.KBvSubNoUnderflowExpr
import org.ksmt.expr.KBvUnsignedDivExpr
import org.ksmt.expr.KBvUnsignedGreaterExpr
import org.ksmt.expr.KBvUnsignedGreaterOrEqualExpr
import org.ksmt.expr.KBvUnsignedLessExpr
import org.ksmt.expr.KBvUnsignedLessOrEqualExpr
import org.ksmt.expr.KBvUnsignedRemExpr
import org.ksmt.expr.KBvXNorExpr
import org.ksmt.expr.KBvXorExpr
import org.ksmt.expr.KBvZeroExtensionExpr
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KIntSort
import org.ksmt.utils.asExpr
import java.math.BigInteger

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

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            bvLessOrEqual(lhs, rhs, signed = false)
        }

    override fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            bvLessOrEqual(lhs, rhs, signed = true)
        }

    // (uge a b) ==> (ule b a)
    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            mkBvUnsignedLessOrEqualExpr(rhs, lhs)
        }

    // (ult a b) ==> (not (ule b a))
    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            mkNot(mkBvUnsignedLessOrEqualExpr(rhs, lhs))
        }

    // (ugt a b) ==> (not (ule a b))
    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            mkNot(mkBvUnsignedLessOrEqualExpr(lhs, rhs))
        }

    // (sge a b) ==> (sle b a)
    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            mkBvSignedLessOrEqualExpr(rhs, lhs)
        }

    // (slt a b) ==> (not (sle b a))
    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            mkNot(mkBvSignedLessOrEqualExpr(rhs, lhs))
        }

    // (sgt a b) ==> (not (sle a b))
    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            mkNot(mkBvSignedLessOrEqualExpr(lhs, rhs))
        }

    private fun <T : KBvSort> bvLessOrEqual(lhs: KExpr<T>, rhs: KExpr<T>, signed: Boolean): KExpr<KBoolSort> =
        with(ctx) {
            if (lhs == rhs) return trueExpr

            val lhsValue = lhs as? KBitVecValue<*>
            val rhsValue = rhs as? KBitVecValue<*>

            if (lhsValue != null && rhsValue != null) {
                val result = if (signed) {
                    lhsValue.signedLessOrEqual(rhsValue)
                } else {
                    lhsValue.unsignedLessOrEqual(rhsValue)
                }
                return result.expr
            }

            if (lhsValue != null || rhsValue != null) {
                val size = lhs.sort.sizeBits
                val (lower, upper) = if (signed) {
                    minValueSigned(size) to maxValueSigned(size)
                } else {
                    zero(size) to maxValueUnsigned(size)
                }

                if (rhsValue != null) {
                    // a <= b, b == MIN_VALUE ==> a == b
                    if (rhsValue == lower) {
                        return (lhs eq rhs.asExpr(lhs.sort))
                    }
                    // a <= b, b == MAX_VALUE ==> true
                    if (rhsValue == upper) {
                        return trueExpr
                    }
                }

                if (lhsValue != null) {
                    // a <= b, a == MIN_VALUE ==> true
                    if (lhsValue == lower) {
                        return trueExpr
                    }
                    // a <= b, a == MAX_VALUE ==> a == b
                    if (lhsValue == upper) {
                        return (lhs.asExpr(lhs.sort) eq rhs)
                    }
                }
            }

            // todo: bv_rewriter.cpp:433

            if (signed) {
                mkBvSignedLessOrEqualExpr(lhs, rhs)
            } else {
                mkBvUnsignedLessOrEqualExpr(lhs, rhs)
            }
        }

    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val lhsValue = lhs as? KBitVecValue<*>
        val rhsValue = rhs as? KBitVecValue<*>

        if (lhsValue != null) {
            if (rhsValue != null) {
                return@simplifyApp (lhsValue + rhsValue).asExpr(expr.sort)
            }

            trySimplifyBvAddWithConstant(lhsValue, rhs, expr.sort)?.let { return@simplifyApp it }
        }

        if (rhsValue != null) {
            trySimplifyBvAddWithConstant(rhsValue, lhs, expr.sort)?.let { return@simplifyApp it }
        }

        mkBvAddExpr(lhs, rhs)
    }

    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        mkBvAddExpr(lhs, mkBvNegationExpr(rhs))
    }

    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val lhsValue = lhs as? KBitVecValue<*>
        val rhsValue = rhs as? KBitVecValue<*>

        if (lhsValue != null) {
            if (rhsValue != null) {
                return@simplifyApp (lhsValue * rhsValue).asExpr(expr.sort)
            }

            trySimplifyBvMulWithConstant(lhsValue, rhs, expr.sort)?.let { return@simplifyApp it }
        }

        if (rhsValue != null) {
            trySimplifyBvMulWithConstant(rhsValue, lhs, expr.sort)?.let { return@simplifyApp it }
        }

        mkBvMulExpr(lhs, rhs)
    }

    private fun <T : KBvSort> trySimplifyBvAddWithConstant(
        value: KBitVecValue<*>,
        other: KExpr<T>,
        sort: T
    ): KExpr<T>? = with(ctx) {
        // (+ 0 a) ==> a
        if (value == zero(value.sort.sizeBits)) {
            return other
        }

        if (other is KBvAddExpr<*>) {
            tryFlatBvAdd(value, other, sort)?.let { return it }
        }

        return null
    }

    private fun <T : KBvSort> trySimplifyBvMulWithConstant(
        value: KBitVecValue<*>,
        other: KExpr<T>,
        sort: T
    ): KExpr<T>? = with(ctx) {
        val size = value.sort.sizeBits

        // (* 0 a) ==> 0
        if (value == zero(size)) {
            return zero(size).asExpr(sort)
        }

        // (* 1 a) ==> a
        if (value == one(size)) {
            return other
        }

        // (* -1 a) ==> -a
        val minusOne = zero(size) - one(size)
        if (value == minusOne) {
            return mkBvNegationExpr(other).asExpr(sort)
        }

        if (other is KBvMulExpr<*>) {
            tryFlatBvMul(value, other, sort)?.let { return it }
        }

        return null
    }

    // (+ const1 (+ const2 x)) ==> (+ (+ const1 const2) x)
    private fun <T : KBvSort> tryFlatBvAdd(value: KBitVecValue<*>, other: KBvAddExpr<*>, sort: T): KExpr<T>? =
        tryFlatBvOperation(
            value = value,
            lhs = other.arg0.asExpr(sort),
            rhs = other.arg1.asExpr(sort),
            sort = sort,
            operation = { a, b -> a + b },
            mkExpr = { a, b -> mkBvAddExpr(a, b) }
        )

    // (* const1 (* const2 x)) ==> (* (* const1 const2) x)
    private fun <T : KBvSort> tryFlatBvMul(value: KBitVecValue<*>, other: KBvMulExpr<*>, sort: T): KExpr<T>? =
        tryFlatBvOperation(
            value = value,
            lhs = other.arg0.asExpr(sort),
            rhs = other.arg1.asExpr(sort),
            sort = sort,
            operation = { a, b -> a * b },
            mkExpr = { a, b -> mkBvMulExpr(a, b) }
        )

    private inline fun <T : KBvSort> tryFlatBvOperation(
        value: KBitVecValue<*>,
        lhs: KExpr<T>,
        rhs: KExpr<T>,
        sort: T,
        operation: (KBitVecValue<*>, KBitVecValue<*>) -> KBitVecValue<*>,
        mkExpr: KContext.(KExpr<T>, KExpr<T>) -> KExpr<T>
    ): KExpr<T>? = with(ctx) {
        val lhsValue = lhs as? KBitVecValue<*>
        val rhsValue = rhs as? KBitVecValue<*>

        if (lhsValue != null) {
            val valueBv = operation(value, lhsValue).asExpr(sort)
            return mkExpr(valueBv, rhs)
        }

        if (rhsValue != null) {
            val valueBv = operation(value, rhsValue).asExpr(sort)
            return mkExpr(valueBv, lhs)
        }

        return null
    }

    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>): KExpr<T> = simplifyApp(expr) { (arg) ->
        if (arg is KBitVecValue<*>) {
            return@simplifyApp (zero(expr.sort.sizeBits) - arg).asExpr(expr.sort)
        }
        mkBvNegationExpr(arg)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val size = expr.sort.sizeBits
        val lhsValue = lhs as? KBitVecValue<*>
        val rhsValue = rhs as? KBitVecValue<*>

        if (rhsValue != null) {
            // ignore zero
            if (rhsValue == zero(size)) {
                return@simplifyApp mkBvSignedDivExpr(lhs, rhs.asExpr(lhs.sort))
            }

            if (rhsValue == one(size)) {
                return@simplifyApp lhs
            }

            if (lhsValue != null) {
                return@simplifyApp lhsValue.signedDivide(rhsValue).asExpr(expr.sort)
            }
        }

        mkBvSignedDivExpr(lhs, rhs)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val size = expr.sort.sizeBits
        val lhsValue = lhs as? KBitVecValue<*>
        val rhsValue = rhs as? KBitVecValue<*>

        if (rhsValue != null) {
            // ignore zero
            if (rhsValue == zero(size)) {
                return@simplifyApp mkBvUnsignedDivExpr(lhs, rhs.asExpr(lhs.sort))
            }

            if (rhsValue == one(size)) {
                return@simplifyApp lhs
            }

            if (lhsValue != null) {
                return@simplifyApp lhsValue.unsignedDivide(rhsValue).asExpr(expr.sort)
            }

            rhsValue.powerOfTwoOrNull()?.let { shift ->
                return@simplifyApp mkBvLogicalShiftRightExpr(lhs, mkBv(shift, size).asExpr(lhs.sort))
            }
        }

        mkBvUnsignedDivExpr(lhs, rhs)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val size = expr.sort.sizeBits
        val lhsValue = lhs as? KBitVecValue<*>
        val rhsValue = rhs as? KBitVecValue<*>

        if (rhsValue != null) {
            // ignore zero
            if (rhsValue == zero(size)) {
                return@simplifyApp mkBvSignedRemExpr(lhs, rhs.asExpr(lhs.sort))
            }

            if (rhsValue == one(size)) {
                return@simplifyApp zero(size).asExpr(expr.sort)
            }

            if (lhsValue != null) {
                return@simplifyApp lhsValue.signedRem(rhsValue).asExpr(expr.sort)
            }
        }

        mkBvSignedRemExpr(lhs, rhs)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val size = expr.sort.sizeBits
        val lhsValue = lhs as? KBitVecValue<*>
        val rhsValue = rhs as? KBitVecValue<*>

        if (rhsValue != null) {
            // ignore zero
            if (rhsValue == zero(size)) {
                return@simplifyApp mkBvUnsignedRemExpr(lhs, rhs.asExpr(lhs.sort))
            }

            if (rhsValue == one(size)) {
                return@simplifyApp zero(size).asExpr(expr.sort)
            }

            if (lhsValue != null) {
                return@simplifyApp lhsValue.unsignedRem(rhsValue).asExpr(expr.sort)
            }

            rhsValue.powerOfTwoOrNull()?.let { shift ->
                return@simplifyApp mkBvConcatExpr(
                    zero(size - shift.toUInt()),
                    mkBvExtractExpr(shift - 1, 0, lhs)
                ).asExpr(lhs.sort)
            }
        }

        mkBvUnsignedRemExpr(lhs, rhs)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val size = expr.sort.sizeBits
        val lhsValue = lhs as? KBitVecValue<*>
        val rhsValue = rhs as? KBitVecValue<*>

        if (rhsValue != null) {
            // ignore zero
            if (rhsValue == zero(size)) {
                return@simplifyApp mkBvSignedModExpr(lhs, rhs.asExpr(lhs.sort))
            }

            if (rhsValue == one(size)) {
                return@simplifyApp zero(size).asExpr(expr.sort)
            }

            if (lhsValue != null) {
                return@simplifyApp lhsValue.signedMod(rhsValue).asExpr(expr.sort)
            }
        }

        mkBvSignedModExpr(lhs, rhs)
    }

    override fun <T : KBvSort> transform(expr: KBvNotExpr<T>): KExpr<T> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>): KExpr<T> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>): KExpr<T> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>): KExpr<T> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>): KExpr<T> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>): KExpr<T> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>): KExpr<T> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>): KExpr<KBv1Sort> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>): KExpr<KBv1Sort> {
        return super.transform(expr)
    }

    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> {
        return super.transform(expr)
    }

    override fun transform(expr: KBvExtractExpr): KExpr<KBvSort> {
        return super.transform(expr)
    }

    override fun transform(expr: KBvSignExtensionExpr): KExpr<KBvSort> {
        return super.transform(expr)
    }

    override fun transform(expr: KBvZeroExtensionExpr): KExpr<KBvSort> {
        return super.transform(expr)
    }

    override fun transform(expr: KBvRepeatExpr): KExpr<KBvSort> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>): KExpr<T> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>): KExpr<T> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>): KExpr<T> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>): KExpr<T> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>): KExpr<T> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>): KExpr<T> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>): KExpr<T> {
        return super.transform(expr)
    }

    override fun transform(expr: KBv2IntExpr): KExpr<KIntSort> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>): KExpr<KBoolSort> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>): KExpr<KBoolSort> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>): KExpr<KBoolSort> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>): KExpr<KBoolSort> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>): KExpr<KBoolSort> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>): KExpr<KBoolSort> {
        return super.transform(expr)
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>): KExpr<KBoolSort> {
        return super.transform(expr)
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

    private fun minValueSigned(size: UInt): KBitVecValue<*> = with(ctx) {
        when (size.toInt()) {
            1 -> mkBv(false)
            Byte.SIZE_BITS -> mkBv(Byte.MIN_VALUE)
            Short.SIZE_BITS -> mkBv(Short.MIN_VALUE)
            Int.SIZE_BITS -> mkBv(Int.MIN_VALUE)
            Long.SIZE_BITS -> mkBv(Long.MIN_VALUE)
            else -> {
                val binaryValue = "1" + "0".repeat(size.toInt() - 1)
                mkBv(binaryValue, size)
            }
        }
    }

    private fun maxValueSigned(size: UInt): KBitVecValue<*> = with(ctx) {
        when (size.toInt()) {
            1 -> mkBv(true)
            Byte.SIZE_BITS -> mkBv(Byte.MAX_VALUE)
            Short.SIZE_BITS -> mkBv(Short.MAX_VALUE)
            Int.SIZE_BITS -> mkBv(Int.MAX_VALUE)
            Long.SIZE_BITS -> mkBv(Long.MAX_VALUE)
            else -> {
                val binaryValue = "0" + "1".repeat(size.toInt() - 1)
                mkBv(binaryValue, size)
            }
        }
    }

    private fun maxValueUnsigned(size: UInt): KBitVecValue<*> = with(ctx) {
        when (size.toInt()) {
            1 -> mkBv(true)
            Byte.SIZE_BITS -> mkBv((-1).toByte())
            Short.SIZE_BITS -> mkBv((-1).toShort())
            Int.SIZE_BITS -> mkBv(-1)
            Long.SIZE_BITS -> mkBv(-1L)
            else -> {
                val binaryValue = "1".repeat(size.toInt())
                mkBv(binaryValue, size)
            }
        }
    }

    private fun zero(size: UInt): KBitVecValue<*> = with(ctx) {
        when (size.toInt()) {
            1 -> mkBv(false)
            Byte.SIZE_BITS -> mkBv(0.toByte())
            Short.SIZE_BITS -> mkBv(0.toShort())
            Int.SIZE_BITS -> mkBv(0)
            Long.SIZE_BITS -> mkBv(0L)
            else -> mkBv(0, size)
        }
    }

    private fun one(size: UInt): KBitVecValue<*> = with(ctx) {
        when (size.toInt()) {
            1 -> mkBv(true)
            Byte.SIZE_BITS -> mkBv(1.toByte())
            Short.SIZE_BITS -> mkBv(1.toShort())
            Int.SIZE_BITS -> mkBv(1)
            Long.SIZE_BITS -> mkBv(1L)
            else -> mkBv(1, size)
        }
    }

    private fun KBitVecValue<*>.signedLessOrEqual(other: KBitVecValue<*>): Boolean = when (this) {
        is KBitVec1Value -> value <= (other as KBitVec1Value).value
        is KBitVec8Value -> numberValue <= (other as KBitVec8Value).numberValue
        is KBitVec16Value -> numberValue <= (other as KBitVec16Value).numberValue
        is KBitVec32Value -> numberValue <= (other as KBitVec32Value).numberValue
        is KBitVec64Value -> numberValue <= (other as KBitVec64Value).numberValue
        else -> signedBigIntFromBinary(stringValue) <= signedBigIntFromBinary(other.stringValue)
    }

    private fun KBitVecValue<*>.unsignedLessOrEqual(other: KBitVecValue<*>): Boolean = when (this) {
        is KBitVec1Value -> value <= (other as KBitVec1Value).value
        is KBitVec8Value -> numberValue.toUByte() <= (other as KBitVec8Value).numberValue.toUByte()
        is KBitVec16Value -> numberValue.toUShort() <= (other as KBitVec16Value).numberValue.toUShort()
        is KBitVec32Value -> numberValue.toUInt() <= (other as KBitVec32Value).numberValue.toUInt()
        is KBitVec64Value -> numberValue.toULong() <= (other as KBitVec64Value).numberValue.toULong()
        // MSB first -> lexical order works
        else -> stringValue <= other.stringValue
    }

    private operator fun KBitVecValue<*>.plus(other: KBitVecValue<*>): KBitVecValue<*> = with(ctx) {
        when (this@plus) {
            is KBitVec1Value -> mkBv(value xor (other as KBitVec1Value).value)
            is KBitVec8Value -> bvOperation(other) { a, b -> (a + b).toByte() }
            is KBitVec16Value -> bvOperation(other) { a, b -> (a + b).toShort() }
            is KBitVec32Value -> bvOperation(other) { a, b -> a + b }
            is KBitVec64Value -> bvOperation(other) { a, b -> a + b }
            else -> bvOperationDefault(other) { a, b -> a + b }
        }
    }

    private operator fun KBitVecValue<*>.minus(other: KBitVecValue<*>): KBitVecValue<*> = with(ctx) {
        when (this@minus) {
            is KBitVec1Value -> mkBv(value xor (other as KBitVec1Value).value)
            is KBitVec8Value -> bvOperation(other) { a, b -> (a - b).toByte() }
            is KBitVec16Value -> bvOperation(other) { a, b -> (a - b).toShort() }
            is KBitVec32Value -> bvOperation(other) { a, b -> a - b }
            is KBitVec64Value -> bvOperation(other) { a, b -> a - b }
            else -> bvOperationDefault(other) { a, b -> a - b }
        }
    }

    private operator fun KBitVecValue<*>.times(other: KBitVecValue<*>): KBitVecValue<*> = with(ctx) {
        when (this@times) {
            is KBitVec1Value -> mkBv(value && (other as KBitVec1Value).value)
            is KBitVec8Value -> bvOperation(other) { a, b -> (a * b).toByte() }
            is KBitVec16Value -> bvOperation(other) { a, b -> (a * b).toShort() }
            is KBitVec32Value -> bvOperation(other) { a, b -> a * b }
            is KBitVec64Value -> bvOperation(other) { a, b -> a * b }
            else -> bvOperationDefault(other) { a, b -> a * b }
        }
    }

    private fun KBitVecValue<*>.signedDivide(other: KBitVecValue<*>): KBitVecValue<*> = with(ctx) {
        when (this@signedDivide) {
            is KBitVec1Value -> mkBv(value == (other as KBitVec1Value).value)
            is KBitVec8Value -> bvOperation(other) { a, b -> (a / b).toByte() }
            is KBitVec16Value -> bvOperation(other) { a, b -> (a / b).toShort() }
            is KBitVec32Value -> bvOperation(other) { a, b -> a / b }
            is KBitVec64Value -> bvOperation(other) { a, b -> a / b }
            else -> bvOperationDefault(other, signed = true) { a, b -> a / b }
        }
    }

    private fun KBitVecValue<*>.unsignedDivide(other: KBitVecValue<*>): KBitVecValue<*> = with(ctx) {
        when (this@unsignedDivide) {
            is KBitVec1Value -> mkBv(value == (other as KBitVec1Value).value)
            is KBitVec8Value -> bvUnsignedOperation(other) { a, b -> (a / b).toUByte() }
            is KBitVec16Value -> bvUnsignedOperation(other) { a, b -> (a / b).toUShort() }
            is KBitVec32Value -> bvUnsignedOperation(other) { a, b -> a / b }
            is KBitVec64Value -> bvUnsignedOperation(other) { a, b -> a / b }
            else -> bvOperationDefault(other, signed = false) { a, b -> a / b }
        }
    }

    private fun KBitVecValue<*>.signedRem(other: KBitVecValue<*>): KBitVecValue<*> = with(ctx) {
        when (this@signedRem) {
            is KBitVec1Value -> mkBv(value != (other as KBitVec1Value).value)
            is KBitVec8Value -> bvOperation(other) { a, b -> a.rem(b).toByte() }
            is KBitVec16Value -> bvOperation(other) { a, b -> a.rem(b).toShort() }
            is KBitVec32Value -> bvOperation(other) { a, b -> a.rem(b) }
            is KBitVec64Value -> bvOperation(other) { a, b -> a.rem(b) }
            else -> bvOperationDefault(other, signed = true) { a, b -> a.rem(b) }
        }
    }

    private fun KBitVecValue<*>.unsignedRem(other: KBitVecValue<*>): KBitVecValue<*> = with(ctx) {
        when (this@unsignedRem) {
            is KBitVec1Value -> mkBv(value != (other as KBitVec1Value).value)
            is KBitVec8Value -> bvUnsignedOperation(other) { a, b -> a.rem(b).toUByte() }
            is KBitVec16Value -> bvUnsignedOperation(other) { a, b -> a.rem(b).toUShort() }
            is KBitVec32Value -> bvUnsignedOperation(other) { a, b -> a.rem(b) }
            is KBitVec64Value -> bvUnsignedOperation(other) { a, b -> a.rem(b) }
            else -> bvOperationDefault(other, signed = false) { a, b -> a.rem(b) }
        }
    }

    private fun KBitVecValue<*>.signedMod(other: KBitVecValue<*>): KBitVecValue<*> = with(ctx) {
        when (this@signedMod) {
            is KBitVec1Value -> mkBv(value != (other as KBitVec1Value).value)
            is KBitVec8Value -> bvOperation(other) { a, b -> a.mod(b) }
            is KBitVec16Value -> bvOperation(other) { a, b -> a.mod(b) }
            is KBitVec32Value -> bvOperation(other) { a, b -> a.mod(b) }
            is KBitVec64Value -> bvOperation(other) { a, b -> a.mod(b) }
            else -> bvOperationDefault(other, signed = true) { a, b -> a.mod(b) }
        }
    }

    private inline fun KBitVec8Value.bvOperation(other: KBitVecValue<*>, op: (Byte, Byte) -> Byte) =
        ctx.mkBv(op(numberValue, (other as KBitVec8Value).numberValue))

    private inline fun KBitVec8Value.bvUnsignedOperation(other: KBitVecValue<*>, op: (UByte, UByte) -> UByte) =
        ctx.mkBv(op(numberValue.toUByte(), (other as KBitVec8Value).numberValue.toUByte()).toByte())

    private inline fun KBitVec16Value.bvOperation(other: KBitVecValue<*>, op: (Short, Short) -> Short) =
        ctx.mkBv(op(numberValue, (other as KBitVec16Value).numberValue))

    private inline fun KBitVec16Value.bvUnsignedOperation(other: KBitVecValue<*>, op: (UShort, UShort) -> UShort) =
        ctx.mkBv(op(numberValue.toUShort(), (other as KBitVec16Value).numberValue.toUShort()).toShort())

    private inline fun KBitVec32Value.bvOperation(other: KBitVecValue<*>, op: (Int, Int) -> Int) =
        ctx.mkBv(op(numberValue, (other as KBitVec32Value).numberValue))

    private inline fun KBitVec32Value.bvUnsignedOperation(other: KBitVecValue<*>, op: (UInt, UInt) -> UInt) =
        ctx.mkBv(op(numberValue.toUInt(), (other as KBitVec32Value).numberValue.toUInt()).toInt())

    private inline fun KBitVec64Value.bvOperation(other: KBitVecValue<*>, op: (Long, Long) -> Long) =
        ctx.mkBv(op(numberValue, (other as KBitVec64Value).numberValue))

    private inline fun KBitVec64Value.bvUnsignedOperation(other: KBitVecValue<*>, op: (ULong, ULong) -> ULong) =
        ctx.mkBv(op(numberValue.toULong(), (other as KBitVec64Value).numberValue.toULong()).toLong())

    private fun KBitVecValue<*>.bvOperationDefault(
        rhs: KBitVecValue<*>,
        signed: Boolean = false,
        operation: (BigInteger, BigInteger) -> BigInteger
    ): KBitVecValue<*> = with(ctx) {
        val lhs = this@bvOperationDefault
        val size = lhs.sort.sizeBits
        val lValue = lhs.stringValue.let { if (!signed) unsignedBigIntFromBinary(it) else signedBigIntFromBinary(it) }
        val rValue = rhs.stringValue.let { if (!signed) unsignedBigIntFromBinary(it) else signedBigIntFromBinary(it) }
        val resultValue = operation(lValue, rValue)
        val normalizedValue = resultValue.mod(BigInteger.valueOf(2).pow(size.toInt()))
        val resultBinary = unsignedBinaryString(normalizedValue)
        mkBv(resultBinary, size)
    }

    private fun KBitVecValue<*>.powerOfTwoOrNull(): Int? {
        val value = unsignedBigIntFromBinary(stringValue)
        val valueMinusOne = value - BigInteger.ONE
        if ((value and valueMinusOne) != BigInteger.ZERO) return null
        return valueMinusOne.bitLength()
    }

    private fun signedBigIntFromBinary(value: String): BigInteger {
        var result = BigInteger(value, 2)
        val maxValue = BigInteger.valueOf(2).pow(value.length - 1)
        if (result >= maxValue) {
            result -= BigInteger.valueOf(2).pow(value.length)
        }
        return result
    }

    private fun unsignedBigIntFromBinary(value: String): BigInteger =
        BigInteger(value, 2)

    private fun unsignedBinaryString(value: BigInteger): String =
        value.toString(2)

}
