package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KApp
import org.ksmt.expr.KBitVec16Value
import org.ksmt.expr.KBitVec32Value
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
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExpr
import org.ksmt.expr.KIteExpr
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KIntSort
import org.ksmt.utils.BvUtils.bitwiseNot
import org.ksmt.utils.BvUtils.bitwiseOr
import org.ksmt.utils.BvUtils.bitwiseXor
import org.ksmt.utils.BvUtils.bvMaxValueSigned
import org.ksmt.utils.BvUtils.bvMaxValueUnsigned
import org.ksmt.utils.BvUtils.bvMinValueSigned
import org.ksmt.utils.BvUtils.bvOne
import org.ksmt.utils.BvUtils.bvZero
import org.ksmt.utils.BvUtils.intValueOrNull
import org.ksmt.utils.BvUtils.minus
import org.ksmt.utils.BvUtils.plus
import org.ksmt.utils.BvUtils.powerOfTwoOrNull
import org.ksmt.utils.BvUtils.shiftLeft
import org.ksmt.utils.BvUtils.shiftRightArith
import org.ksmt.utils.BvUtils.shiftRightLogical
import org.ksmt.utils.BvUtils.signedDivide
import org.ksmt.utils.BvUtils.signedGreaterOrEqual
import org.ksmt.utils.BvUtils.signedLessOrEqual
import org.ksmt.utils.BvUtils.signedMod
import org.ksmt.utils.BvUtils.signedRem
import org.ksmt.utils.BvUtils.times
import org.ksmt.utils.BvUtils.toBigIntegerSigned
import org.ksmt.utils.BvUtils.toBigIntegerUnsigned
import org.ksmt.utils.BvUtils.unsignedDivide
import org.ksmt.utils.BvUtils.unsignedLessOrEqual
import org.ksmt.utils.BvUtils.unsignedRem
import org.ksmt.utils.cast
import org.ksmt.utils.uncheckedCast

@Suppress(
    "LargeClass",
    "LongMethod",
    "ComplexMethod"
)
interface KBvExprSimplifier : KExprSimplifierBase {

    fun <T : KBvSort> simplifyEqBv(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> = with(ctx) {
        if (lhs == rhs) return trueExpr

        if (lhs is KBitVecValue<T> && rhs is KBitVecValue<T>) {
            return falseExpr
        }

        if (lhs is KBvConcatExpr || rhs is KBvConcatExpr) {
            return rewrite(simplifyBvConcatEq(lhs, rhs))
        }

        mkEq(lhs, rhs)
    }

    fun <T : KBvSort> areDefinitelyDistinctBv(lhs: KExpr<T>, rhs: KExpr<T>): Boolean {
        if (lhs is KBitVecValue<T> && rhs is KBitVecValue<T>) {
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
        simplifyApp(
            expr = expr,
            preprocess = { mkBvUnsignedLessOrEqualExpr(expr.arg1, expr.arg0) }
        ) {
            error("Always preprocessed")
        }

    // (ult a b) ==> (not (ule b a))
    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>): KExpr<KBoolSort> =
        simplifyApp(
            expr = expr,
            preprocess = { mkNot(mkBvUnsignedLessOrEqualExpr(expr.arg1, expr.arg0)) }
        ) {
            error("Always preprocessed")
        }

    // (ugt a b) ==> (not (ule a b))
    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>): KExpr<KBoolSort> =
        simplifyApp(
            expr = expr,
            preprocess = { mkNot(mkBvUnsignedLessOrEqualExpr(expr.arg0, expr.arg1)) }
        ) {
            error("Always preprocessed")
        }

    // (sge a b) ==> (sle b a)
    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyApp(
            expr = expr,
            preprocess = { mkBvSignedLessOrEqualExpr(expr.arg1, expr.arg0) }
        ) {
            error("Always preprocessed")
        }

    // (slt a b) ==> (not (sle b a))
    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>): KExpr<KBoolSort> =
        simplifyApp(
            expr = expr,
            preprocess = { mkNot(mkBvSignedLessOrEqualExpr(expr.arg1, expr.arg0)) }
        ) {
            error("Always preprocessed")
        }

    // (sgt a b) ==> (not (sle a b))
    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>): KExpr<KBoolSort> =
        simplifyApp(
            expr = expr,
            preprocess = { mkNot(mkBvSignedLessOrEqualExpr(expr.arg0, expr.arg1)) }
        ) {
            error("Always preprocessed")
        }

    @Suppress("NestedBlockDepth")
    private fun <T : KBvSort> bvLessOrEqual(lhs: KExpr<T>, rhs: KExpr<T>, signed: Boolean): KExpr<KBoolSort> =
        with(ctx) {
            if (lhs == rhs) return trueExpr

            val lhsValue = lhs as? KBitVecValue<T>
            val rhsValue = rhs as? KBitVecValue<T>

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
                    bvMinValueSigned(size) to bvMaxValueSigned(size)
                } else {
                    bvZero(size) to bvMaxValueUnsigned(size)
                }

                if (rhsValue != null) {
                    // a <= b, b == MIN_VALUE ==> a == b
                    if (rhsValue == lower) {
                        return rewrite(
                            auxExpr { KEqExpr(ctx, lhs, rhs) }
                        )
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
                        return rewrite(
                            auxExpr { KEqExpr(ctx, lhs, rhs) }
                        )
                    }
                }
            }

            if (signed) {
                mkBvSignedLessOrEqualExpr(lhs, rhs)
            } else {
                mkBvUnsignedLessOrEqualExpr(lhs, rhs)
            }
        }

    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>): KExpr<T> =
        simplifyApp(expr = expr, preprocess = { flatBvAdd(expr) }) {
            error("Always preprocessed")
        }

    private fun <T : KBvSort> transform(expr: SimplifierFlatBvAddExpr<T>): KExpr<T> = simplifyApp(expr) { flatten ->
        val zero = bvZero(expr.sort.sizeBits)
        var constantValue = zero
        val resultParts = arrayListOf<KExpr<T>>()

        for (arg in flatten) {
            if (arg is KBitVecValue<T>) {
                constantValue += arg
                continue
            }
            resultParts += arg
        }

        if (resultParts.isEmpty()) {
            return@simplifyApp constantValue.uncheckedCast()
        }

        if (constantValue != zero) {
            resultParts.add(constantValue.uncheckedCast())
        }

        resultParts.reduceBinaryBvExpr(::mkBvAddExpr)
    }

    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>): KExpr<T> =
        simplifyApp(
            expr = expr,
            preprocess = { mkBvAddExpr(expr.arg0, mkBvNegationExpr(expr.arg1)) }
        ) {
            error("Always preprocessed")
        }

    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>): KExpr<T> =
        simplifyApp(expr = expr, preprocess = { flatBvMul(expr) }) {
            error("Always preprocessed")
        }

    private fun <T : KBvSort> transform(expr: SimplifierFlatBvMulExpr<T>): KExpr<T> = simplifyApp(expr) { flatten ->
        val zero = bvZero(expr.sort.sizeBits)
        val one = bvOne(expr.sort.sizeBits)

        var constantValue = one
        val resultParts = arrayListOf<KExpr<T>>()

        for (arg in flatten) {
            if (arg is KBitVecValue<T>) {
                constantValue *= arg
                continue
            }
            resultParts += arg
        }

        // (* 0 a) ==> 0
        if (constantValue == zero) {
            return@simplifyApp zero.uncheckedCast()
        }

        if (resultParts.isEmpty()) {
            return@simplifyApp constantValue.uncheckedCast()
        }

        // (* 1 a) ==> a
        if (constantValue == one) {
            return@simplifyApp resultParts.reduceBinaryBvExpr(::mkBvMulExpr)
        }

        // (* -1 a) ==> -a
        val minusOne = zero - one
        if (constantValue == minusOne) {
            val value = resultParts.reduceBinaryBvExpr(::mkBvMulExpr)
            return@simplifyApp mkBvNegationExpr(value)
        }

        resultParts.add(constantValue.uncheckedCast())
        resultParts.reduceBinaryBvExpr(::mkBvMulExpr)
    }

    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>): KExpr<T> = simplifyApp(expr) { (arg) ->
        if (arg is KBitVecValue<T>) {
            return@simplifyApp (bvZero(expr.sort.sizeBits) - arg).uncheckedCast()
        }
        mkBvNegationExpr(arg)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val size = expr.sort.sizeBits
        val lhsValue = lhs as? KBitVecValue<T>
        val rhsValue = rhs as? KBitVecValue<T>

        if (rhsValue != null) {
            // ignore zero
            if (rhsValue == bvZero(size)) {
                return@simplifyApp mkBvSignedDivExpr(lhs, rhs)
            }

            if (rhsValue == bvOne(size)) {
                return@simplifyApp lhs
            }

            if (lhsValue != null) {
                return@simplifyApp lhsValue.signedDivide(rhsValue).uncheckedCast()
            }
        }

        mkBvSignedDivExpr(lhs, rhs)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val size = expr.sort.sizeBits
        val lhsValue = lhs as? KBitVecValue<T>
        val rhsValue = rhs as? KBitVecValue<T>

        if (rhsValue != null) {
            // ignore zero
            if (rhsValue == bvZero(size)) {
                return@simplifyApp mkBvUnsignedDivExpr(lhs, rhs)
            }

            if (rhsValue == bvOne(size)) {
                return@simplifyApp lhs
            }

            if (lhsValue != null) {
                return@simplifyApp lhsValue.unsignedDivide(rhsValue).uncheckedCast()
            }

            rhsValue.powerOfTwoOrNull()?.let { shift ->
                return@simplifyApp rewrite(
                    auxExpr { KBvLogicalShiftRightExpr(ctx, lhs, mkBv(shift, size).uncheckedCast()) }
                )
            }
        }

        mkBvUnsignedDivExpr(lhs, rhs)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val size = expr.sort.sizeBits
        val lhsValue = lhs as? KBitVecValue<T>
        val rhsValue = rhs as? KBitVecValue<T>

        if (rhsValue != null) {
            // ignore zero
            if (rhsValue == bvZero(size)) {
                return@simplifyApp mkBvSignedRemExpr(lhs, rhs)
            }

            if (rhsValue == bvOne(size)) {
                return@simplifyApp bvZero(size).uncheckedCast()
            }

            if (lhsValue != null) {
                return@simplifyApp lhsValue.signedRem(rhsValue).uncheckedCast()
            }
        }

        mkBvSignedRemExpr(lhs, rhs)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val size = expr.sort.sizeBits
        val lhsValue = lhs as? KBitVecValue<T>
        val rhsValue = rhs as? KBitVecValue<T>

        if (rhsValue != null) {
            // ignore zero
            if (rhsValue == bvZero(size)) {
                return@simplifyApp mkBvUnsignedRemExpr(lhs, rhs)
            }

            if (rhsValue == bvOne(size)) {
                return@simplifyApp bvZero(size).uncheckedCast()
            }

            if (lhsValue != null) {
                return@simplifyApp lhsValue.unsignedRem(rhsValue).uncheckedCast()
            }

            rhsValue.powerOfTwoOrNull()?.let { shift ->
                return@simplifyApp rewrite(
                    auxExpr {
                        KBvConcatExpr(
                            ctx,
                            bvZero(size - shift.toUInt()).uncheckedCast(),
                            KBvExtractExpr(ctx, shift - 1, 0, lhs.uncheckedCast())
                        ).uncheckedCast()
                    }
                )
            }
        }

        mkBvUnsignedRemExpr(lhs, rhs)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val size = expr.sort.sizeBits
        val lhsValue = lhs as? KBitVecValue<T>
        val rhsValue = rhs as? KBitVecValue<T>

        if (rhsValue != null) {
            // ignore zero
            if (rhsValue == bvZero(size)) {
                return@simplifyApp mkBvSignedModExpr(lhs, rhs)
            }

            if (rhsValue == bvOne(size)) {
                return@simplifyApp bvZero(size).uncheckedCast()
            }

            if (lhsValue != null) {
                return@simplifyApp lhsValue.signedMod(rhsValue).uncheckedCast()
            }
        }

        mkBvSignedModExpr(lhs, rhs)
    }

    override fun <T : KBvSort> transform(expr: KBvNotExpr<T>): KExpr<T> = simplifyApp(expr) { (arg) ->
        // (bvnot (bvnot a)) ==> a
        if (arg is KBvNotExpr<T>) {
            return@simplifyApp arg.value
        }

        if (arg is KBitVecValue<T>) {
            return@simplifyApp arg.bitwiseNot().uncheckedCast()
        }

        // (bvnot (concat a b)) ==> (concat (bvnot a) (bvnot b))
        if (arg is KBvConcatExpr) {
            val concatParts = flatConcat(arg).args
            return@simplifyApp rewrite(
                auxExpr {
                    val negatedParts = concatParts.map { KBvNotExpr(ctx, it) }
                    SimplifierFlatBvConcatExpr(ctx, arg.sort, negatedParts).uncheckedCast()
                }
            )
        }

        // (bvnot (ite c a b)) ==> (ite c (bvnot a) (bvnot b))
        if (arg is KIteExpr<T>) {
            val trueValue = arg.trueBranch as? KBitVecValue<T>
            val falseValue = arg.falseBranch as? KBitVecValue<T>
            if (trueValue != null || falseValue != null) {
                return@simplifyApp rewrite(
                    mkIte(
                        condition = arg.condition,
                        trueBranch = mkBvNotExpr(arg.trueBranch),
                        falseBranch = mkBvNotExpr(arg.falseBranch)
                    )
                )
            }
        }

        mkBvNotExpr(arg)
    }

    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>): KExpr<T> =
        simplifyApp(expr = expr, preprocess = { flatBvOr(expr) }) {
            error("Always preprocessed")
        }

    @Suppress("LoopWithTooManyJumpStatements")
    private fun <T : KBvSort> transform(expr: SimplifierFlatBvOrExpr<T>): KExpr<T> = simplifyApp(expr) { flatten ->
        val zero = bvZero(expr.sort.sizeBits)
        val maxValue = bvMaxValueUnsigned(expr.sort.sizeBits)
        var constantValue = zero
        val resultParts = arrayListOf<KExpr<T>>()
        val positiveTerms = hashSetOf<KExpr<T>>()
        val negativeTerms = hashSetOf<KExpr<T>>()

        for (arg in flatten) {
            if (arg is KBitVecValue<T>) {
                constantValue = constantValue.bitwiseOr(arg)
                continue
            }

            if (arg is KBvNotExpr<T>) {
                val term = arg.value
                // (bvor (bvnot a) b (bvnot a)) ==> (bvor (bvnot a) b)
                if (!negativeTerms.add(term)) {
                    continue
                }

                // (bvor a (bvnot a)) ==> 0xFFFF...
                if (term in positiveTerms) {
                    return@simplifyApp maxValue.uncheckedCast()
                }
            } else {
                // (bvor a b a) ==> (bvor a b)
                if (!positiveTerms.add(arg)) {
                    continue
                }

                // (bvor a (bvnot a)) ==> 0xFFFF...
                if (arg in negativeTerms) {
                    return@simplifyApp maxValue.uncheckedCast()
                }
            }

            resultParts += arg
        }

        // (bvor 0xFFFF... a) ==> 0xFFFF...
        if (constantValue == maxValue) {
            return@simplifyApp maxValue.uncheckedCast()
        }

        if (resultParts.isEmpty()) {
            return@simplifyApp constantValue.uncheckedCast()
        }

        // (bvor 0 a) ==> a
        if (constantValue != zero) {
            resultParts.add(constantValue.uncheckedCast())
        }

        /**
         * (bvor (concat a b) c) ==>
         *  (concat
         *      (bvor (extract (0, <a_size>) c))
         *      (bvor b (extract (<a_size>, <a_size> + <b_size>) c))
         *  )
         * */
        if (resultParts.any { it is KBvConcatExpr }) {
            return@simplifyApp rewrite(distributeOrOverConcat(resultParts))
        }

        resultParts.reduceBinaryBvExpr(::mkBvOrExpr)
    }

    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>): KExpr<T> =
        simplifyApp(expr = expr, preprocess = { flatBvXor(expr) }) {
            error("Always preprocessed")
        }

    private fun <T : KBvSort> transform(expr: SimplifierFlatBvXorExpr<T>): KExpr<T> = simplifyApp(expr) { flatten ->
        val zero = bvZero(expr.sort.sizeBits)
        val maxValue = bvMaxValueUnsigned(expr.sort.sizeBits)
        var constantValue = zero

        val positiveParts = mutableSetOf<KExpr<T>>()
        val negativeParts = mutableSetOf<KExpr<T>>()

        for (arg in flatten) {
            if (arg is KBitVecValue<T>) {
                constantValue = constantValue.bitwiseXor(arg)
                continue
            }

            if (arg is KBvNotExpr<T>) {
                when (val term = arg.value) {
                    in negativeParts -> {
                        // (bxor (bvnot a) b (bvnot a)) ==> (bvxor 0 b)
                        negativeParts.remove(term)
                    }
                    in positiveParts -> {
                        // (bvxor a b (bvnot a)) ==> (bvxor b 0xFFFF...)
                        positiveParts.remove(term)
                        constantValue = constantValue.bitwiseXor(maxValue)
                    }
                    else -> {
                        negativeParts.add(term)
                    }
                }
            } else {
                when (arg) {
                    in positiveParts -> {
                        // (bvxor a b a) ==> (bvxor 0 b)
                        positiveParts.remove(arg)
                    }
                    in negativeParts -> {
                        // (bvxor (bvnot a) b a) ==> (bvxor b 0xFFFF...)
                        negativeParts.remove(arg)
                        constantValue = constantValue.bitwiseXor(maxValue)
                    }
                    else -> {
                        positiveParts.add(arg)
                    }
                }
            }
        }

        val resultParts = arrayListOf<KExpr<T>>().apply {
            addAll(positiveParts)
            addAll(negativeParts.map { mkBvNotExpr(it) })
        }

        if (resultParts.isEmpty()) {
            return@simplifyApp constantValue.uncheckedCast()
        }

        var negateResult = false
        when (constantValue) {
            zero -> {
                // (bvxor 0 a) ==> a
            }
            maxValue -> {
                // (bvxor 0xFFFF... a) ==> (bvnot a)
                negateResult = true
            }
            else -> {
                resultParts.add(constantValue.uncheckedCast())
            }
        }

        if (resultParts.any { it is KBvConcatExpr }) {
            val preResult = distributeXorOverConcat(resultParts)
            val result = if (negateResult) {
                auxExpr {
                    KBvNotExpr(ctx, preResult.expr)
                }
            } else {
                preResult
            }
            return@simplifyApp rewrite(result)
        }

        val preResult = resultParts.reduceBinaryBvExpr(::mkBvXorExpr)
        return@simplifyApp if (negateResult) mkBvNotExpr(preResult) else preResult
    }

    // (bvand a b) ==> (bvnot (bvor (bvnot a) (bvnot b)))
    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>): KExpr<T> =
        simplifyApp(
            expr = expr,
            preprocess = { mkBvNotExpr(mkBvOrExpr(mkBvNotExpr(expr.arg0), mkBvNotExpr(expr.arg1))) }
        ) {
            error("Always preprocessed")
        }

    // (bvnand a b) ==> (bvor (bvnot a) (bvnot b))
    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>): KExpr<T> =
        simplifyApp(
            expr = expr,
            preprocess = { mkBvOrExpr(mkBvNotExpr(expr.arg0), mkBvNotExpr(expr.arg1)) }
        ) {
            error("Always preprocessed")
        }

    // (bvnor a b) ==> (bvnot (bvor a b))
    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>): KExpr<T> =
        simplifyApp(
            expr = expr,
            preprocess = { mkBvNotExpr(mkBvOrExpr(expr.arg0, expr.arg1)) }
        ) {
            error("Always preprocessed")
        }

    // (bvxnor a b) ==> (bvnot (bvxor a b))
    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>): KExpr<T> =
        simplifyApp(
            expr = expr,
            preprocess = { mkBvNotExpr(mkBvXorExpr(expr.arg0, expr.arg1)) }
        ) {
            error("Always preprocessed")
        }

    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>): KExpr<KBv1Sort> = simplifyApp(expr) { (arg) ->
        if (arg is KBitVecValue<T>) {
            val result = arg == bvMaxValueUnsigned(arg.sort.sizeBits)
            return@simplifyApp mkBv(result)
        }
        mkBvReductionAndExpr(arg)
    }

    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>): KExpr<KBv1Sort> = simplifyApp(expr) { (arg) ->
        if (arg is KBitVecValue<T>) {
            val result = arg != bvZero(arg.sort.sizeBits)
            return@simplifyApp mkBv(result)
        }
        mkBvReductionOrExpr(arg)
    }

    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> =
        simplifyApp(expr = expr, preprocess = { flatConcat(expr) }) {
            error("Always preprocessed")
        }

    @Suppress("LoopWithTooManyJumpStatements")
    private fun transform(expr: SimplifierFlatBvConcatExpr): KExpr<KBvSort> = simplifyApp(expr) { flatten ->
        val mergedParts = arrayListOf(flatten.first())

        for (part in flatten.drop(1)) {
            val lastPart = mergedParts.last()

            // (concat (concat a const1) (concat const2 b)) ==> (concat a (concat (concat const1 const2) b))
            if (lastPart is KBitVecValue<*> && part is KBitVecValue<*>) {
                mergedParts.removeLast()
                mergedParts.add(concatBv(lastPart, part).cast())
                continue
            }

            // (concat (extract[h1, l1] a) (extract[h2, l2] a)), l1 == h2 + 1 ==> (extract[h1, l2] a)
            if (lastPart is KBvExtractExpr && part is KBvExtractExpr) {
                val possiblyMerged = tryMergeBvConcatExtract(lastPart, part)
                if (possiblyMerged != null) {
                    mergedParts.removeLast()
                    mergedParts.add(possiblyMerged)
                    continue
                }
            }
            mergedParts.add(part)
        }

        if (mergedParts.size < flatten.size) {
            rewrite(SimplifierFlatBvConcatExpr(ctx, expr.sort, mergedParts))
        } else {
            mergedParts.reduceBinaryBvExpr(::mkBvConcatExpr)
        }
    }

    override fun transform(expr: KBvExtractExpr): KExpr<KBvSort> = simplifyApp(expr) { (arg) ->
        // (extract [size-1:0] x) ==> x
        if (expr.low == 0 && expr.high == arg.sort.sizeBits.toInt() - 1) {
            return@simplifyApp arg
        }

        if (arg is KBitVecValue<*>) {
            return@simplifyApp extractBv(arg, expr.high, expr.low).uncheckedCast()
        }

        // (extract[high:low] (extract[_:nestedLow] x)) ==> (extract[high+nestedLow : low+nestedLow] x)
        if (arg is KBvExtractExpr) {
            val nestedLow = arg.low
            return@simplifyApp rewrite(
                auxExpr {
                    KBvExtractExpr(
                        ctx,
                        high = expr.high + nestedLow,
                        low = expr.low + nestedLow,
                        value = arg.value
                    )
                }
            )
        }

        // (extract (concat a b)) ==> (concat (extract a) (extract b))
        if (arg is KBvConcatExpr) {
            return@simplifyApp rewrite(distributeExtractOverConcat(arg, expr.high, expr.low))
        }

        val simplified = when {
            // (extract [h:l] (bvnot x)) ==> (bvnot (extract [h:l] x))
            arg is KBvNotExpr<*> -> mkBvNotExpr(mkBvExtractExpr(expr.high, expr.low, arg.value))
            // (extract [h:l] (bvor a b)) ==> (bvor (extract [h:l] a) (extract [h:l] b))
            arg is KBvOrExpr<*> -> {
                val lhs = mkBvExtractExpr(expr.high, expr.low, arg.arg0)
                val rhs = mkBvExtractExpr(expr.high, expr.low, arg.arg1)
                mkBvOrExpr(lhs, rhs)
            }
            // (extract [h:l] (bvxor a b)) ==> (bvxor (extract [h:l] a) (extract [h:l] b))
            arg is KBvXorExpr<*> -> {
                val lhs = mkBvExtractExpr(expr.high, expr.low, arg.arg0)
                val rhs = mkBvExtractExpr(expr.high, expr.low, arg.arg1)
                mkBvXorExpr(lhs, rhs)
            }
            // (extract [h:0] (bvadd a b)) ==> (bvadd (extract [h:0] a) (extract [h:0] b))
            arg is KBvAddExpr<*> && expr.low == 0 -> {
                val lhs = mkBvExtractExpr(expr.high, low = 0, arg.arg0)
                val rhs = mkBvExtractExpr(expr.high, low = 0, arg.arg1)
                mkBvAddExpr(lhs, rhs)
            }
            // (extract [h:0] (bvmul a b)) ==> (bvmul (extract [h:0] a) (extract [h:0] b))
            arg is KBvMulExpr<*> && expr.low == 0 -> {
                val lhs = mkBvExtractExpr(expr.high, low = 0, arg.arg0)
                val rhs = mkBvExtractExpr(expr.high, low = 0, arg.arg1)
                mkBvMulExpr(lhs, rhs)
            }
            else -> null
        }

        if (simplified != null) {
            return@simplifyApp rewrite(simplified)
        }

        mkBvExtractExpr(expr.high, expr.low, arg)
    }

    /**
     * (extract (concat a b)) ==> (concat (extract a) (extract b))
     * */
    @Suppress("LoopWithTooManyJumpStatements", "NestedBlockDepth")
    private fun distributeExtractOverConcat(
        concatenation: KBvConcatExpr,
        high: Int,
        low: Int
    ): SimplifierAuxExpression<KBvSort> = auxExpr {
        val parts = flatConcat(concatenation).args

        var idx = concatenation.sort.sizeBits.toInt()
        var firstPartIdx = 0

        // find first part to extract from
        do {
            val firstPart = parts[firstPartIdx]
            val firstPartSize = firstPart.sort.sizeBits.toInt()
            idx -= firstPartSize

            // before first part
            if (idx > high) {
                firstPartIdx++
                continue
            }

            // extract from a single part
            if (idx <= low) {
                return@auxExpr if (idx == low && high - idx == firstPartSize) {
                    firstPart
                } else {
                    KBvExtractExpr(
                        ctx,
                        high = high - idx,
                        low = low - idx,
                        value = firstPart
                    )
                }
            }

            /**
             * idx <= high && idx > low
             * extract from multiple parts starting from firstPartIdx
             * */
            break

        } while (firstPartIdx < parts.size)


        // extract from multiple parts
        val partsToExtractFrom = arrayListOf<KExpr<KBvSort>>()
        val firstPart = parts[firstPartIdx]
        val firstPartSize = firstPart.sort.sizeBits.toInt()

        if (high - idx == firstPartSize - 1) {
            partsToExtractFrom += firstPart
        } else {
            partsToExtractFrom += KBvExtractExpr(
                ctx,
                high = high - idx,
                low = 0,
                value = firstPart
            )
        }

        for (partIdx in firstPartIdx + 1 until parts.size) {
            val part = parts[partIdx]
            val partSize = part.sort.sizeBits.toInt()
            idx -= partSize

            when {
                idx > low -> {
                    // not a last part
                    partsToExtractFrom += part
                    continue
                }

                idx == low -> {
                    partsToExtractFrom += part
                    break
                }

                else -> {
                    partsToExtractFrom += KBvExtractExpr(
                        ctx,
                        high = partSize - 1,
                        low = low - idx,
                        value = part
                    )
                    break
                }
            }
        }

        SimplifierFlatBvConcatExpr(ctx, concatenation.sort, partsToExtractFrom)
    }

    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>): KExpr<T> = simplifyApp(expr) { (arg, shift) ->
        val size = expr.sort.sizeBits
        val argValue = arg as? KBitVecValue<T>
        val shiftValue = shift as? KBitVecValue<T>

        if (shiftValue != null) {
            // (x << 0) ==> x
            if (shiftValue == bvZero(size)) {
                return@simplifyApp arg
            }

            // (x << shift), shift >= size ==> 0
            if (shiftValue.signedGreaterOrEqual(size.toInt())) {
                return@simplifyApp bvZero(size).uncheckedCast()
            }

            if (argValue != null) {
                return@simplifyApp argValue.shiftLeft(shiftValue).uncheckedCast()
            }

            // (bvshl x shift) ==> (concat (extract [size-1-shift:0] x) 0.[shift].0)
            val intShiftValue = shiftValue.intValueOrNull()
            if (intShiftValue != null && intShiftValue >= 0) {
                return@simplifyApp rewrite(
                    auxExpr {
                        KBvConcatExpr(
                            ctx,
                            KBvExtractExpr(ctx, high = size.toInt() - 1 - intShiftValue, low = 0, arg.uncheckedCast()),
                            bvZero(intShiftValue.toUInt()).uncheckedCast()
                        ).uncheckedCast()
                    }
                )
            }
        }

        /**
         * (bvshl (bvshl x nestedShift) shift) ==>
         *      (ite (bvule nestedShift (+ nestedShift shift)) (bvshl x (+ nestedShift shift)) 0)
         * */
        if (arg is KBvShiftLeftExpr<T>) {
            val nestedArg = arg.arg
            val nestedShift = arg.shift
            val sum = mkBvAddExpr(nestedShift, shift)
            val cond = mkBvUnsignedLessOrEqualExpr(nestedShift, sum)
            return@simplifyApp rewrite(
                mkIte(
                    condition = cond,
                    trueBranch = mkBvShiftLeftExpr(nestedArg, sum),
                    falseBranch = bvZero(size).uncheckedCast()
                )
            )
        }

        mkBvShiftLeftExpr(arg, shift)
    }

    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>): KExpr<T> =
        simplifyApp(expr) { (arg, shift) ->
            val size = expr.sort.sizeBits
            val argValue = arg as? KBitVecValue<T>
            val shiftValue = shift as? KBitVecValue<T>

            if (shiftValue != null) {
                // (x >>> 0) ==> x
                if (shiftValue == bvZero(size)) {
                    return@simplifyApp arg
                }

                // (x >>> shift), shift >= size ==> 0
                if (shiftValue.signedGreaterOrEqual(size.toInt())) {
                    return@simplifyApp bvZero(size).uncheckedCast()
                }

                if (argValue != null) {
                    return@simplifyApp argValue.shiftRightLogical(shiftValue).uncheckedCast()
                }

                // (bvlshr x shift) ==> (concat 0.[shift].0 (extract [size-1:shift] x))
                val intShiftValue = shiftValue.intValueOrNull()
                if (intShiftValue != null && intShiftValue >= 0) {
                    return@simplifyApp rewrite(
                        auxExpr {
                            val lhs = bvZero(intShiftValue.toUInt())
                            val rhs = KBvExtractExpr(
                                ctx,
                                high = size.toInt() - 1,
                                low = intShiftValue,
                                arg.uncheckedCast()
                            )
                            KBvConcatExpr(ctx, lhs.uncheckedCast(), rhs).uncheckedCast()
                        }
                    )
                }
            }

            // (x >>> x) ==> 0
            if (arg == shift) {
                return@simplifyApp bvZero(size).uncheckedCast()
            }

            mkBvLogicalShiftRightExpr(arg, shift)
        }

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>): KExpr<T> =
        simplifyApp(expr) { (arg, shift) ->
            val size = expr.sort.sizeBits
            val argValue = arg as? KBitVecValue<T>
            val shiftValue = shift as? KBitVecValue<T>

            if (shiftValue != null) {
                // (x >> 0) ==> x
                if (shiftValue == bvZero(size)) {
                    return@simplifyApp arg
                }

                if (argValue != null) {
                    return@simplifyApp argValue.shiftRightArith(shiftValue).uncheckedCast()
                }
            }

            mkBvArithShiftRightExpr(arg, shift)
        }

    // (repeat a x) ==> (concat a a ..[x].. a)
    override fun transform(expr: KBvRepeatExpr): KExpr<KBvSort> = simplifyApp(expr) { (arg) ->
        val repeats = arrayListOf<KExpr<KBvSort>>()
        repeat(expr.repeatNumber) {
            repeats += arg
        }

        if (repeats.size == 0) {
            return@simplifyApp mkBvRepeatExpr(expr.repeatNumber, arg)
        }

        return@simplifyApp rewrite(SimplifierFlatBvConcatExpr(ctx, expr.sort, repeats))
    }

    // (zeroext a) ==> (concat 0 a)
    override fun transform(expr: KBvZeroExtensionExpr): KExpr<KBvSort> = simplifyApp(expr) { (arg) ->
        if (expr.extensionSize == 0) {
            return@simplifyApp arg
        }

        val extension = bvZero(expr.extensionSize.toUInt())
        return@simplifyApp rewrite(
            auxExpr {
                KBvConcatExpr(ctx, extension.uncheckedCast(), arg)
            }
        )
    }

    override fun transform(expr: KBvSignExtensionExpr): KExpr<KBvSort> = simplifyApp(expr) { (arg) ->
        if (expr.extensionSize == 0) {
            return@simplifyApp arg
        }

        if (arg is KBitVecValue<*>) {
            return@simplifyApp arg.signExtension(expr.extensionSize).uncheckedCast()
        }

        return@simplifyApp mkBvSignExtensionExpr(expr.extensionSize, arg)
    }

    // (rotateLeft a x) ==> (concat (extract [size-1-x:0] a) (extract [size-1:size-x] a))
    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>): KExpr<T> = simplifyApp(expr) { (arg) ->
        val size = expr.sort.sizeBits.toInt()
        return@simplifyApp rotateLeft(arg, size, expr.rotationNumber)
    }

    // (rotateRight a x) ==> (rotateLeft a (- size x))
    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>): KExpr<T> = simplifyApp(expr) { (arg) ->
        val size = expr.sort.sizeBits.toInt()
        val rotation = expr.rotationNumber % size
        return@simplifyApp rotateLeft(arg, size, rotationNumber = size - rotation)
    }

    fun <T : KBvSort> rotateLeft(arg: KExpr<T>, size: Int, rotationNumber: Int): KExpr<T> = with(ctx) {
        val rotation = rotationNumber % size

        if (rotation == 0 || size == 1) {
            return arg
        }

        return rewrite(
            auxExpr {
                val lhs = KBvExtractExpr(ctx, high = size - rotation - 1, low = 0, arg.uncheckedCast())
                val rhs = KBvExtractExpr(ctx, high = size - 1, low = size - rotation, arg.uncheckedCast())
                KBvConcatExpr(ctx, lhs, rhs).uncheckedCast()
            }
        )
    }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>): KExpr<T> = simplifyApp(expr) { (arg, rotation) ->
        if (rotation is KBitVecValue<T>) {
            val intValue = rotation.intValueOrNull()
            if (intValue != null && intValue >= 0) {
                return@simplifyApp rewrite(
                    auxExpr {
                        KBvRotateLeftIndexedExpr(ctx, intValue, arg)
                    }
                )
            }
        }
        return@simplifyApp mkBvRotateLeftExpr(arg, rotation)
    }

    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>): KExpr<T> = simplifyApp(expr) { (arg, rotation) ->
        if (rotation is KBitVecValue<T>) {
            val intValue = rotation.intValueOrNull()
            if (intValue != null && intValue >= 0) {
                return@simplifyApp rewrite(
                    auxExpr {
                        KBvRotateRightIndexedExpr(ctx, intValue, arg)
                    }
                )
            }
        }
        return@simplifyApp mkBvRotateRightExpr(arg, rotation)
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            if (expr.isSigned) {
                /**
                 * (bvadd no ovf signed a b) ==>
                 *    (=> (and (bvslt 0 a) (bvslt 0 b)) (bvslt 0 (bvadd a b)))
                 * */

                val zero: KExpr<T> = bvZero(lhs.sort.sizeBits).uncheckedCast()
                val zeroSltA = mkBvSignedLessExpr(zero, lhs)
                val zeroSltB = mkBvSignedLessExpr(zero, rhs)
                val sum = mkBvAddExpr(lhs, rhs)
                val zeroSltSum = mkBvSignedLessExpr(zero, sum)

                return@simplifyApp rewrite(mkImplies(zeroSltA and zeroSltB, zeroSltSum))
            } else {
                /**
                 * (bvadd no ovf unsigned a b) ==>
                 *    (= 0 (extract [highestBit] (bvadd (concat 0 a) (concat 0 b))))
                 * */

                val zeroBit = mkBv(false)
                val extA = mkBvConcatExpr(zeroBit, lhs)
                val extB = mkBvConcatExpr(zeroBit, rhs)
                val sum = mkBvAddExpr(extA, extB)
                val highestBitIdx = sum.sort.sizeBits.toInt() - 1
                val sumFirstBit = mkBvExtractExpr(highestBitIdx, highestBitIdx, sum)

                return@simplifyApp rewrite(zeroBit eq sumFirstBit.uncheckedCast())
            }
        }

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            /**
             * (bvadd no udf a b) ==>
             *    (=> (and (bvslt a 0) (bvslt b 0)) (bvslt (bvadd a b) 0))
             * */

            val zero: KExpr<T> = bvZero(lhs.sort.sizeBits).uncheckedCast()
            val aLtZero = mkBvSignedLessExpr(lhs, zero)
            val bLtZero = mkBvSignedLessExpr(rhs, zero)
            val sum = mkBvAddExpr(lhs, rhs)
            val sumLtZero = mkBvSignedLessExpr(sum, zero)

            return@simplifyApp rewrite(mkImplies(aLtZero and bLtZero, sumLtZero))
        }

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            /**
             * (bvsub no ovf a b) ==>
             *     (ite (= b MIN_VALUE) (bvslt a 0) (bvadd no ovf signed a (bvneg b)))
             * */

            val zero: KExpr<T> = bvZero(lhs.sort.sizeBits).uncheckedCast()
            val minValue: KExpr<T> = bvMinValueSigned(lhs.sort.sizeBits).uncheckedCast()

            val minusB = mkBvNegationExpr(rhs)
            val bIsMin = rhs eq minValue
            val aLtZero = mkBvSignedLessExpr(lhs, zero)
            val noOverflow = mkBvAddNoOverflowExpr(lhs, minusB, isSigned = true)

            return@simplifyApp rewrite(mkIte(bIsMin, aLtZero, noOverflow))
        }

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            if (expr.isSigned) {
                /**
                 * (bvsub no udf signed a b) ==>
                 *    (=> (bvslt 0 b) (bvadd no udf (bvneg b)))
                 * */
                val zero: KExpr<T> = bvZero(lhs.sort.sizeBits).uncheckedCast()
                val minusB = mkBvNegationExpr(rhs)
                val zeroLtB = mkBvSignedLessExpr(zero, rhs)
                val noOverflow = mkBvAddNoUnderflowExpr(lhs, minusB)

                return@simplifyApp rewrite(mkImplies(zeroLtB, noOverflow))
            } else {
                /**
                 * (bvsub no udf unsigned a b) ==>
                 *    (bvule b a)
                 * */
                return@simplifyApp rewrite(mkBvUnsignedLessOrEqualExpr(rhs, lhs))
            }
        }

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            /**
             * (bvneg no ovf a) ==> (not (= a MIN_VALUE))
             * */
            val minValue = bvMinValueSigned(arg.sort.sizeBits)
            return@simplifyApp rewrite(!(arg eq minValue.uncheckedCast()))
        }

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            /**
             * (bvsdiv no ovf a b) ==>
             *     (not (and (= a MSB) (= b -1)))
             * */
            val size = lhs.sort.sizeBits
            val mostSignificantBit = if (size == 1u) {
                mkBv(true)
            } else {
                mkBvConcatExpr(mkBv(true), bvZero(size - 1u))
            }
            val minusOne = bvZero(size) - bvOne(size)

            val aIsMsb = lhs eq mostSignificantBit.uncheckedCast()
            val bIsMinusOne = rhs eq minusOne.uncheckedCast()
            return@simplifyApp rewrite(!(aIsMsb and bIsMinusOne))
        }

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            val simplified = if (expr.isSigned) {
                trySimplifyBvSignedMulNoOverflow(lhs, rhs, isOverflow = true)
            } else {
                trySimplifyBvUnsignedMulNoOverflow(lhs, rhs)
            }
            simplified ?: mkBvMulNoOverflowExpr(lhs, rhs, expr.isSigned)
        }

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            trySimplifyBvSignedMulNoOverflow(lhs, rhs, isOverflow = false) ?: mkBvMulNoUnderflowExpr(lhs, rhs)
        }

    @Suppress("NestedBlockDepth", "ComplexCondition")
    private fun <T : KBvSort> trySimplifyBvSignedMulNoOverflow(
        lhs: KExpr<T>,
        rhs: KExpr<T>,
        isOverflow: Boolean
    ): KExpr<KBoolSort>? = with(ctx) {
        val size = lhs.sort.sizeBits
        val lhsValue = lhs as? KBitVecValue<T>
        val rhsValue = rhs as? KBitVecValue<T>
        val zero = bvZero(size)
        val one = bvOne(size)

        if (lhsValue != null && (lhsValue == zero || (size != 1u && lhsValue == one))) {
            return trueExpr
        }

        if (rhsValue != null && (rhsValue == zero || (size != 1u && rhsValue == one))) {
            return trueExpr
        }

        if (lhsValue != null && rhsValue != null) {
            val lhsSign = lhsValue.stringValue[0] == '1'
            val rhsSign = rhsValue.stringValue[0] == '1'

            when {
                // lhs < 0 && rhs < 0
                lhsSign && rhsSign -> {
                    // no underflow possible
                    if (!isOverflow) return trueExpr
                    // overflow if rhs <= (MAX_VALUE / lhs - 1)
                    val maxValue = bvMaxValueSigned(size)
                    val limit = maxValue.signedDivide(lhsValue)
                    return rhsValue.signedLessOrEqual(limit - one).expr
                }
                // lhs > 0 && rhs > 0
                !lhsSign && !rhsSign -> {
                    // no underflow possible
                    if (!isOverflow) return trueExpr
                    // overflow if MAX_VALUE / rhs <= lhs - 1
                    val maxValue = bvMaxValueSigned(size)
                    val limit = maxValue.signedDivide(rhsValue)
                    return limit.signedLessOrEqual(lhsValue - one).expr
                }
                // lhs < 0 && rhs > 0
                lhsSign && !rhsSign -> {
                    // no overflow possible
                    if (isOverflow) return trueExpr
                    // underflow if lhs <= MIN_VALUE / rhs - 1
                    val minValue = bvMinValueSigned(size)
                    val limit = minValue.signedDivide(rhsValue)
                    return lhsValue.signedLessOrEqual(limit - one).expr
                }
                // lhs > 0 && rhs < 0
                else -> {
                    // no overflow possible
                    if (isOverflow) return trueExpr
                    // underflow if rhs <= MIN_VALUE / lhs - 1
                    val minValue = bvMinValueSigned(size)
                    val limit = minValue.signedDivide(lhsValue)
                    return rhsValue.signedLessOrEqual(limit - one).expr
                }
            }
        }
        return null
    }

    private fun <T : KBvSort> trySimplifyBvUnsignedMulNoOverflow(
        lhs: KExpr<T>,
        rhs: KExpr<T>
    ): KExpr<KBoolSort>? = with(ctx) {
        val size = lhs.sort.sizeBits
        val lhsValue = lhs as? KBitVecValue<T>
        val rhsValue = rhs as? KBitVecValue<T>
        val zero = bvZero(size)
        val one = bvOne(size)

        if (lhsValue != null && (lhsValue == zero || (lhsValue == one))) {
            return trueExpr
        }

        if (rhsValue != null && (rhsValue == zero || (rhsValue == one))) {
            return trueExpr
        }

        if (lhsValue != null && rhsValue != null) {
            val longLhs = concatBv(zero, lhsValue)
            val longRhs = concatBv(zero, rhsValue)
            val longMaxValue = concatBv(zero, bvMaxValueUnsigned(size))

            val product = longLhs * longRhs
            return product.signedLessOrEqual(longMaxValue).expr
        }

        return null
    }

    override fun transform(expr: KBv2IntExpr): KExpr<KIntSort> = simplifyApp(expr) { (arg) ->
        if (arg is KBitVecValue<*>) {
            val integerValue = if (expr.isSigned) {
                arg.toBigIntegerSigned()
            } else {
                arg.toBigIntegerUnsigned()
            }
            return@simplifyApp mkIntNum(integerValue)
        }

        mkBv2IntExpr(arg, expr.isSigned)
    }

    /**
     * (= (concat a b) c) ==>
     *  (and
     *      (= a (extract (0, <a_size>) c))
     *      (= b (extract (<a_size>, <a_size> + <b_size>) c))
     *  )
     * */
    fun <T : KBvSort> simplifyBvConcatEq(l: KExpr<T>, r: KExpr<T>): KExpr<KBoolSort> = with(ctx) {
        val lArgs = if (l is KBvConcatExpr) flatConcat(l).args else listOf(l)
        val rArgs = if (r is KBvConcatExpr) flatConcat(r).args else listOf(r)
        val result = arrayListOf<KExpr<KBoolSort>>()
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
                    result += newL eq newR
                    lowL = 0
                    lowR = 0
                    lIdx--
                    rIdx--
                }

                remainSizeL < remainSizeR -> {
                    val newL = mkBvExtractExpr(high = lSize - 1, low = lowL, value = lArg)
                    val newR = mkBvExtractExpr(high = remainSizeL + lowR - 1, low = lowR, value = rArg)
                    result += newL eq newR
                    lowL = 0
                    lowR += remainSizeL
                    lIdx--
                }

                else -> {
                    val newL = mkBvExtractExpr(high = remainSizeR + lowL - 1, low = lowL, value = lArg)
                    val newR = mkBvExtractExpr(high = rSize - 1, low = lowR, value = rArg)
                    result += newL eq newR
                    lowL += remainSizeR
                    lowR = 0
                    rIdx--
                }
            }
        }

        // restore concat order
        result.reverse()

        return mkAnd(result)
    }

    /**
     * (bvor (concat a b) c) ==>
     *  (concat
     *      (bvor (extract (0, <a_size>) c))
     *      (bvor b (extract (<a_size>, <a_size> + <b_size>) c))
     *  )
     * */
    private fun <T : KBvSort> distributeOrOverConcat(args: List<KExpr<T>>): SimplifierAuxExpression<T> =
        distributeOperationOverConcat(args) { a, b -> KBvOrExpr(ctx, a, b) }

    /**
     * (bvxor (concat a b) c) ==>
     *  (concat
     *      (bvxor (extract (0, <a_size>) c))
     *      (bvxor b (extract (<a_size>, <a_size> + <b_size>) c))
     *  )
     * */
    private fun <T : KBvSort> distributeXorOverConcat(args: List<KExpr<T>>): SimplifierAuxExpression<T> =
        distributeOperationOverConcat(args) { a, b -> KBvXorExpr(ctx, a, b) }

    private inline fun <T : KBvSort> distributeOperationOverConcat(
        args: List<KExpr<T>>,
        crossinline operation: (KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<KBvSort>
    ): SimplifierAuxExpression<T> = auxExpr {
        val firstConcat = args.first { it is KBvConcatExpr } as KBvConcatExpr
        val size = firstConcat.sort.sizeBits.toInt()
        val partSize = firstConcat.arg0.sort.sizeBits.toInt()

        val args1 = arrayListOf<KExpr<KBvSort>>()
        val args2 = arrayListOf<KExpr<KBvSort>>()

        for (expr in args) {
            args1 += KBvExtractExpr(
                ctx,
                high = size - 1,
                low = size - partSize,
                expr.uncheckedCast()
            )
            args2 += KBvExtractExpr(
                ctx,
                high = size - partSize - 1,
                low = 0,
                expr.uncheckedCast()
            )
        }

        val mergedArgs1 = args1.reduceBinaryBvExpr(operation)
        val mergedArgs2 = args2.reduceBinaryBvExpr(operation)

        KBvConcatExpr(ctx, mergedArgs1, mergedArgs2).uncheckedCast()
    }

    private fun flatConcat(expr: KBvConcatExpr): SimplifierFlatBvConcatExpr {
        val flatten = flatBinaryBvExpr<KBvConcatExpr>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 },
            getRhs = { it.arg1 }
        )
        return SimplifierFlatBvConcatExpr(ctx, expr.sort, flatten)
    }

    @Suppress("UNCHECKED_CAST")
    private fun <S : KBvSort> flatBvAdd(expr: KBvAddExpr<S>): SimplifierFlatBvAddExpr<S> {
        val flatten = flatBinaryBvExpr<KBvAddExpr<*>>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 as KExpr<KBvSort> },
            getRhs = { it.arg1 as KExpr<KBvSort> }
        )
        return SimplifierFlatBvAddExpr(ctx, flatten) as SimplifierFlatBvAddExpr<S>
    }

    @Suppress("UNCHECKED_CAST")
    private fun <S : KBvSort> flatBvMul(expr: KBvMulExpr<S>): SimplifierFlatBvMulExpr<S> {
        val flatten = flatBinaryBvExpr<KBvMulExpr<*>>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 as KExpr<KBvSort> },
            getRhs = { it.arg1 as KExpr<KBvSort> }
        )
        return SimplifierFlatBvMulExpr(ctx, flatten) as SimplifierFlatBvMulExpr<S>
    }

    @Suppress("UNCHECKED_CAST")
    private fun <S : KBvSort> flatBvOr(expr: KBvOrExpr<S>): SimplifierFlatBvOrExpr<S> {
        val flatten = flatBinaryBvExpr<KBvOrExpr<*>>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 as KExpr<KBvSort> },
            getRhs = { it.arg1 as KExpr<KBvSort> }
        )
        return SimplifierFlatBvOrExpr(ctx, flatten) as SimplifierFlatBvOrExpr<S>
    }

    @Suppress("UNCHECKED_CAST")
    private fun <S : KBvSort> flatBvXor(expr: KBvXorExpr<S>): SimplifierFlatBvXorExpr<S> {
        val flatten = flatBinaryBvExpr<KBvXorExpr<*>>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 as KExpr<KBvSort> },
            getRhs = { it.arg1 as KExpr<KBvSort> }
        )
        return SimplifierFlatBvXorExpr(ctx, flatten) as SimplifierFlatBvXorExpr<S>
    }

    private inline fun <reified T> flatBinaryBvExpr(
        initial: KExpr<KBvSort>,
        getLhs: (T) -> KExpr<KBvSort>,
        getRhs: (T) -> KExpr<KBvSort>
    ): List<KExpr<KBvSort>> {
        val flatten = arrayListOf<KExpr<KBvSort>>()
        val unprocessed = arrayListOf<KExpr<KBvSort>>()
        unprocessed += initial
        while (unprocessed.isNotEmpty()) {
            val e = unprocessed.removeLast()
            if (e !is T) {
                flatten += e
                continue
            }
            unprocessed += getRhs(e)
            unprocessed += getLhs(e)
        }
        return flatten
    }

    /**
     * Reduce to an almost balanced expression tree.
     * */
    private inline fun <T : KBvSort> List<KExpr<T>>.reduceBinaryBvExpr(
        crossinline reducer: (KExpr<T>, KExpr<T>) -> KExpr<T>
    ): KExpr<T> {
        if (isEmpty()) {
            throw NoSuchElementException("List is empty")
        }
        val result = toMutableList()
        var size = size
        while (size > 1) {
            val unpairedLastElement = size % 2 == 1
            val realSize = if (unpairedLastElement) size - 1 else size
            var writeIdx = 0
            var readIdx = 0
            while (readIdx < realSize) {
                val arg0 = result[readIdx]
                val arg1 = result[readIdx + 1]
                result[writeIdx] = reducer(arg0, arg1)
                readIdx += 2
                writeIdx++
            }
            if (unpairedLastElement) {
                result[writeIdx] = result[size - 1]
                writeIdx++
            }
            size = writeIdx
        }
        return result[0]
    }

    /**
     * Auxiliary expression to store n-ary bv-add.
     * @see [SimplifierAuxExpression]
     * */
    private class SimplifierFlatBvAddExpr<T : KBvSort>(
        ctx: KContext,
        override val args: List<KExpr<T>>
    ) : KApp<T, KExpr<T>>(ctx) {

        override val decl: KDecl<T>
            get() = ctx.mkBvAddDecl(sort, sort)

        override val sort: T
            get() = args.first().sort

        override fun accept(transformer: KTransformerBase): KExpr<T> {
            transformer as KBvExprSimplifier
            return transformer.transform(this)
        }
    }

    /**
     * Auxiliary expression to store n-ary bv-mul.
     * @see [SimplifierAuxExpression]
     * */
    private class SimplifierFlatBvMulExpr<T : KBvSort>(
        ctx: KContext,
        override val args: List<KExpr<T>>
    ) : KApp<T, KExpr<T>>(ctx) {

        override val decl: KDecl<T>
            get() = ctx.mkBvMulDecl(sort, sort)

        override val sort: T
            get() = args.first().sort

        override fun accept(transformer: KTransformerBase): KExpr<T> {
            transformer as KBvExprSimplifier
            return transformer.transform(this)
        }
    }

    /**
     * Auxiliary expression to store n-ary bv-or.
     * @see [SimplifierAuxExpression]
     * */
    private class SimplifierFlatBvOrExpr<T : KBvSort>(
        ctx: KContext,
        override val args: List<KExpr<T>>
    ) : KApp<T, KExpr<T>>(ctx) {

        override val decl: KDecl<T>
            get() = ctx.mkBvOrDecl(sort, sort)

        override val sort: T
            get() = args.first().sort

        override fun accept(transformer: KTransformerBase): KExpr<T> {
            transformer as KBvExprSimplifier
            return transformer.transform(this)
        }
    }

    /**
     * Auxiliary expression to store n-ary bv-xor.
     * @see [SimplifierAuxExpression]
     * */
    private class SimplifierFlatBvXorExpr<T : KBvSort>(
        ctx: KContext,
        override val args: List<KExpr<T>>
    ) : KApp<T, KExpr<T>>(ctx) {

        override val decl: KDecl<T>
            get() = ctx.mkBvXorDecl(sort, sort)

        override val sort: T
            get() = args.first().sort

        override fun accept(transformer: KTransformerBase): KExpr<T> {
            transformer as KBvExprSimplifier
            return transformer.transform(this)
        }
    }

    /**
     * Auxiliary expression to store n-ary bv-concat.
     * @see [SimplifierAuxExpression]
     * */
    private class SimplifierFlatBvConcatExpr(
        ctx: KContext,
        override val sort: KBvSort,
        override val args: List<KExpr<KBvSort>>
    ) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {

        // We have no decl, but we don't care since decl is unused
        override val decl: KDecl<KBvSort>
            get() = error("Decl of SimplifierFlatBvConcatExpr should not be used")

        override fun accept(transformer: KTransformerBase): KExpr<KBvSort> {
            transformer as KBvExprSimplifier
            return transformer.transform(this)
        }

        override fun print(builder: StringBuilder): Unit = with(builder) {
            append("(concat")
            for (arg in args) {
                append(" ")
                arg.print(this)
            }
            append(")")
        }
    }

    private fun concatBv(lhs: KBitVecValue<*>, rhs: KBitVecValue<*>): KBitVecValue<*> = with(ctx) {
        when {
            lhs is KBitVec8Value && rhs is KBitVec8Value -> {
                var result = lhs.numberValue.toUByte().toInt() shl Byte.SIZE_BITS
                result = result or rhs.numberValue.toUByte().toInt()
                mkBv(result.toShort())
            }

            lhs is KBitVec16Value && rhs is KBitVec16Value -> {
                var result = lhs.numberValue.toUShort().toInt() shl Short.SIZE_BITS
                result = result or rhs.numberValue.toUShort().toInt()
                mkBv(result)
            }

            lhs is KBitVec32Value && rhs is KBitVec32Value -> {
                var result = lhs.numberValue.toUInt().toLong() shl Int.SIZE_BITS
                result = result or rhs.numberValue.toUInt().toLong()
                mkBv(result)
            }

            else -> {
                val concatenatedBinary = lhs.stringValue + rhs.stringValue
                mkBv(concatenatedBinary, lhs.sort.sizeBits + rhs.sort.sizeBits)
            }
        }
    }

    private fun extractBv(value: KBitVecValue<*>, high: Int, low: Int): KBitVecValue<*> = with(ctx) {
        val size = high - low + 1
        val allBits = value.stringValue
        val highBitIdx = allBits.length - high - 1
        val lowBitIdx = allBits.length - low - 1
        val bits = allBits.substring(highBitIdx, lowBitIdx + 1)
        mkBv(bits, size.toUInt())
    }

    // (concat (extract[h1, l1] a) (extract[h2, l2] a)), l1 == h2 + 1 ==> (extract[h1, l2] a)
    private fun tryMergeBvConcatExtract(lhs: KBvExtractExpr, rhs: KBvExtractExpr): KExpr<KBvSort>? = with(ctx) {
        if (lhs.value != rhs.value || lhs.low != rhs.high + 1) {
            return null
        }
        mkBvExtractExpr(lhs.high, rhs.low, lhs.value)
    }

    private fun KBitVecValue<*>.signExtension(extensionSize: Int): KBitVecValue<*> {
        val binary = stringValue
        val sign = binary[0]
        val extension = "$sign".repeat(extensionSize)
        return ctx.mkBv(extension + binary, sort.sizeBits + extensionSize.toUInt())
    }

}
