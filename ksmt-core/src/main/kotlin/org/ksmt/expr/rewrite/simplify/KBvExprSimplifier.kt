package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KAndExpr
import org.ksmt.expr.KApp
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
import org.ksmt.expr.KNotExpr
import org.ksmt.expr.printer.ExpressionPrinter
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KIntSort
import org.ksmt.utils.BvUtils.bigIntValue
import org.ksmt.utils.BvUtils.bitwiseAnd
import org.ksmt.utils.BvUtils.bitwiseNot
import org.ksmt.utils.BvUtils.bitwiseOr
import org.ksmt.utils.BvUtils.bitwiseXor
import org.ksmt.utils.BvUtils.bvMaxValueUnsigned
import org.ksmt.utils.BvUtils.bvOne
import org.ksmt.utils.BvUtils.bvZero
import org.ksmt.utils.BvUtils.concatBv
import org.ksmt.utils.BvUtils.extractBv
import org.ksmt.utils.BvUtils.isBvOne
import org.ksmt.utils.BvUtils.isBvZero
import org.ksmt.utils.BvUtils.minus
import org.ksmt.utils.BvUtils.plus
import org.ksmt.utils.BvUtils.shiftLeft
import org.ksmt.utils.BvUtils.shiftRightLogical
import org.ksmt.utils.BvUtils.signExtension
import org.ksmt.utils.BvUtils.signedGreaterOrEqual
import org.ksmt.utils.BvUtils.times
import org.ksmt.utils.BvUtils.zeroExtension
import org.ksmt.utils.cast
import org.ksmt.utils.toBigInteger
import org.ksmt.utils.uncheckedCast
import java.math.BigInteger

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

        mkEqNoSimplify(lhs, rhs)
    }

    fun <T : KBvSort> areDefinitelyDistinctBv(lhs: KExpr<T>, rhs: KExpr<T>): Boolean {
        if (lhs is KBitVecValue<T> && rhs is KBitVecValue<T>) {
            return lhs != rhs
        }
        return false
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            simplifyBvUnsignedLessOrEqualExpr(lhs, rhs)
        }

    override fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            simplifyBvSignedLessOrEqualExpr(lhs, rhs)
        }

    // (uge a b) ==> (ule b a)
    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyApp(
            expr = expr,
            preprocess = { KBvUnsignedLessOrEqualExpr(this, expr.arg1, expr.arg0) }
        ) {
            error("Always preprocessed")
        }

    // (ult a b) ==> (not (ule b a))
    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>): KExpr<KBoolSort> =
        simplifyApp(
            expr = expr,
            preprocess = {
                val ule = KBvUnsignedLessOrEqualExpr(this, expr.arg1, expr.arg0)
                KNotExpr(this, ule)
            }
        ) {
            error("Always preprocessed")
        }

    // (ugt a b) ==> (not (ule a b))
    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>): KExpr<KBoolSort> =
        simplifyApp(
            expr = expr,
            preprocess = {
                val ule = KBvUnsignedLessOrEqualExpr(this, expr.arg0, expr.arg1)
                KNotExpr(this, ule)
            }
        ) {
            error("Always preprocessed")
        }

    // (sge a b) ==> (sle b a)
    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyApp(
            expr = expr,
            preprocess = { KBvSignedLessOrEqualExpr(this, expr.arg1, expr.arg0) }
        ) {
            error("Always preprocessed")
        }

    // (slt a b) ==> (not (sle b a))
    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>): KExpr<KBoolSort> =
        simplifyApp(
            expr = expr,
            preprocess = {
                val sle = KBvSignedLessOrEqualExpr(this, expr.arg1, expr.arg0)
                KNotExpr(this, sle)
            }
        ) {
            error("Always preprocessed")
        }

    // (sgt a b) ==> (not (sle a b))
    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>): KExpr<KBoolSort> =
        simplifyApp(
            expr = expr,
            preprocess = {
                val sle = KBvSignedLessOrEqualExpr(this, expr.arg0, expr.arg1)
                KNotExpr(this, sle)
            }
        ) {
            error("Always preprocessed")
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

        resultParts.reduceBinaryBvExpr(::mkBvAddExprNoSimplify)
    }

    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>): KExpr<T> =
        simplifyApp(
            expr = expr,
            preprocess = {
                val negativeRhs = KBvNegationExpr(this, expr.arg1)
                KBvAddExpr(this, expr.arg0, negativeRhs)
            }
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
        if (constantValue.isBvZero()) {
            return@simplifyApp zero.uncheckedCast()
        }

        if (resultParts.isEmpty()) {
            return@simplifyApp constantValue.uncheckedCast()
        }

        // (* 1 a) ==> a
        if (constantValue.isBvOne()) {
            return@simplifyApp resultParts.reduceBinaryBvExpr(::mkBvMulExprNoSimplify)
        }

        // (* -1 a) ==> -a
        val minusOne = zero - one
        if (constantValue == minusOne) {
            val value = resultParts.reduceBinaryBvExpr(::mkBvMulExprNoSimplify)
            return@simplifyApp mkBvNegationExprNoSimplify(value)
        }

        resultParts.add(constantValue.uncheckedCast())
        resultParts.reduceBinaryBvExpr(::mkBvMulExprNoSimplify)
    }

    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>): KExpr<T> = simplifyApp(expr) { (arg) ->
        simplifyBvNegationExpr(arg)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        simplifyBvSignedDivExpr(lhs, rhs)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        simplifyBvUnsignedDivExpr(lhs, rhs)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        simplifyBvSignedRemExpr(lhs, rhs)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        simplifyBvUnsignedRemExpr(lhs, rhs)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        simplifyBvSignedModExpr(lhs, rhs)
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
        if (arg is KBvConcatExpr && canPerformBoundedRewrite()) {
            val concatParts = flatConcat(arg).args
            return@simplifyApp boundedRewrite(
                auxExpr {
                    val negatedParts = concatParts.map { KBvNotExpr(ctx, it) }
                    SimplifierFlatBvConcatExpr(ctx, arg.sort, negatedParts).uncheckedCast()
                }
            )
        }

        // (bvnot (ite c a b)) ==> (ite c (bvnot a) (bvnot b))
        if (arg is KIteExpr<T> && canPerformBoundedRewrite()) {
            if (arg.trueBranch is KBitVecValue<T> || arg.falseBranch is KBitVecValue<T>) {
                return@simplifyApp boundedRewrite(
                    auxExpr {
                        val trueBranch = KBvNotExpr(this, arg.trueBranch)
                        val falseBranch = KBvNotExpr(this, arg.falseBranch)
                        KIteExpr(this, arg.condition, trueBranch, falseBranch)
                    }
                )
            }
        }

        mkBvNotExprNoSimplify(arg)
    }

    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>): KExpr<T> =
        simplifyApp(expr = expr, preprocess = { flatBvOr(expr) }) {
            error("Always preprocessed")
        }

    @Suppress("LoopWithTooManyJumpStatements")
    private fun <T : KBvSort> transform(expr: SimplifierFlatBvOrExpr<T>): KExpr<T> = simplifyApp(expr) { flatten ->
        simplifyBvAndOr(
            args = flatten,
            neutralElement = bvZero(expr.sort.sizeBits).uncheckedCast(),
            zeroElement = bvMaxValueUnsigned(expr.sort.sizeBits).uncheckedCast(),
            operation = { a, b -> a.bitwiseOr(b).uncheckedCast() }
        ) { resultParts ->

            /**
             * (bvor (concat a b) c) ==>
             *  (concat
             *      (bvor (extract (0, <a_size>) c))
             *      (bvor b (extract (<a_size>, <a_size> + <b_size>) c))
             *  )
             * */
            if (resultParts.any { it is KBvConcatExpr } && canPerformBoundedRewrite()) {
                return@simplifyApp boundedRewrite(distributeOrOverConcat(resultParts))
            }

            resultParts.reduceBinaryBvExpr(::mkBvOrExprNoSimplify)
        }
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
            addAll(negativeParts.map { mkBvNotExprNoSimplify(it) })
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

        if (resultParts.any { it is KBvConcatExpr } && canPerformBoundedRewrite()) {
            val preResult = distributeXorOverConcat(resultParts)
            val result = if (negateResult) {
                auxExpr {
                    KBvNotExpr(ctx, preResult.expr)
                }
            } else {
                preResult
            }
            return@simplifyApp boundedRewrite(result)
        }

        val preResult = resultParts.reduceBinaryBvExpr(::mkBvXorExprNoSimplify)
        return@simplifyApp if (negateResult) mkBvNotExprNoSimplify(preResult) else preResult
    }

    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>): KExpr<T> =
        simplifyApp(expr = expr, preprocess = { flatBvAnd(expr) }) {
            error("Always preprocessed")
        }

    private fun <T : KBvSort> transform(expr: SimplifierFlatBvAndExpr<T>): KExpr<T> = simplifyApp(expr) { flatten ->
        simplifyBvAndOr(
            args = flatten,
            neutralElement = bvMaxValueUnsigned(expr.sort.sizeBits).uncheckedCast(),
            zeroElement = bvZero(expr.sort.sizeBits).uncheckedCast(),
            operation = { a, b -> a.bitwiseAnd(b).uncheckedCast() }
        ) { resultParts ->

            /**
             * (bvand (concat a b) c) ==>
             *  (concat
             *      (bvand (extract (0, <a_size>) c))
             *      (bvand b (extract (<a_size>, <a_size> + <b_size>) c))
             *  )
             * */
            if (resultParts.any { it is KBvConcatExpr } && canPerformBoundedRewrite()) {
                return@simplifyApp boundedRewrite(distributeAndOverConcat(resultParts))
            }

            resultParts.reduceBinaryBvExpr(::mkBvAndExprNoSimplify)
        }
    }

    // (bvnand a b) ==> (bvor (bvnot a) (bvnot b))
    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>): KExpr<T> =
        simplifyApp(
            expr = expr,
            preprocess = {
                KBvOrExpr(
                    this,
                    KBvNotExpr(this, expr.arg0),
                    KBvNotExpr(this, expr.arg1)
                )
            }
        ) {
            error("Always preprocessed")
        }

    // (bvnor a b) ==> (bvnot (bvor a b))
    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>): KExpr<T> =
        simplifyApp(
            expr = expr,
            preprocess = { KBvNotExpr(this, KBvOrExpr(this, expr.arg0, expr.arg1)) }
        ) {
            error("Always preprocessed")
        }

    // (bvxnor a b) ==> (bvnot (bvxor a b))
    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>): KExpr<T> =
        simplifyApp(
            expr = expr,
            preprocess = { KBvNotExpr(this, KBvXorExpr(this, expr.arg0, expr.arg1)) }
        ) {
            error("Always preprocessed")
        }

    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>): KExpr<KBv1Sort> = simplifyApp(expr) { (arg) ->
        simplifyBvReductionAndExpr(arg)
    }

    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>): KExpr<KBv1Sort> = simplifyApp(expr) { (arg) ->
        simplifyBvReductionOrExpr(arg)
    }

    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> =
        simplifyApp(expr = expr, preprocess = { flatConcat(expr) }) {
            error("Always preprocessed")
        }

    @Suppress("LoopWithTooManyJumpStatements")
    private fun transform(expr: SimplifierFlatBvConcatExpr): KExpr<KBvSort> = simplifyApp(expr) { flatten ->
        val mergedParts = arrayListOf(flatten.first())
        var hasSimplifierAuxTerms = false

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
                    mergedParts.add(possiblyMerged.expr)
                    hasSimplifierAuxTerms = true
                    continue
                }
            }
            mergedParts.add(part)
        }

        if (hasSimplifierAuxTerms) {
            rewrite(SimplifierFlatBvConcatExpr(ctx, expr.sort, mergedParts))
        } else {
            mergedParts.reduceBinaryBvExpr(::mkBvConcatExprNoSimplify)
        }
    }

    override fun transform(expr: KBvExtractExpr): KExpr<KBvSort> = simplifyApp(expr) { (arg) ->
        // (extract [size-1:0] x) ==> x
        if (expr.low == 0 && expr.high == arg.sort.sizeBits.toInt() - 1) {
            return@simplifyApp arg
        }

        if (arg is KBitVecValue<*>) {
            return@simplifyApp arg.extractBv(expr.high, expr.low).uncheckedCast()
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


        val simplified = when {
            // Check bounded rewrite is available before applying rules
            !canPerformBoundedRewrite() -> null

            // (extract (concat a b)) ==> (concat (extract a) (extract b))
            arg is KBvConcatExpr -> {
                distributeExtractOverConcat(arg, expr.high, expr.low)
            }
            // (extract [h:l] (bvnot x)) ==> (bvnot (extract [h:l] x))
            arg is KBvNotExpr<*> -> auxExpr {
                KBvNotExpr(ctx, KBvExtractExpr(ctx, expr.high, expr.low, arg.value.uncheckedCast()))
            }
            // (extract [h:l] (bvor a b)) ==> (bvor (extract [h:l] a) (extract [h:l] b))
            arg is KBvOrExpr<*> -> auxExpr {
                val lhs = KBvExtractExpr(ctx, expr.high, expr.low, arg.arg0.uncheckedCast())
                val rhs = KBvExtractExpr(ctx, expr.high, expr.low, arg.arg1.uncheckedCast())
                KBvOrExpr(ctx, lhs, rhs)
            }
            // (extract [h:l] (bvand a b)) ==> (bvand (extract [h:l] a) (extract [h:l] b))
            arg is KBvAndExpr<*> -> auxExpr {
                val lhs = KBvExtractExpr(ctx, expr.high, expr.low, arg.arg0.uncheckedCast())
                val rhs = KBvExtractExpr(ctx, expr.high, expr.low, arg.arg1.uncheckedCast())
                KBvAndExpr(ctx, lhs, rhs)
            }
            // (extract [h:l] (bvxor a b)) ==> (bvxor (extract [h:l] a) (extract [h:l] b))
            arg is KBvXorExpr<*> -> auxExpr {
                val lhs = KBvExtractExpr(ctx, expr.high, expr.low, arg.arg0.uncheckedCast())
                val rhs = KBvExtractExpr(ctx, expr.high, expr.low, arg.arg1.uncheckedCast())
                KBvXorExpr(ctx, lhs, rhs)
            }
            // (extract [h:0] (bvadd a b)) ==> (bvadd (extract [h:0] a) (extract [h:0] b))
            arg is KBvAddExpr<*> && expr.low == 0 -> auxExpr {
                val lhs = KBvExtractExpr(ctx, expr.high, low = 0, arg.arg0.uncheckedCast())
                val rhs = KBvExtractExpr(ctx, expr.high, low = 0, arg.arg1.uncheckedCast())
                KBvAddExpr(ctx, lhs, rhs)
            }
            // (extract [h:0] (bvmul a b)) ==> (bvmul (extract [h:0] a) (extract [h:0] b))
            arg is KBvMulExpr<*> && expr.low == 0 -> auxExpr {
                val lhs = KBvExtractExpr(ctx, expr.high, low = 0, arg.arg0.uncheckedCast())
                val rhs = KBvExtractExpr(ctx, expr.high, low = 0, arg.arg1.uncheckedCast())
                KBvMulExpr(ctx, lhs, rhs)
            }

            else -> null
        }

        if (simplified != null && canPerformBoundedRewrite()) {
            return@simplifyApp boundedRewrite(simplified)
        }

        mkBvExtractExprNoSimplify(expr.high, expr.low, arg)
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
            if (shiftValue.isBvZero()) {
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
            val intShiftValue = shiftValue.bigIntValue()
            if (intShiftValue >= BigInteger.ZERO && intShiftValue <= Int.MAX_VALUE.toBigInteger()) {
                return@simplifyApp rewrite(
                    auxExpr {
                        KBvConcatExpr(
                            ctx,
                            KBvExtractExpr(
                                ctx,
                                high = size.toInt() - 1 - intShiftValue.toInt(),
                                low = 0,
                                arg.uncheckedCast()
                            ),
                            bvZero(intShiftValue.toInt().toUInt()).uncheckedCast()
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
            return@simplifyApp rewrite(
                auxExpr {
                    val nestedArg = arg.arg
                    val nestedShift = arg.shift
                    val sum = KBvAddExpr(ctx, nestedShift, shift)
                    val cond = KBvUnsignedLessOrEqualExpr(ctx, nestedShift, sum)
                    KIteExpr(
                        ctx,
                        condition = cond,
                        trueBranch = KBvShiftLeftExpr(ctx, nestedArg, sum),
                        falseBranch = bvZero(size).uncheckedCast()
                    )
                }
            )
        }

        mkBvShiftLeftExprNoSimplify(arg, shift)
    }

    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>): KExpr<T> =
        simplifyApp(expr) { (arg, shift) ->
            val size = expr.sort.sizeBits
            val argValue = arg as? KBitVecValue<T>
            val shiftValue = shift as? KBitVecValue<T>

            if (shiftValue != null) {
                // (x >>> 0) ==> x
                if (shiftValue.isBvZero()) {
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
                val intShiftValue = shiftValue.bigIntValue()
                if (intShiftValue >= BigInteger.ZERO && intShiftValue <= Int.MAX_VALUE.toBigInteger()) {
                    return@simplifyApp rewrite(
                        auxExpr {
                            val lhs = bvZero(intShiftValue.toInt().toUInt())
                            val rhs = KBvExtractExpr(
                                ctx,
                                high = size.toInt() - 1,
                                low = intShiftValue.toInt(),
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

            mkBvLogicalShiftRightExprNoSimplify(arg, shift)
        }

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>): KExpr<T> =
        simplifyApp(expr) { (arg, shift) ->
            simplifyBvArithShiftRightExpr(arg, shift)
        }

    // (repeat a x) ==> (concat a a ..[x].. a)
    override fun transform(expr: KBvRepeatExpr): KExpr<KBvSort> = simplifyApp(expr) { (arg) ->
        val repeats = arrayListOf<KExpr<KBvSort>>()
        repeat(expr.repeatNumber) {
            repeats += arg
        }

        if (repeats.size == 0) {
            return@simplifyApp mkBvRepeatExprNoSimplify(expr.repeatNumber, arg)
        }

        return@simplifyApp rewrite(SimplifierFlatBvConcatExpr(ctx, expr.sort, repeats))
    }

    // (zeroext a) ==> (concat 0 a)
    override fun transform(expr: KBvZeroExtensionExpr): KExpr<KBvSort> = simplifyApp(expr) { (arg) ->
        if (expr.extensionSize == 0) {
            return@simplifyApp arg
        }

        if (arg is KBitVecValue<*>) {
            return@simplifyApp arg.zeroExtension(expr.extensionSize.toUInt()).uncheckedCast()
        }

        return@simplifyApp rewrite(
            auxExpr {
                KBvConcatExpr(ctx, bvZero(expr.extensionSize.toUInt()).uncheckedCast(), arg)
            }
        )
    }

    override fun transform(expr: KBvSignExtensionExpr): KExpr<KBvSort> = simplifyApp(expr) { (arg) ->
        if (expr.extensionSize == 0) {
            return@simplifyApp arg
        }

        if (arg is KBitVecValue<*>) {
            return@simplifyApp arg.signExtension(expr.extensionSize.toUInt()).uncheckedCast()
        }

        return@simplifyApp mkBvSignExtensionExprNoSimplify(expr.extensionSize, arg)
    }

    // (rotateLeft a x) ==> (concat (extract [size-1-x:0] a) (extract [size-1:size-x] a))
    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>): KExpr<T> = simplifyApp(expr) { (arg) ->
        simplifyBvRotateLeftIndexedExpr(expr.rotationNumber, arg)
    }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>): KExpr<T> = simplifyApp(expr) { (arg, rotation) ->
        simplifyBvRotateLeftExpr(arg, rotation)
    }

    // (rotateRight a x) ==> (rotateLeft a (- size x))
    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>): KExpr<T> = simplifyApp(expr) { (arg) ->
        simplifyBvRotateRightIndexedExpr(expr.rotationNumber, arg)
    }

    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>): KExpr<T> = simplifyApp(expr) { (arg, rotation) ->
        simplifyBvRotateRightExpr(arg, rotation)
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            rewriteBvAddNoOverflowExpr(lhs, rhs, expr.isSigned)
        }

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            rewriteBvAddNoUnderflowExpr(lhs, rhs)
        }

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            rewriteBvSubNoOverflowExpr(lhs, rhs)
        }

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            rewriteBvSubNoUnderflowExpr(lhs, rhs, expr.isSigned)
        }

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            rewriteBvNegNoOverflowExpr(arg)
        }

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            rewriteBvDivNoOverflowExpr(lhs, rhs)
        }

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            rewriteBvMulNoOverflowExpr(lhs, rhs, expr.isSigned)
        }

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            rewriteBvMulNoUnderflowExpr(lhs, rhs)
        }

    override fun transform(expr: KBv2IntExpr): KExpr<KIntSort> = simplifyApp(expr) { (arg) ->
        simplifyBv2IntExpr(arg, expr.isSigned)
    }

    /**
     * (= (concat a b) c) ==>
     *  (and
     *      (= a (extract (0, <a_size>) c))
     *      (= b (extract (<a_size>, <a_size> + <b_size>) c))
     *  )
     * */
    private fun <T : KBvSort> simplifyBvConcatEq(
        l: KExpr<T>,
        r: KExpr<T>
    ): SimplifierAuxExpression<KBoolSort> = auxExpr {
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
                    val newL = KBvExtractExpr(
                        ctx, high = lSize - 1, low = lowL, value = lArg.uncheckedCast()
                    )
                    val newR = KBvExtractExpr(
                        ctx, high = rSize - 1, low = lowR, value = rArg.uncheckedCast()
                    )
                    result += KEqExpr(ctx, newL, newR)
                    lowL = 0
                    lowR = 0
                    lIdx--
                    rIdx--
                }

                remainSizeL < remainSizeR -> {
                    val newL = KBvExtractExpr(
                        ctx, high = lSize - 1, low = lowL, value = lArg.uncheckedCast()
                    )
                    val newR = KBvExtractExpr(
                        ctx, high = remainSizeL + lowR - 1, low = lowR, value = rArg.uncheckedCast()
                    )
                    result += KEqExpr(ctx, newL, newR)
                    lowL = 0
                    lowR += remainSizeL
                    lIdx--
                }

                else -> {
                    val newL = KBvExtractExpr(
                        ctx, high = remainSizeR + lowL - 1, low = lowL, value = lArg.uncheckedCast()
                    )
                    val newR = KBvExtractExpr(
                        ctx, high = rSize - 1, low = lowR, value = rArg.uncheckedCast()
                    )
                    result += KEqExpr(ctx, newL, newR)
                    lowL += remainSizeR
                    lowR = 0
                    rIdx--
                }
            }
        }

        // restore concat order
        result.reverse()

        KAndExpr(ctx, result)
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
     * (bvand (concat a b) c) ==>
     *  (concat
     *      (bvand (extract (0, <a_size>) c))
     *      (bvand b (extract (<a_size>, <a_size> + <b_size>) c))
     *  )
     * */
    private fun <T : KBvSort> distributeAndOverConcat(args: List<KExpr<T>>): SimplifierAuxExpression<T> =
        distributeOperationOverConcat(args) { a, b -> KBvAndExpr(ctx, a, b) }

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
        return SimplifierFlatBvAddExpr(ctx, flatten).uncheckedCast()
    }

    @Suppress("UNCHECKED_CAST")
    private fun <S : KBvSort> flatBvMul(expr: KBvMulExpr<S>): SimplifierFlatBvMulExpr<S> {
        val flatten = flatBinaryBvExpr<KBvMulExpr<*>>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 as KExpr<KBvSort> },
            getRhs = { it.arg1 as KExpr<KBvSort> }
        )
        return SimplifierFlatBvMulExpr(ctx, flatten).uncheckedCast()
    }

    @Suppress("UNCHECKED_CAST")
    private fun <S : KBvSort> flatBvOr(expr: KBvOrExpr<S>): SimplifierFlatBvOrExpr<S> {
        val flatten = flatBinaryBvExpr<KBvOrExpr<*>>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 as KExpr<KBvSort> },
            getRhs = { it.arg1 as KExpr<KBvSort> }
        )
        return SimplifierFlatBvOrExpr(ctx, flatten).uncheckedCast()
    }

    @Suppress("UNCHECKED_CAST")
    private fun <S : KBvSort> flatBvAnd(expr: KBvAndExpr<S>): SimplifierFlatBvAndExpr<S> {
        val flatten = flatBinaryBvExpr<KBvAndExpr<*>>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 as KExpr<KBvSort> },
            getRhs = { it.arg1 as KExpr<KBvSort> }
        )
        return SimplifierFlatBvAndExpr(ctx, flatten).uncheckedCast()
    }

    @Suppress("UNCHECKED_CAST")
    private fun <S : KBvSort> flatBvXor(expr: KBvXorExpr<S>): SimplifierFlatBvXorExpr<S> {
        val flatten = flatBinaryBvExpr<KBvXorExpr<*>>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 as KExpr<KBvSort> },
            getRhs = { it.arg1 as KExpr<KBvSort> }
        )
        return SimplifierFlatBvXorExpr(ctx, flatten).uncheckedCast()
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
    ) : KApp<T, T>(ctx) {

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
    ) : KApp<T, T>(ctx) {

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
    ) : KApp<T, T>(ctx) {

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
     * Auxiliary expression to store n-ary bv-and.
     * @see [SimplifierAuxExpression]
     * */
    private class SimplifierFlatBvAndExpr<T : KBvSort>(
        ctx: KContext,
        override val args: List<KExpr<T>>
    ) : KApp<T, T>(ctx) {

        override val decl: KDecl<T>
            get() = ctx.mkBvAndDecl(sort, sort)

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
    ) : KApp<T, T>(ctx) {

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
    ) : KApp<KBvSort, KBvSort>(ctx) {

        // We have no decl, but we don't care since decl is unused
        override val decl: KDecl<KBvSort>
            get() = error("Decl of SimplifierFlatBvConcatExpr should not be used")

        override fun accept(transformer: KTransformerBase): KExpr<KBvSort> {
            transformer as KBvExprSimplifier
            return transformer.transform(this)
        }

        override fun print(printer: ExpressionPrinter) {
            with(printer) {
                append("(concat")
                for (arg in args) {
                    append(" ")
                    append(arg)
                }
                append(")")
            }
        }
    }

    // (concat (extract[h1, l1] a) (extract[h2, l2] a)), l1 == h2 + 1 ==> (extract[h1, l2] a)
    private fun tryMergeBvConcatExtract(lhs: KBvExtractExpr, rhs: KBvExtractExpr): SimplifierAuxExpression<KBvSort>? {
        if (lhs.value != rhs.value || lhs.low != rhs.high + 1) {
            return null
        }
        return auxExpr { KBvExtractExpr(ctx, lhs.high, rhs.low, lhs.value) }
    }

    private inline fun <T : KBvSort> simplifyBvAndOr(
        args: List<KExpr<T>>,
        neutralElement: KBitVecValue<T>,
        zeroElement: KBitVecValue<T>,
        operation: (KBitVecValue<T>, KBitVecValue<T>) -> KBitVecValue<T>,
        buildResult: (List<KExpr<T>>) -> KExpr<T>
    ): KExpr<T> {
        var constantValue = neutralElement
        val resultParts = arrayListOf<KExpr<T>>()
        val positiveTerms = hashSetOf<KExpr<T>>()
        val negativeTerms = hashSetOf<KExpr<T>>()

        for (arg in args) {
            if (arg is KBitVecValue<T>) {
                constantValue = operation(constantValue, arg)
                continue
            }

            if (arg is KBvNotExpr<T>) {
                val term = arg.value
                // (bvop (bvnot a) b (bvnot a)) ==> (bvop (bvnot a) b)
                if (!negativeTerms.add(term)) {
                    continue
                }

                // (bvop a (bvnot a)) ==> zero
                if (term in positiveTerms) {
                    return zeroElement
                }
            } else {
                // (bvop a b a) ==> (bvop a b)
                if (!positiveTerms.add(arg)) {
                    continue
                }

                // (bvop a (bvnot a)) ==> zero
                if (arg in negativeTerms) {
                    return zeroElement
                }
            }

            resultParts += arg
        }

        // (bvop zero a) ==> zero
        if (constantValue == zeroElement) {
            return zeroElement
        }

        if (resultParts.isEmpty()) {
            return constantValue.uncheckedCast()
        }

        // (bvop neutral a) ==> a
        if (constantValue != neutralElement) {
            resultParts.add(constantValue.uncheckedCast())
        }

        return buildResult(resultParts)
    }
}
