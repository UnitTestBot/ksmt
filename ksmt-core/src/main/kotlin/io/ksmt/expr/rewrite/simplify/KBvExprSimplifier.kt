package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KApp
import io.ksmt.expr.KBitVecValue
import io.ksmt.expr.KBv2IntExpr
import io.ksmt.expr.KBvAddExpr
import io.ksmt.expr.KBvAddNoOverflowExpr
import io.ksmt.expr.KBvAddNoUnderflowExpr
import io.ksmt.expr.KBvAndExpr
import io.ksmt.expr.KBvArithShiftRightExpr
import io.ksmt.expr.KBvConcatExpr
import io.ksmt.expr.KBvDivNoOverflowExpr
import io.ksmt.expr.KBvExtractExpr
import io.ksmt.expr.KBvLogicalShiftRightExpr
import io.ksmt.expr.KBvMulExpr
import io.ksmt.expr.KBvMulNoOverflowExpr
import io.ksmt.expr.KBvMulNoUnderflowExpr
import io.ksmt.expr.KBvNAndExpr
import io.ksmt.expr.KBvNegNoOverflowExpr
import io.ksmt.expr.KBvNegationExpr
import io.ksmt.expr.KBvNorExpr
import io.ksmt.expr.KBvNotExpr
import io.ksmt.expr.KBvOrExpr
import io.ksmt.expr.KBvReductionAndExpr
import io.ksmt.expr.KBvReductionOrExpr
import io.ksmt.expr.KBvRepeatExpr
import io.ksmt.expr.KBvRotateLeftExpr
import io.ksmt.expr.KBvRotateLeftIndexedExpr
import io.ksmt.expr.KBvRotateRightExpr
import io.ksmt.expr.KBvRotateRightIndexedExpr
import io.ksmt.expr.KBvShiftLeftExpr
import io.ksmt.expr.KBvSignExtensionExpr
import io.ksmt.expr.KBvSignedDivExpr
import io.ksmt.expr.KBvSignedGreaterExpr
import io.ksmt.expr.KBvSignedGreaterOrEqualExpr
import io.ksmt.expr.KBvSignedLessExpr
import io.ksmt.expr.KBvSignedLessOrEqualExpr
import io.ksmt.expr.KBvSignedModExpr
import io.ksmt.expr.KBvSignedRemExpr
import io.ksmt.expr.KBvSubExpr
import io.ksmt.expr.KBvSubNoOverflowExpr
import io.ksmt.expr.KBvSubNoUnderflowExpr
import io.ksmt.expr.KBvUnsignedDivExpr
import io.ksmt.expr.KBvUnsignedGreaterExpr
import io.ksmt.expr.KBvUnsignedGreaterOrEqualExpr
import io.ksmt.expr.KBvUnsignedLessExpr
import io.ksmt.expr.KBvUnsignedLessOrEqualExpr
import io.ksmt.expr.KBvUnsignedRemExpr
import io.ksmt.expr.KBvXNorExpr
import io.ksmt.expr.KBvXorExpr
import io.ksmt.expr.KBvZeroExtensionExpr
import io.ksmt.expr.KEqExpr
import io.ksmt.expr.KExpr
import io.ksmt.expr.KIteExpr
import io.ksmt.expr.KNotExpr
import io.ksmt.expr.printer.ExpressionPrinter
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KIntSort
import io.ksmt.utils.BvUtils.bigIntValue
import io.ksmt.utils.BvUtils.bitwiseAnd
import io.ksmt.utils.BvUtils.bitwiseNot
import io.ksmt.utils.BvUtils.bitwiseOr
import io.ksmt.utils.BvUtils.bitwiseXor
import io.ksmt.utils.BvUtils.bvMaxValueUnsigned
import io.ksmt.utils.BvUtils.bvOne
import io.ksmt.utils.BvUtils.bvZero
import io.ksmt.utils.BvUtils.concatBv
import io.ksmt.utils.BvUtils.extractBv
import io.ksmt.utils.BvUtils.isBvOne
import io.ksmt.utils.BvUtils.isBvZero
import io.ksmt.utils.BvUtils.minus
import io.ksmt.utils.BvUtils.plus
import io.ksmt.utils.BvUtils.shiftLeft
import io.ksmt.utils.BvUtils.shiftRightLogical
import io.ksmt.utils.BvUtils.signExtension
import io.ksmt.utils.BvUtils.signedGreaterOrEqual
import io.ksmt.utils.BvUtils.times
import io.ksmt.utils.BvUtils.zeroExtension
import io.ksmt.utils.cast
import io.ksmt.utils.toBigInteger
import io.ksmt.utils.uncheckedCast
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

        withExpressionsOrdered(lhs, rhs, ::mkEqNoSimplify)
    }

    fun <T : KBvSort> areDefinitelyDistinctBv(lhs: KExpr<T>, rhs: KExpr<T>): Boolean {
        if (lhs is KBitVecValue<T> && rhs is KBitVecValue<T>) {
            return lhs != rhs
        }
        return false
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            simplifyBvUnsignedLessOrEqualExpr(lhs, rhs)
        }

    override fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            simplifyBvSignedLessOrEqualExpr(lhs, rhs)
        }

    // (uge a b) ==> (ule b a)
    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            preprocess = { KBvUnsignedLessOrEqualExpr(this, expr.arg1, expr.arg0) }
        )

    // (ult a b) ==> (not (ule b a))
    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            preprocess = {
                val ule = KBvUnsignedLessOrEqualExpr(this, expr.arg1, expr.arg0)
                KNotExpr(this, ule)
            }
        )

    // (ugt a b) ==> (not (ule a b))
    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            preprocess = {
                val ule = KBvUnsignedLessOrEqualExpr(this, expr.arg0, expr.arg1)
                KNotExpr(this, ule)
            }
        )

    // (sge a b) ==> (sle b a)
    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            preprocess = { KBvSignedLessOrEqualExpr(this, expr.arg1, expr.arg0) }
        )

    // (slt a b) ==> (not (sle b a))
    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            preprocess = {
                val sle = KBvSignedLessOrEqualExpr(this, expr.arg1, expr.arg0)
                KNotExpr(this, sle)
            }
        )

    // (sgt a b) ==> (not (sle a b))
    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            preprocess = {
                val sle = KBvSignedLessOrEqualExpr(this, expr.arg0, expr.arg1)
                KNotExpr(this, sle)
            }
        )


    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            preprocess = { flatBvAdd(expr) }
        )

    private fun <T : KBvSort> transform(expr: SimplifierFlatBvAddExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.args) { flatten ->
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
                return@simplifyExpr constantValue.uncheckedCast()
            }

            if (constantValue != zero) {
                resultParts.add(constantValue.uncheckedCast())
            }

            resultParts.reduceBinaryBvExpr(::mkBvAddExprNoSimplify)
        }

    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            preprocess = {
                val negativeRhs = KBvNegationExpr(this, expr.arg1)
                KBvAddExpr(this, expr.arg0, negativeRhs)
            }
        )

    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            preprocess = { flatBvMul(expr) }
        )

    private fun <T : KBvSort> transform(expr: SimplifierFlatBvMulExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.args) { flatten ->
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
                return@simplifyExpr zero.uncheckedCast()
            }

            if (resultParts.isEmpty()) {
                return@simplifyExpr constantValue.uncheckedCast()
            }

            // (* 1 a) ==> a
            if (constantValue.isBvOne()) {
                return@simplifyExpr resultParts.reduceBinaryBvExpr(::mkBvMulExprNoSimplify)
            }

            // (* -1 a) ==> -a
            val minusOne = zero - one
            if (constantValue == minusOne) {
                val value = resultParts.reduceBinaryBvExpr(::mkBvMulExprNoSimplify)
                return@simplifyExpr mkBvNegationExprNoSimplify(value)
            }

            resultParts.add(constantValue.uncheckedCast())
            resultParts.reduceBinaryBvExpr(::mkBvMulExprNoSimplify)
        }

    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>): KExpr<T> = simplifyExpr(expr, expr.value) { arg ->
        simplifyBvNegationExpr(arg)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            simplifyBvSignedDivExpr(lhs, rhs)
        }

    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            simplifyBvUnsignedDivExpr(lhs, rhs)
        }

    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            simplifyBvSignedRemExpr(lhs, rhs)
        }

    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            simplifyBvUnsignedRemExpr(lhs, rhs)
        }

    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            simplifyBvSignedModExpr(lhs, rhs)
        }

    override fun <T : KBvSort> transform(expr: KBvNotExpr<T>): KExpr<T> = simplifyExpr(expr, expr.value) { arg ->
        // (bvnot (bvnot a)) ==> a
        if (arg is KBvNotExpr<T>) {
            return@simplifyExpr arg.value
        }

        if (arg is KBitVecValue<T>) {
            return@simplifyExpr arg.bitwiseNot().uncheckedCast()
        }

        // (bvnot (concat a b)) ==> (concat (bvnot a) (bvnot b))
        if (arg is KBvConcatExpr && canPerformBoundedRewrite()) {
            val concatParts = flatConcat(arg).args
            return@simplifyExpr boundedRewrite(
                auxExpr {
                    val negatedParts = concatParts.map { KBvNotExpr(ctx, it) }
                    SimplifierFlatBvConcatExpr(ctx, arg.sort, negatedParts).uncheckedCast()
                }
            )
        }

        // (bvnot (ite c a b)) ==> (ite c (bvnot a) (bvnot b))
        if (arg is KIteExpr<T> && canPerformBoundedRewrite()) {
            if (arg.trueBranch is KBitVecValue<T> || arg.falseBranch is KBitVecValue<T>) {
                return@simplifyExpr boundedRewrite(
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
        simplifyExpr(
            expr = expr,
            preprocess = { flatBvOr(expr) }
        )

    private fun <T : KBvSort> transform(expr: SimplifierFlatBvOrExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.args) { flatten ->
            simplifyBvAndOr(
                args = flatten,
                // (bvor a b 0x0000...) ==> (bvor a b)
                neutralElement = bvZero(expr.sort.sizeBits).uncheckedCast(),
                // (bvor a b 0xFFFF...) ==> 0xFFFF...
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
                    return@simplifyExpr boundedRewrite(distributeOrOverConcat(resultParts))
                }

                resultParts.reduceBinaryBvExpr(::mkBvOrExprNoSimplify)
            }
        }

    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            preprocess = { flatBvXor(expr) }
        )

    private fun <T : KBvSort> transform(expr: SimplifierFlatBvXorExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.args) { flatten ->
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
            return@simplifyExpr constantValue.uncheckedCast()
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
            return@simplifyExpr boundedRewrite(result)
        }

            val preResult = resultParts.reduceBinaryBvExpr(::mkBvXorExprNoSimplify)
            return@simplifyExpr if (negateResult) mkBvNotExprNoSimplify(preResult) else preResult
        }

    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            preprocess = { flatBvAnd(expr) }
        )

    private fun <T : KBvSort> transform(expr: SimplifierFlatBvAndExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.args) { flatten ->
            simplifyBvAndOr(
                args = flatten,
                // (bvand a b 0xFFFF...) ==> (bvand a b)
                neutralElement = bvMaxValueUnsigned(expr.sort.sizeBits).uncheckedCast(),
                // (bvand a b 0x0000...) ==> 0x0000...
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
                    return@simplifyExpr boundedRewrite(distributeAndOverConcat(resultParts))
                }

                resultParts.reduceBinaryBvExpr(::mkBvAndExprNoSimplify)
            }
    }

    // (bvnand a b) ==> (bvor (bvnot a) (bvnot b))
    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            preprocess = {
                KBvOrExpr(
                    this,
                    KBvNotExpr(this, expr.arg0),
                    KBvNotExpr(this, expr.arg1)
                )
            }
        )

    // (bvnor a b) ==> (bvnot (bvor a b))
    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            preprocess = { KBvNotExpr(this, KBvOrExpr(this, expr.arg0, expr.arg1)) }
        )

    // (bvxnor a b) ==> (bvnot (bvxor a b))
    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            preprocess = { KBvNotExpr(this, KBvXorExpr(this, expr.arg0, expr.arg1)) }
        )

    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>): KExpr<KBv1Sort> =
        simplifyExpr(expr, expr.value) { arg ->
            simplifyBvReductionAndExpr(arg)
        }

    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>): KExpr<KBv1Sort> =
        simplifyExpr(expr, expr.value) { arg ->
            simplifyBvReductionOrExpr(arg)
        }

    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> =
        simplifyExpr(
            expr = expr,
            preprocess = { flatConcat(expr) }
        )

    @Suppress("LoopWithTooManyJumpStatements")
    private fun transform(expr: SimplifierFlatBvConcatExpr): KExpr<KBvSort> =
        simplifyExpr(expr, expr.args) { flatten ->
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

    override fun transform(expr: KBvExtractExpr): KExpr<KBvSort> = simplifyExpr(expr, expr.value) { arg ->
        // (extract [size-1:0] x) ==> x
        if (expr.low == 0 && expr.high == arg.sort.sizeBits.toInt() - 1) {
            return@simplifyExpr arg
        }

        if (arg is KBitVecValue<*>) {
            return@simplifyExpr arg.extractBv(expr.high, expr.low).uncheckedCast()
        }

        // (extract[high:low] (extract[_:nestedLow] x)) ==> (extract[high+nestedLow : low+nestedLow] x)
        if (arg is KBvExtractExpr) {
            val nestedLow = arg.low
            return@simplifyExpr rewrite(
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
            return@simplifyExpr boundedRewrite(simplified)
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

    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.arg, expr.shift) { arg, shift ->
            val size = expr.sort.sizeBits
            val argValue = arg as? KBitVecValue<T>
            val shiftValue = shift as? KBitVecValue<T>

            if (shiftValue != null) {
                // (x << 0) ==> x
                if (shiftValue.isBvZero()) {
                    return@simplifyExpr arg
                }

                // (x << shift), shift >= size ==> 0
                if (shiftValue.signedGreaterOrEqual(size.toInt())) {
                    return@simplifyExpr bvZero(size).uncheckedCast()
                }

                if (argValue != null) {
                    return@simplifyExpr argValue.shiftLeft(shiftValue).uncheckedCast()
                }

                // (bvshl x shift) ==> (concat (extract [size-1-shift:0] x) 0.[shift].0)
                val intShiftValue = shiftValue.bigIntValue()
                if (intShiftValue >= BigInteger.ZERO && intShiftValue <= Int.MAX_VALUE.toBigInteger()) {
                    return@simplifyExpr rewrite(
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
                return@simplifyExpr rewrite(
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
        simplifyExpr(expr, expr.arg, expr.shift) { arg, shift ->
            val size = expr.sort.sizeBits
            val argValue = arg as? KBitVecValue<T>
            val shiftValue = shift as? KBitVecValue<T>

            if (shiftValue != null) {
                // (x >>> 0) ==> x
                if (shiftValue.isBvZero()) {
                    return@simplifyExpr arg
                }

                // (x >>> shift), shift >= size ==> 0
                if (shiftValue.signedGreaterOrEqual(size.toInt())) {
                    return@simplifyExpr bvZero(size).uncheckedCast()
                }

                if (argValue != null) {
                    return@simplifyExpr argValue.shiftRightLogical(shiftValue).uncheckedCast()
                }

                // (bvlshr x shift) ==> (concat 0.[shift].0 (extract [size-1:shift] x))
                val intShiftValue = shiftValue.bigIntValue()
                if (intShiftValue >= BigInteger.ZERO && intShiftValue <= Int.MAX_VALUE.toBigInteger()) {
                    return@simplifyExpr rewrite(
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
                return@simplifyExpr bvZero(size).uncheckedCast()
            }

            mkBvLogicalShiftRightExprNoSimplify(arg, shift)
        }

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.arg, expr.shift) { arg, shift ->
            simplifyBvArithShiftRightExpr(arg, shift)
        }

    // (repeat a x) ==> (concat a a ..[x].. a)
    override fun transform(expr: KBvRepeatExpr): KExpr<KBvSort> = simplifyExpr(expr, expr.value) { arg ->
        val repeats = arrayListOf<KExpr<KBvSort>>()
        repeat(expr.repeatNumber) {
            repeats += arg
        }

        if (repeats.size == 0) {
            return@simplifyExpr mkBvRepeatExprNoSimplify(expr.repeatNumber, arg)
        }

        return@simplifyExpr rewrite(SimplifierFlatBvConcatExpr(ctx, expr.sort, repeats))
    }

    // (zeroext a) ==> (concat 0 a)
    override fun transform(expr: KBvZeroExtensionExpr): KExpr<KBvSort> = simplifyExpr(expr, expr.value) { arg ->
        if (expr.extensionSize == 0) {
            return@simplifyExpr arg
        }

        if (arg is KBitVecValue<*>) {
            return@simplifyExpr arg.zeroExtension(expr.extensionSize.toUInt()).uncheckedCast()
        }

        return@simplifyExpr rewrite(
            auxExpr {
                KBvConcatExpr(ctx, bvZero(expr.extensionSize.toUInt()).uncheckedCast(), arg)
            }
        )
    }

    override fun transform(expr: KBvSignExtensionExpr): KExpr<KBvSort> = simplifyExpr(expr, expr.value) { arg ->
        if (expr.extensionSize == 0) {
            return@simplifyExpr arg
        }

        if (arg is KBitVecValue<*>) {
            return@simplifyExpr arg.signExtension(expr.extensionSize.toUInt()).uncheckedCast()
        }

        return@simplifyExpr mkBvSignExtensionExprNoSimplify(expr.extensionSize, arg)
    }

    // (rotateLeft a x) ==> (concat (extract [size-1-x:0] a) (extract [size-1:size-x] a))
    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.value) { arg ->
            simplifyBvRotateLeftIndexedExpr(expr.rotationNumber, arg)
        }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.arg, expr.rotation) { arg, rotation ->
            simplifyBvRotateLeftExpr(arg, rotation)
        }

    // (rotateRight a x) ==> (rotateLeft a (- size x))
    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.value) { arg ->
            simplifyBvRotateRightIndexedExpr(expr.rotationNumber, arg)
        }

    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.arg, expr.rotation) { arg, rotation ->
            simplifyBvRotateRightExpr(arg, rotation)
        }

    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            rewriteBvAddNoOverflowExpr(lhs, rhs, expr.isSigned)
        }

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            rewriteBvAddNoUnderflowExpr(lhs, rhs)
        }

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            rewriteBvSubNoOverflowExpr(lhs, rhs)
        }

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            rewriteBvSubNoUnderflowExpr(lhs, rhs, expr.isSigned)
        }

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.value) { arg ->
            rewriteBvNegNoOverflowExpr(arg)
        }

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            rewriteBvDivNoOverflowExpr(lhs, rhs)
        }

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            rewriteBvMulNoOverflowExpr(lhs, rhs, expr.isSigned)
        }

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.arg0, expr.arg1) { lhs, rhs ->
            rewriteBvMulNoUnderflowExpr(lhs, rhs)
        }

    override fun transform(expr: KBv2IntExpr): KExpr<KIntSort> = simplifyExpr(expr, expr.value) { arg ->
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

        ctx.mkAndAuxExpr(result)
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

    @Suppress("LoopWithTooManyJumpStatements")
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
