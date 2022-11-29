package org.ksmt.expr.rewrite.simplify

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
import org.ksmt.expr.KIteExpr
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KSort
import org.ksmt.utils.asExpr
import org.ksmt.utils.cast
import java.math.BigInteger
import kotlin.experimental.inv
import kotlin.experimental.or
import kotlin.experimental.xor

@Suppress(
    "LargeClass",
    "LongMethod",
    "LongParameterList",
    "ComplexMethod"
)
interface KBvExprSimplifier : KExprSimplifierBase {

    fun <T : KBvSort> simplifyEqBv(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> = with(ctx) {
        if (lhs == rhs) return trueExpr

        if (lhs is KBitVecValue<*> && rhs is KBitVecValue<*>) {
            return falseExpr
        }

        if (lhs is KBvConcatExpr || rhs is KBvConcatExpr) {
            return simplifyBvConcatEq(lhs, rhs).also { rewrite(it) }
        }

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
                        return (lhs eq rhs.asExpr(lhs.sort)).also { rewrite(it) }
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
                        return (lhs.asExpr(lhs.sort) eq rhs).also { rewrite(it) }
                    }
                }
            }

            if (signed) {
                mkBvSignedLessOrEqualExpr(lhs, rhs)
            } else {
                mkBvUnsignedLessOrEqualExpr(lhs, rhs)
            }
        }

    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val lhsValue = lhs as? KBitVecValue<*>
        val rhsValue = rhs as? KBitVecValue<*>

        if (lhsValue != null && rhsValue != null) {
            return@simplifyApp (lhsValue + rhsValue).asExpr(expr.sort)
        }

        if (lhsValue != null || rhsValue != null) {
            // (+ const1 (+ const2 x)) ==> (+ (+ const1 const2) x)
            if (lhs is KBvAddExpr<*> || rhs is KBvAddExpr<*>) {
                val flatten = flatBvAdd(lhs) + flatBvAdd(rhs)

                val zero = zero(expr.sort.sizeBits)
                var constantValue = zero
                val resultParts = arrayListOf<KExpr<KBvSort>>()

                for (arg in flatten) {
                    if (arg is KBitVecValue<*>) {
                        constantValue += arg
                        continue
                    }
                    resultParts += arg
                }

                if (constantValue != zero) {
                    resultParts.add(constantValue.asExpr(expr.sort))
                }

                if (resultParts.size < flatten.size) {
                    return@simplifyApp resultParts.reduce(::mkBvAddExpr).asExpr(expr.sort)
                }
            }
        }

        mkBvAddExpr(lhs, rhs)
    }

    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>): KExpr<T> =
        simplifyApp(
            expr = expr,
            preprocess = { mkBvAddExpr(expr.arg0, mkBvNegationExpr(expr.arg1)) }
        ) {
            error("Always preprocessed")
        }

    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val lhsValue = lhs as? KBitVecValue<*>
        val rhsValue = rhs as? KBitVecValue<*>

        if (lhsValue != null && rhsValue != null) {
            return@simplifyApp (lhsValue * rhsValue).asExpr(expr.sort)
        }

        if (lhsValue != null || rhsValue != null) {
            // (* const1 (* const2 x)) ==> (* (* const1 const2) x)
            if (lhs is KBvMulExpr<*> || rhs is KBvMulExpr<*>) {
                val flatten = flatBvMul(lhs) + flatBvMul(rhs)

                val zero = zero(expr.sort.sizeBits)
                val one = one(expr.sort.sizeBits)

                var constantValue = one
                val resultParts = arrayListOf<KExpr<KBvSort>>()

                for (arg in flatten) {
                    if (arg is KBitVecValue<*>) {
                        constantValue *= arg
                        continue
                    }
                    resultParts += arg
                }

                // (* 0 a) ==> 0
                if (constantValue == zero) {
                    return@simplifyApp zero.asExpr(expr.sort)
                }

                if (resultParts.isEmpty()) {
                    return@simplifyApp constantValue.asExpr(expr.sort)
                }

                // (* 1 a) ==> a
                if (constantValue == one) {
                    return@simplifyApp resultParts.reduce(::mkBvMulExpr).asExpr(expr.sort)
                }

                // (* -1 a) ==> -a
                val minusOne = zero - one
                if (constantValue == minusOne) {
                    val value = resultParts.reduce(::mkBvMulExpr).asExpr(expr.sort)
                    return@simplifyApp mkBvNegationExpr(value).also { rewrite(it) }
                }

                resultParts.add(constantValue.asExpr(expr.sort))
                if (resultParts.size < flatten.size) {
                    return@simplifyApp resultParts.reduce(::mkBvMulExpr).asExpr(expr.sort)
                }
            }
        }

        mkBvMulExpr(lhs, rhs)
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
                    .also { rewrite(it) }
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
                ).asExpr(lhs.sort).also { rewrite(it) }
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

    override fun <T : KBvSort> transform(expr: KBvNotExpr<T>): KExpr<T> = simplifyApp(expr) { (arg) ->
        // (bvnot (bvnot a)) ==> a
        if (arg is KBvNotExpr<*>) {
            return@simplifyApp arg.value.asExpr(expr.sort)
        }

        if (arg is KBitVecValue<*>) {
            return@simplifyApp arg.bitwiseNot().asExpr(expr.sort)
        }

        // (bvnot (concat a b)) ==> (concat (bvnot a) (bvnot b))
        if (arg is KBvConcatExpr) {
            val concatParts = flatConcat(arg)
            val negatedParts = concatParts.map { mkBvNotExpr(it) }
            return@simplifyApp negatedParts.reduceRight(::mkBvConcatExpr)
                .asExpr(expr.sort).also { rewrite(it) }
        }

        // (bvnot (ite c a b)) ==> (ite c (bvnot a) (bvnot b))
        if (arg is KIteExpr<*>) {
            val trueValue = arg.trueBranch as? KBitVecValue<*>
            val falseValue = arg.falseBranch as? KBitVecValue<*>
            if (trueValue != null || falseValue != null) {
                return@simplifyApp mkIte(
                    condition = arg.condition,
                    trueBranch = mkBvNotExpr(arg.trueBranch.asExpr(expr.sort)),
                    falseBranch = mkBvNotExpr(arg.falseBranch.asExpr(expr.sort))
                ).also { rewrite(it) }
            }
        }

        mkBvNotExpr(arg)
    }

    @Suppress("LoopWithTooManyJumpStatements")
    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val size = expr.sort.sizeBits
        val lhsValue = lhs as? KBitVecValue<*>
        val rhsValue = rhs as? KBitVecValue<*>

        if (lhsValue != null && rhsValue != null) {
            return@simplifyApp (lhsValue.bitwiseOr(rhsValue)).asExpr(expr.sort)
        }

        if (lhsValue != null || rhsValue != null) {
            // (bvor const1 (bvor const2 x)) ==> (bvor (bvor const1 const2) x)
            if (lhs is KBvOrExpr<*> || rhs is KBvOrExpr<*>) {
                val flatten = flatBvOr(lhs) + flatBvOr(rhs)

                val zero = zero(expr.sort.sizeBits)
                val maxValue = maxValueUnsigned(expr.sort.sizeBits)
                var constantValue = zero
                val resultParts = arrayListOf<KExpr<KBvSort>>()
                val positiveTerms = hashSetOf<KExpr<*>>()
                val negativeTerms = hashSetOf<KExpr<*>>()

                for (arg in flatten) {
                    if (arg is KBitVecValue<*>) {
                        constantValue = constantValue.bitwiseOr(arg)
                        continue
                    }

                    if (arg is KBvNotExpr<*>) {
                        val term = arg.value
                        // (bvor a b a) ==> (bvor a b)
                        if (!negativeTerms.add(term)) {
                            continue
                        }

                        // (bvor a (bvnot a)) ==> 0xFFFF...
                        if (term in positiveTerms) {
                            return@simplifyApp maxValue.asExpr(expr.sort)
                        }
                    } else {
                        // (bvor a b a) ==> (bvor a b)
                        if (!positiveTerms.add(arg)) {
                            continue
                        }

                        // (bvor a (bvnot a)) ==> 0xFFFF...
                        if (arg in negativeTerms) {
                            return@simplifyApp maxValue.asExpr(expr.sort)
                        }
                    }

                    resultParts += arg
                }

                // (bvor 0xFFFF... a) ==> 0xFFFF...
                if (constantValue == maxValue) {
                    return@simplifyApp maxValue.asExpr(expr.sort)
                }

                // (bvor 0 a) ==> a
                if (constantValue != zero) {
                    resultParts.add(constantValue.asExpr(expr.sort))
                }

                if (resultParts.size < flatten.size) {
                    return@simplifyApp resultParts.reduce(::mkBvOrExpr)
                        .asExpr(expr.sort).also { rewrite(it) }
                }
            }
        }

        // (bvor a a) ==> a
        if (lhs == rhs) {
            return@simplifyApp lhs
        }

        // (bvor (bvnot a) a) ==> 0xFFFF...
        if (lhs is KBvNotExpr<*> && lhs.value == rhs) {
            return@simplifyApp maxValueUnsigned(size).asExpr(expr.sort)
        }

        // (bvor a (bvnot a)) ==> 0xFFFF...
        if (rhs is KBvNotExpr<*> && rhs.value == lhs) {
            return@simplifyApp maxValueUnsigned(size).asExpr(expr.sort)
        }

        if (lhs is KBvConcatExpr || rhs is KBvConcatExpr) {
            return@simplifyApp distributeOrOverConcat(lhs, rhs)
                .asExpr(expr.sort).also { rewrite(it) }
        }

        mkBvOrExpr(lhs, rhs)
    }

    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>): KExpr<T> = simplifyApp(expr) { (lhs, rhs) ->
        val size = expr.sort.sizeBits
        val lhsValue = lhs as? KBitVecValue<*>
        val rhsValue = rhs as? KBitVecValue<*>

        if (lhsValue != null && rhsValue != null) {
            return@simplifyApp (lhsValue.bitwiseXor(rhsValue)).asExpr(expr.sort)
        }

        if (lhsValue != null || rhsValue != null) {
            // (bvxor const1 (bvxor const2 x)) ==> (bvxor (bvxor const1 const2) x)
            if (lhs is KBvOrExpr<*> || rhs is KBvOrExpr<*>) {
                val flatten = flatBvXor(lhs) + flatBvXor(rhs)

                val zero = zero(expr.sort.sizeBits)
                val maxValue = maxValueUnsigned(expr.sort.sizeBits)
                var constantValue = zero
                val resultParts = arrayListOf<KExpr<KBvSort>>()

                for (arg in flatten) {
                    if (arg is KBitVecValue<*>) {
                        constantValue = constantValue.bitwiseXor(arg)
                        continue
                    }
                    resultParts += arg
                }

                // (bvxor 0 a) ==> a
                if (constantValue == zero) {
                    return@simplifyApp resultParts.reduce(::mkBvXorExpr)
                        .asExpr(expr.sort).also { rewrite(it) }
                }

                // (bvxor 0xFFFF... a) ==> (bvnot a)
                if (constantValue == maxValue) {
                    val value = resultParts.reduce(::mkBvXorExpr).asExpr(expr.sort)
                    return@simplifyApp mkBvNotExpr(value).also { rewrite(it) }
                }

                resultParts.add(constantValue.asExpr(expr.sort))
                if (resultParts.size < flatten.size) {
                    return@simplifyApp resultParts.reduce(::mkBvXorExpr)
                        .asExpr(expr.sort).also { rewrite(it) }
                }
            }
        }

        // (bvxor a a) ==> 0
        if (lhs == rhs) {
            return@simplifyApp zero(size).asExpr(expr.sort)
        }

        // (bvxor (bvnot a) a) ==> 0xFFFF...
        if (lhs is KBvNotExpr<*> && lhs.value == rhs) {
            return@simplifyApp maxValueUnsigned(size).asExpr(expr.sort)
        }

        // (bvxor a (bvnot a)) ==> 0xFFFF...
        if (rhs is KBvNotExpr<*> && rhs.value == lhs) {
            return@simplifyApp maxValueUnsigned(size).asExpr(expr.sort)
        }

        if (lhs is KBvConcatExpr || rhs is KBvConcatExpr) {
            return@simplifyApp distributeXorOverConcat(lhs, rhs)
                .asExpr(expr.sort).also { rewrite(it) }
        }

        mkBvXorExpr(lhs, rhs)
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
        if (arg is KBitVecValue<*>) {
            val result = arg == maxValueUnsigned(arg.sort.sizeBits)
            return@simplifyApp mkBv(result)
        }
        mkBvReductionAndExpr(arg)
    }

    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>): KExpr<KBv1Sort> = simplifyApp(expr) { (arg) ->
        if (arg is KBitVecValue<*>) {
            val result = arg != zero(arg.sort.sizeBits)
            return@simplifyApp mkBv(result)
        }
        mkBvReductionOrExpr(arg)
    }

    @Suppress("LoopWithTooManyJumpStatements")
    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> = simplifyApp(expr) { (lhs, rhs) ->
        val lhsValue = lhs as? KBitVecValue<*>
        val rhsValue = rhs as? KBitVecValue<*>

        if (lhsValue != null && rhsValue != null) {
            return@simplifyApp concatBv(lhsValue, rhsValue).asExpr(expr.sort)
        }

        if (lhs is KBvConcatExpr || rhs is KBvConcatExpr) {
            val lhsParts = flatConcat(lhs)
            val rhsParts = flatConcat(rhs)
            val allParts = lhsParts + rhsParts
            val mergedParts = arrayListOf(allParts.first())
            for (part in allParts.drop(1)) {
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

            if (mergedParts.size < allParts.size) {
                return@simplifyApp mergedParts.reduceRight(::mkBvConcatExpr)
                    .also { rewrite(it) }
            }
        }

        // (concat (extract[h1, l1] a) (extract[h2, l2] a)), l1 == h2 + 1 ==> (extract[h1, l2] a)
        if (lhs is KBvExtractExpr && rhs is KBvExtractExpr) {
            tryMergeBvConcatExtract(lhs, rhs)?.let { return@simplifyApp rewrite(it) }
        }

        mkBvConcatExpr(lhs, rhs)
    }

    override fun transform(expr: KBvExtractExpr): KExpr<KBvSort> = simplifyApp(expr) { (arg) ->
        // (extract [size-1:0] x) ==> x
        if (expr.low == 0 && expr.high == arg.sort.sizeBits.toInt() - 1) {
            return@simplifyApp arg
        }

        if (arg is KBitVecValue<*>) {
            return@simplifyApp extractBv(arg, expr.high, expr.low).asExpr(expr.sort)
        }

        // (extract[high:low] (extract[_:nestedLow] x)) ==> (extract[high+nestedLow : low+nestedLow] x)
        if (arg is KBvExtractExpr) {
            val nestedLow = arg.low
            return@simplifyApp mkBvExtractExpr(
                high = expr.high + nestedLow,
                low = expr.low + nestedLow,
                value = arg.value
            ).also { rewrite(it) }
        }

        // (extract (concat a b)) ==> (concat (extract a) (extract b))
        if (arg is KBvConcatExpr) {
            return@simplifyApp distributeExtractOverConcat(arg, expr.high, expr.low)
                .also { rewrite(it) }
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
            return@simplifyApp simplified.also { rewrite(it) }
        }

        mkBvExtractExpr(expr.high, expr.low, arg)
    }

    // (extract (concat a b)) ==> (concat (extract a) (extract b))
    @Suppress("LoopWithTooManyJumpStatements", "NestedBlockDepth")
    private fun distributeExtractOverConcat(
        concatenation: KBvConcatExpr,
        high: Int,
        low: Int
    ): KExpr<KBvSort> = with(ctx) {
        val parts = flatConcat(concatenation)

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
                return if (idx == low && high - idx == firstPartSize) {
                    firstPart
                } else {
                    mkBvExtractExpr(
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
            partsToExtractFrom += mkBvExtractExpr(
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
                    partsToExtractFrom += mkBvExtractExpr(
                        high = partSize - 1,
                        low = low - idx,
                        value = part
                    )
                    break
                }
            }
        }

        return partsToExtractFrom.reduceRight(::mkBvConcatExpr)
    }

    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>): KExpr<T> = simplifyApp(expr) { (arg, shift) ->
        val size = expr.sort.sizeBits
        val argValue = arg as? KBitVecValue<*>
        val shiftValue = shift as? KBitVecValue<*>

        if (shiftValue != null) {
            // (x << 0) ==> x
            if (shiftValue == zero(size)) {
                return@simplifyApp arg
            }

            // (x << shift), shift >= size ==> 0
            if (shiftValue.signedGreaterOrEqual(size.toInt())) {
                return@simplifyApp zero(size).asExpr(expr.sort)
            }

            if (argValue != null) {
                return@simplifyApp argValue.shiftLeft(shiftValue).asExpr(expr.sort)
            }

            // (bvshl x shift) ==> (concat (extract [size-1-shift:0] x) 0.[shift].0)
            val intShiftValue = shiftValue.intValueOrNull()
            if (intShiftValue != null) {
                val lhs = mkBvExtractExpr(high = size.toInt() - 1 - intShiftValue, low = 0, arg)
                val rhs = zero(intShiftValue.toUInt())
                return@simplifyApp mkBvConcatExpr(lhs, rhs)
                    .asExpr(expr.sort).also { rewrite(it) }
            }
        }

        /**
         * (bvshl (bvshl x nestedShift) shift) ==>
         *      (ite (bvule nestedShift (+ nestedShift shift)) (bvshl x (+ nestedShift shift)) 0)
         * */
        if (arg is KBvShiftLeftExpr<*>) {
            val nestedArg = arg.arg0.asExpr(expr.sort)
            val nestedShift = arg.arg1.asExpr(expr.sort)
            val sum = mkBvAddExpr(nestedShift, shift)
            val cond = mkBvUnsignedLessOrEqualExpr(nestedShift, sum)
            return@simplifyApp mkIte(
                condition = cond,
                trueBranch = mkBvShiftLeftExpr(nestedArg, sum),
                falseBranch = zero(size).asExpr(expr.sort)
            ).also { rewrite(it) }
        }

        mkBvShiftLeftExpr(arg, shift)
    }

    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>): KExpr<T> =
        simplifyApp(expr) { (arg, shift) ->
            val size = expr.sort.sizeBits
            val argValue = arg as? KBitVecValue<*>
            val shiftValue = shift as? KBitVecValue<*>

            if (shiftValue != null) {
                // (x >>> 0) ==> x
                if (shiftValue == zero(size)) {
                    return@simplifyApp arg
                }

                // (x >>> shift), shift >= size ==> 0
                if (shiftValue.signedGreaterOrEqual(size.toInt())) {
                    return@simplifyApp zero(size).asExpr(expr.sort)
                }

                if (argValue != null) {
                    return@simplifyApp argValue.shiftRightLogical(shiftValue).asExpr(expr.sort)
                }

                // (bvlshr x shift) ==> (concat 0.[shift].0 (extract [size-1:shift] x))
                val intShiftValue = shiftValue.intValueOrNull()
                if (intShiftValue != null) {
                    val lhs = zero(intShiftValue.toUInt())
                    val rhs = mkBvExtractExpr(high = size.toInt() - 1, low = intShiftValue, arg)
                    return@simplifyApp mkBvConcatExpr(lhs, rhs)
                        .asExpr(expr.sort).also { rewrite(it) }
                }
            }

            // (x >>> x) ==> 0
            if (arg == shift) {
                return@simplifyApp zero(size).asExpr(expr.sort)
            }

            mkBvLogicalShiftRightExpr(arg, shift)
        }

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>): KExpr<T> =
        simplifyApp(expr) { (arg, shift) ->
            val size = expr.sort.sizeBits
            val argValue = arg as? KBitVecValue<*>
            val shiftValue = shift as? KBitVecValue<*>

            if (shiftValue != null) {
                // (x >> 0) ==> x
                if (shiftValue == zero(size)) {
                    return@simplifyApp arg
                }

                if (argValue != null) {
                    return@simplifyApp argValue.shiftRightArith(shiftValue).asExpr(expr.sort)
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

        return@simplifyApp repeats.reduce(::mkBvConcatExpr).also { rewrite(it) }
    }

    // (zeroext a) ==> (concat 0 a)
    override fun transform(expr: KBvZeroExtensionExpr): KExpr<KBvSort> = simplifyApp(expr) { (arg) ->
        if (expr.extensionSize == 0) {
            return@simplifyApp arg
        }

        val extension = zero(expr.extensionSize.toUInt())
        return@simplifyApp mkBvConcatExpr(extension, arg).also { rewrite(it) }
    }

    override fun transform(expr: KBvSignExtensionExpr): KExpr<KBvSort> = simplifyApp(expr) { (arg) ->
        if (expr.extensionSize == 0) {
            return@simplifyApp arg
        }

        if (arg is KBitVecValue<*>) {
            return@simplifyApp arg.signExtension(expr.extensionSize).asExpr(expr.sort)
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

        val lhs = mkBvExtractExpr(high = size - rotation - 1, low = 0, arg)
        val rhs = mkBvExtractExpr(high = size - 1, low = size - rotation, arg)

        return mkBvConcatExpr(lhs, rhs).asExpr(arg.sort).also { rewrite(it) }
    }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>): KExpr<T> = simplifyApp(expr) { (arg, rotation) ->
        if (rotation is KBitVecValue<*>) {
            val intValue = rotation.intValueOrNull()
            if (intValue != null) {
                return@simplifyApp mkBvRotateLeftIndexedExpr(intValue, arg).also { rewrite(it) }
            }
        }
        return@simplifyApp mkBvRotateLeftExpr(arg, rotation)
    }

    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>): KExpr<T> = simplifyApp(expr) { (arg, rotation) ->
        if (rotation is KBitVecValue<*>) {
            val intValue = rotation.intValueOrNull()
            if (intValue != null) {
                return@simplifyApp mkBvRotateRightIndexedExpr(intValue, arg).also { rewrite(it) }
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

                val zero = zero(lhs.sort.sizeBits).asExpr(lhs.sort)
                val zeroSltA = mkBvSignedLessExpr(zero, lhs)
                val zeroSltB = mkBvSignedLessExpr(zero, rhs)
                val sum = mkBvAddExpr(lhs, rhs)
                val zeroSltSum = mkBvSignedLessExpr(zero, sum)

                return@simplifyApp mkImplies(zeroSltA and zeroSltB, zeroSltSum).also { rewrite(it) }
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

                return@simplifyApp (zeroBit eq sumFirstBit.asExpr(bv1Sort)).also { rewrite(it) }
            }
        }

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            /**
             * (bvadd no udf a b) ==>
             *    (=> (and (bvslt a 0) (bvslt b 0)) (bvslt (bvadd a b) 0))
             * */

            val zero = zero(lhs.sort.sizeBits).asExpr(lhs.sort)
            val aLtZero = mkBvSignedLessExpr(lhs, zero)
            val bLtZero = mkBvSignedLessExpr(rhs, zero)
            val sum = mkBvAddExpr(lhs, rhs)
            val sumLtZero = mkBvSignedLessExpr(sum, zero)

            return@simplifyApp mkImplies(aLtZero and bLtZero, sumLtZero).also { rewrite(it) }
        }

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            /**
             * (bvsub no ovf a b) ==>
             *     (ite (= b MIN_VALUE) (bvslt a 0) (bvadd no ovf signed a (bvneg b)))
             * */

            val zero = zero(lhs.sort.sizeBits).asExpr(lhs.sort)
            val minValue = minValueSigned(lhs.sort.sizeBits).asExpr(lhs.sort)

            val minusB = mkBvNegationExpr(rhs)
            val bIsMin = rhs eq minValue
            val aLtZero = mkBvSignedLessExpr(lhs, zero)
            val noOverflow = mkBvAddNoOverflowExpr(lhs, minusB, isSigned = true)

            return@simplifyApp mkIte(bIsMin, aLtZero, noOverflow).also { rewrite(it) }
        }

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (lhs, rhs) ->
            if (expr.isSigned) {
                /**
                 * (bvsub no udf signed a b) ==>
                 *    (=> (bvslt 0 b) (bvadd no udf (bvneg b)))
                 * */
                val zero = zero(lhs.sort.sizeBits).asExpr(lhs.sort)
                val minusB = mkBvNegationExpr(rhs)
                val zeroLtB = mkBvSignedLessExpr(zero, rhs)
                val noOverflow = mkBvAddNoUnderflowExpr(lhs, minusB)

                return@simplifyApp mkImplies(zeroLtB, noOverflow).also { rewrite(it) }
            } else {
                /**
                 * (bvsub no udf unsigned a b) ==>
                 *    (bvule b a)
                 * */
                return@simplifyApp mkBvUnsignedLessOrEqualExpr(rhs, lhs).also { rewrite(it) }
            }
        }

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>): KExpr<KBoolSort> =
        simplifyApp(expr) { (arg) ->
            /**
             * (bvneg no ovf a) ==> (not (= a MIN_VALUE))
             * */
            val minValue = minValueSigned(arg.sort.sizeBits).asExpr(arg.sort)
            return@simplifyApp !(arg eq minValue).also { rewrite(it) }
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
                mkBvConcatExpr(mkBv(true), zero(size - 1u))
            }.asExpr(lhs.sort)
            val minusOne = (zero(size) - one(size)).asExpr(lhs.sort)

            val aIsMsb = lhs eq mostSignificantBit
            val bIsMinusOne = rhs eq minusOne
            return@simplifyApp !(aIsMsb and bIsMinusOne).also { rewrite(it) }
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
        val lhsValue = lhs as? KBitVecValue<*>
        val rhsValue = rhs as? KBitVecValue<*>
        val zero = zero(size)
        val one = one(size)

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
                    val maxValue = maxValueSigned(size)
                    val limit = maxValue.signedDivide(lhsValue)
                    return rhsValue.signedLessOrEqual(limit - one).expr
                }
                // lhs > 0 && rhs > 0
                !lhsSign && !rhsSign -> {
                    // no underflow possible
                    if (!isOverflow) return trueExpr
                    // overflow if MAX_VALUE / rhs <= lhs - 1
                    val maxValue = maxValueSigned(size)
                    val limit = maxValue.signedDivide(rhsValue)
                    return limit.signedLessOrEqual(lhsValue - one).expr
                }
                // lhs < 0 && rhs > 0
                lhsSign && !rhsSign -> {
                    // no overflow possible
                    if (isOverflow) return trueExpr
                    // underflow if lhs <= MIN_VALUE / rhs - 1
                    val minValue = minValueSigned(size)
                    val limit = minValue.signedDivide(rhsValue)
                    return lhsValue.signedLessOrEqual(limit - one).expr
                }
                // lhs > 0 && rhs < 0
                else -> {
                    // no overflow possible
                    if (isOverflow) return trueExpr
                    // underflow if rhs <= MIN_VALUE / lhs - 1
                    val minValue = minValueSigned(size)
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
        val lhsValue = lhs as? KBitVecValue<*>
        val rhsValue = rhs as? KBitVecValue<*>
        val zero = zero(size)
        val one = one(size)

        if (lhsValue != null && (lhsValue == zero || (lhsValue == one))) {
            return trueExpr
        }

        if (rhsValue != null && (rhsValue == zero || (rhsValue == one))) {
            return trueExpr
        }

        if (lhsValue != null && rhsValue != null) {
            val longLhs = concatBv(zero, lhsValue)
            val longRhs = concatBv(zero, rhsValue)
            val longMaxValue = concatBv(zero, maxValueUnsigned(size))

            val product = longLhs * longRhs
            return product.signedLessOrEqual(longMaxValue).expr
        }

        return null
    }

    override fun transform(expr: KBv2IntExpr): KExpr<KIntSort> = simplifyApp(expr) { (arg) ->
        if (arg is KBitVecValue<*>) {
            val integerValue = if (expr.isSigned) {
                signedBigIntFromBinary(arg.stringValue)
            } else {
                unsignedBigIntFromBinary(arg.stringValue)
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
        val newEqualities = distributeOperationOverConcat(l, r) { a, b -> a eq b }
        return mkAnd(newEqualities)
    }

    /**
     * (bvor (concat a b) c) ==>
     *  (concat
     *      (bvor (extract (0, <a_size>) c))
     *      (bvor b (extract (<a_size>, <a_size> + <b_size>) c))
     *  )
     * */
    fun <T : KBvSort> distributeOrOverConcat(l: KExpr<T>, r: KExpr<T>): KExpr<T> = with(ctx) {
        val concatParts = distributeOperationOverConcat(l, r) { a, b -> mkBvOrExpr(a, b) }
        return concatParts.reduceRight(::mkBvConcatExpr).asExpr(l.sort)
    }

    /**
     * (bvxor (concat a b) c) ==>
     *  (concat
     *      (bvxor (extract (0, <a_size>) c))
     *      (bvxor b (extract (<a_size>, <a_size> + <b_size>) c))
     *  )
     * */
    fun <T : KBvSort> distributeXorOverConcat(l: KExpr<T>, r: KExpr<T>): KExpr<T> = with(ctx) {
        val concatParts = distributeOperationOverConcat(l, r) { a, b -> mkBvXorExpr(a, b) }
        return concatParts.reduceRight(::mkBvConcatExpr).asExpr(l.sort)
    }

    private inline fun <T : KBvSort, R : KSort> distributeOperationOverConcat(
        l: KExpr<T>,
        r: KExpr<T>,
        operation: (KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<R>
    ): List<KExpr<R>> = with(ctx) {
        val lArgs = flatConcat(l)
        val rArgs = flatConcat(r)
        val result = arrayListOf<KExpr<R>>()
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
                    result += operation(newL, newR)
                    lowL = 0
                    lowR = 0
                    lIdx--
                    rIdx--
                }

                remainSizeL < remainSizeR -> {
                    val newL = mkBvExtractExpr(high = lSize - 1, low = lowL, value = lArg)
                    val newR = mkBvExtractExpr(high = remainSizeL + lowR - 1, low = lowR, value = rArg)
                    result += operation(newL, newR)
                    lowL = 0
                    lowR += remainSizeL
                    lIdx--
                }

                else -> {
                    val newL = mkBvExtractExpr(high = remainSizeR + lowL - 1, low = lowL, value = lArg)
                    val newR = mkBvExtractExpr(high = rSize - 1, low = lowR, value = rArg)
                    result += operation(newL, newR)
                    lowL += remainSizeR
                    lowR = 0
                    rIdx--
                }
            }
        }

        // restore concat order
        result.reverse()

        return result
    }

    @Suppress("UNCHECKED_CAST")
    private fun <S : KBvSort> flatConcat(expr: KExpr<S>): List<KExpr<KBvSort>> =
        flatBinaryBvExpr<KBvConcatExpr>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 },
            getRhs = { it.arg1 }
        )

    @Suppress("UNCHECKED_CAST")
    private fun <S : KBvSort> flatBvAdd(expr: KExpr<S>): List<KExpr<KBvSort>> =
        flatBinaryBvExpr<KBvAddExpr<*>>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 as KExpr<KBvSort> },
            getRhs = { it.arg1 as KExpr<KBvSort> }
        )

    @Suppress("UNCHECKED_CAST")
    private fun <S : KBvSort> flatBvMul(expr: KExpr<S>): List<KExpr<KBvSort>> =
        flatBinaryBvExpr<KBvMulExpr<*>>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 as KExpr<KBvSort> },
            getRhs = { it.arg1 as KExpr<KBvSort> }
        )

    @Suppress("UNCHECKED_CAST")
    private fun <S : KBvSort> flatBvOr(expr: KExpr<S>): List<KExpr<KBvSort>> =
        flatBinaryBvExpr<KBvOrExpr<*>>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 as KExpr<KBvSort> },
            getRhs = { it.arg1 as KExpr<KBvSort> }
        )

    @Suppress("UNCHECKED_CAST")
    private fun <S : KBvSort> flatBvXor(expr: KExpr<S>): List<KExpr<KBvSort>> =
        flatBinaryBvExpr<KBvXorExpr<*>>(
            expr as KExpr<KBvSort>,
            getLhs = { it.arg0 as KExpr<KBvSort> },
            getRhs = { it.arg1 as KExpr<KBvSort> }
        )

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

    fun minValueSigned(size: UInt): KBitVecValue<*> = with(ctx) {
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

    fun maxValueSigned(size: UInt): KBitVecValue<*> = with(ctx) {
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

    fun maxValueUnsigned(size: UInt): KBitVecValue<*> = with(ctx) {
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

    private fun zero(size: UInt): KBitVecValue<*> = bvValue(size, 0)
    private fun one(size: UInt): KBitVecValue<*> = bvValue(size, 1)

    private fun bvValue(size: UInt, value: Int): KBitVecValue<*> = with(ctx) {
        when (size.toInt()) {
            1 -> mkBv(value == 1)
            Byte.SIZE_BITS -> mkBv(value.toByte())
            Short.SIZE_BITS -> mkBv(value.toShort())
            Int.SIZE_BITS -> mkBv(value)
            Long.SIZE_BITS -> mkBv(value.toLong())
            else -> mkBv(value, size)
        }
    }

    fun KBitVecValue<*>.signedGreaterOrEqual(other: Int): Boolean = when (this) {
        is KBitVec1Value -> value >= (other == 1)
        is KBitVec8Value -> numberValue >= other
        is KBitVec16Value -> numberValue >= other
        is KBitVec32Value -> numberValue >= other
        is KBitVec64Value -> numberValue >= other
        else -> signedBigIntFromBinary(stringValue) >= other.toBigInteger()
    }

    fun KBitVecValue<*>.signedLessOrEqual(other: KBitVecValue<*>): Boolean = when (this) {
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

    operator fun KBitVecValue<*>.plus(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> a xor b },
        bv8 = { a, b -> (a + b).toByte() },
        bv16 = { a, b -> (a + b).toShort() },
        bv32 = { a, b -> a + b },
        bv64 = { a, b -> a + b },
        bvDefault = { a, b -> a + b },
    )

    operator fun KBitVecValue<*>.minus(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> a xor b },
        bv8 = { a, b -> (a - b).toByte() },
        bv16 = { a, b -> (a - b).toShort() },
        bv32 = { a, b -> a - b },
        bv64 = { a, b -> a - b },
        bvDefault = { a, b -> a - b },
    )

    private operator fun KBitVecValue<*>.times(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> a && b },
        bv8 = { a, b -> (a * b).toByte() },
        bv16 = { a, b -> (a * b).toShort() },
        bv32 = { a, b -> a * b },
        bv64 = { a, b -> a * b },
        bvDefault = { a, b -> a * b },
    )

    private fun KBitVecValue<*>.signedDivide(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> a == b },
        bv8 = { a, b -> (a / b).toByte() },
        bv16 = { a, b -> (a / b).toShort() },
        bv32 = { a, b -> a / b },
        bv64 = { a, b -> a / b },
        bvDefault = { a, b -> a / b },
    )

    private fun KBitVecValue<*>.unsignedDivide(other: KBitVecValue<*>): KBitVecValue<*> = bvUnsignedOperation(
        other = other,
        bv1 = { a, b -> a == b },
        bv8 = { a, b -> (a / b).toUByte() },
        bv16 = { a, b -> (a / b).toUShort() },
        bv32 = { a, b -> a / b },
        bv64 = { a, b -> a / b },
        bvDefault = { a, b -> a / b },
    )

    private fun KBitVecValue<*>.signedRem(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> a != b },
        bv8 = { a, b -> (a.rem(b)).toByte() },
        bv16 = { a, b -> (a.rem(b)).toShort() },
        bv32 = { a, b -> a.rem(b) },
        bv64 = { a, b -> a.rem(b) },
        bvDefault = { a, b -> a.rem(b) },
    )

    private fun KBitVecValue<*>.unsignedRem(other: KBitVecValue<*>): KBitVecValue<*> = bvUnsignedOperation(
        other = other,
        bv1 = { a, b -> a != b },
        bv8 = { a, b -> (a.rem(b)).toUByte() },
        bv16 = { a, b -> (a.rem(b)).toUShort() },
        bv32 = { a, b -> a.rem(b) },
        bv64 = { a, b -> a.rem(b) },
        bvDefault = { a, b -> a.rem(b) },
    )

    private fun KBitVecValue<*>.signedMod(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> a != b },
        bv8 = { a, b -> a.mod(b) },
        bv16 = { a, b -> a.mod(b) },
        bv32 = { a, b -> a.mod(b) },
        bv64 = { a, b -> a.mod(b) },
        bvDefault = { a, b -> a.mod(b) },
    )

    private fun KBitVecValue<*>.bitwiseNot(): KBitVecValue<*> = bvOperation(
        other = zero(sort.sizeBits),
        bv1 = { a, _ -> a.not() },
        bv8 = { a, _ -> a.inv() },
        bv16 = { a, _ -> a.inv() },
        bv32 = { a, _ -> a.inv() },
        bv64 = { a, _ -> a.inv() },
        bvDefault = { a, _ -> a.inv() },
    )

    private fun KBitVecValue<*>.bitwiseOr(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> a || b },
        bv8 = { a, b -> a or b },
        bv16 = { a, b -> a or b },
        bv32 = { a, b -> a or b },
        bv64 = { a, b -> a or b },
        bvDefault = { a, b -> a or b },
    )

    private fun KBitVecValue<*>.bitwiseXor(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> a xor b },
        bv8 = { a, b -> a xor b },
        bv16 = { a, b -> a xor b },
        bv32 = { a, b -> a xor b },
        bv64 = { a, b -> a xor b },
        bvDefault = { a, b -> a xor b },
    )

    private fun KBitVecValue<*>.shiftLeft(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, b -> if (b) false else a },
        bv8 = { a, b -> (a.toInt() shl b.toInt()).toByte() },
        bv16 = { a, b -> (a.toInt() shl b.toInt()).toShort() },
        bv32 = { a, b -> a shl b },
        bv64 = { a, b -> a shl b.toInt() },
        bvDefault = { a, b -> a shl b.toInt() },
    )

    private fun KBitVecValue<*>.shiftRightLogical(other: KBitVecValue<*>): KBitVecValue<*> = bvUnsignedOperation(
        other = other,
        bv1 = { a, b -> if (b) false else a },
        bv8 = { a, b -> (a.toInt() ushr b.toInt()).toUByte() },
        bv16 = { a, b -> (a.toInt() ushr b.toInt()).toUShort() },
        bv32 = { a, b -> (a.toInt() ushr b.toInt()).toUInt() },
        bv64 = { a, b -> (a.toInt() ushr b.toInt()).toULong() },
        bvDefault = { a, b -> (a shr b.toInt()) },
    )

    private fun KBitVecValue<*>.shiftRightArith(other: KBitVecValue<*>): KBitVecValue<*> = bvOperation(
        other = other,
        bv1 = { a, _ -> a },
        bv8 = { a, b -> (a.toInt() shr b.toInt()).toByte() },
        bv16 = { a, b -> (a.toInt() shr b.toInt()).toShort() },
        bv32 = { a, b -> a shr b },
        bv64 = { a, b -> a shr b.toInt() },
        bvDefault = { a, b -> a shr b.toInt() },
    )

    private inline fun KBitVecValue<*>.bvUnsignedOperation(
        other: KBitVecValue<*>,
        bv1: (Boolean, Boolean) -> Boolean,
        bv8: (UByte, UByte) -> UByte,
        bv16: (UShort, UShort) -> UShort,
        bv32: (UInt, UInt) -> UInt,
        bv64: (ULong, ULong) -> ULong,
        bvDefault: (BigInteger, BigInteger) -> BigInteger,
    ): KBitVecValue<*> = when (this@bvUnsignedOperation) {
        is KBitVec1Value -> bv1Operation(other, bv1)
        is KBitVec8Value -> bv8UnsignedOperation(other, bv8)
        is KBitVec16Value -> bv16UnsignedOperation(other, bv16)
        is KBitVec32Value -> bv32UnsignedOperation(other, bv32)
        is KBitVec64Value -> bv64UnsignedOperation(other, bv64)
        else -> bvOperationDefault(other, signed = false, bvDefault)
    }

    private inline fun KBitVecValue<*>.bvOperation(
        other: KBitVecValue<*>,
        bv1: (Boolean, Boolean) -> Boolean,
        bv8: (Byte, Byte) -> Byte,
        bv16: (Short, Short) -> Short,
        bv32: (Int, Int) -> Int,
        bv64: (Long, Long) -> Long,
        bvDefault: (BigInteger, BigInteger) -> BigInteger,
    ): KBitVecValue<*> = when (this@bvOperation) {
        is KBitVec1Value -> bv1Operation(other, bv1)
        is KBitVec8Value -> bv8Operation(other, bv8)
        is KBitVec16Value -> bv16Operation(other, bv16)
        is KBitVec32Value -> bv32Operation(other, bv32)
        is KBitVec64Value -> bv64Operation(other, bv64)
        else -> bvOperationDefault(other, signed = true, bvDefault)
    }

    private inline fun KBitVec1Value.bv1Operation(other: KBitVecValue<*>, op: (Boolean, Boolean) -> Boolean) =
        ctx.mkBv(op(value, (other as KBitVec1Value).value))

    private inline fun KBitVec8Value.bv8Operation(other: KBitVecValue<*>, op: (Byte, Byte) -> Byte) =
        ctx.mkBv(op(numberValue, (other as KBitVec8Value).numberValue))

    private inline fun KBitVec8Value.bv8UnsignedOperation(other: KBitVecValue<*>, op: (UByte, UByte) -> UByte) =
        ctx.mkBv(op(numberValue.toUByte(), (other as KBitVec8Value).numberValue.toUByte()).toByte())

    private inline fun KBitVec16Value.bv16Operation(other: KBitVecValue<*>, op: (Short, Short) -> Short) =
        ctx.mkBv(op(numberValue, (other as KBitVec16Value).numberValue))

    private inline fun KBitVec16Value.bv16UnsignedOperation(other: KBitVecValue<*>, op: (UShort, UShort) -> UShort) =
        ctx.mkBv(op(numberValue.toUShort(), (other as KBitVec16Value).numberValue.toUShort()).toShort())

    private inline fun KBitVec32Value.bv32Operation(other: KBitVecValue<*>, op: (Int, Int) -> Int) =
        ctx.mkBv(op(numberValue, (other as KBitVec32Value).numberValue))

    private inline fun KBitVec32Value.bv32UnsignedOperation(other: KBitVecValue<*>, op: (UInt, UInt) -> UInt) =
        ctx.mkBv(op(numberValue.toUInt(), (other as KBitVec32Value).numberValue.toUInt()).toInt())

    private inline fun KBitVec64Value.bv64Operation(other: KBitVecValue<*>, op: (Long, Long) -> Long) =
        ctx.mkBv(op(numberValue, (other as KBitVec64Value).numberValue))

    private inline fun KBitVec64Value.bv64UnsignedOperation(other: KBitVecValue<*>, op: (ULong, ULong) -> ULong) =
        ctx.mkBv(op(numberValue.toULong(), (other as KBitVec64Value).numberValue.toULong()).toLong())

    private inline fun KBitVecValue<*>.bvOperationDefault(
        rhs: KBitVecValue<*>,
        signed: Boolean = false,
        operation: (BigInteger, BigInteger) -> BigInteger
    ): KBitVecValue<*> = with(ctx) {
        val lhs = this@bvOperationDefault
        val size = lhs.sort.sizeBits
        val lValue = lhs.stringValue.let { if (!signed) unsignedBigIntFromBinary(it) else signedBigIntFromBinary(it) }
        val rValue = rhs.stringValue.let { if (!signed) unsignedBigIntFromBinary(it) else signedBigIntFromBinary(it) }
        val resultValue = operation(lValue, rValue)
        mkBvFromBigInteger(resultValue, size)
    }

    fun mkBvFromBigInteger(value: BigInteger, size: UInt): KBitVecValue<KBvSort> {
        val normalizedValue = value.mod(BigInteger.valueOf(2).pow(size.toInt()))
        val resultBinary = unsignedBinaryString(normalizedValue).padStart(size.toInt(), '0')
        return ctx.mkBv(resultBinary, size)
    }

    private fun KBitVecValue<*>.powerOfTwoOrNull(): Int? {
        val value = unsignedBigIntFromBinary(stringValue)
        val valueMinusOne = value - BigInteger.ONE
        if ((value and valueMinusOne) != BigInteger.ZERO) return null
        return valueMinusOne.bitLength()
    }

    private fun KBitVecValue<*>.intValueOrNull(): Int? = when (this) {
        is KBitVec1Value -> if (value) 1 else 0
        is KBitVec8Value -> numberValue.toInt()
        is KBitVec16Value -> numberValue.toInt()
        is KBitVec32Value -> numberValue
        is KBitVec64Value -> if (numberValue <= Int.MAX_VALUE) numberValue.toInt() else null
        else -> stringValue.toIntOrNull(radix = 2)
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
