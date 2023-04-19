package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KBvToFpExpr
import org.ksmt.expr.KConst
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFpAbsExpr
import org.ksmt.expr.KFpAddExpr
import org.ksmt.expr.KFpDivExpr
import org.ksmt.expr.KFpEqualExpr
import org.ksmt.expr.KFpFromBvExpr
import org.ksmt.expr.KFpFusedMulAddExpr
import org.ksmt.expr.KFpGreaterExpr
import org.ksmt.expr.KFpGreaterOrEqualExpr
import org.ksmt.expr.KFpIsInfiniteExpr
import org.ksmt.expr.KFpIsNaNExpr
import org.ksmt.expr.KFpIsNegativeExpr
import org.ksmt.expr.KFpIsNormalExpr
import org.ksmt.expr.KFpIsPositiveExpr
import org.ksmt.expr.KFpIsSubnormalExpr
import org.ksmt.expr.KFpIsZeroExpr
import org.ksmt.expr.KFpLessExpr
import org.ksmt.expr.KFpLessOrEqualExpr
import org.ksmt.expr.KFpMaxExpr
import org.ksmt.expr.KFpMinExpr
import org.ksmt.expr.KFpMulExpr
import org.ksmt.expr.KFpNegationExpr
import org.ksmt.expr.KFpRemExpr
import org.ksmt.expr.KFpRoundToIntegralExpr
import org.ksmt.expr.KFpSqrtExpr
import org.ksmt.expr.KFpSubExpr
import org.ksmt.expr.KFpToBvExpr
import org.ksmt.expr.KFpToFpExpr
import org.ksmt.expr.KFpToIEEEBvExpr
import org.ksmt.expr.KFpValue
import org.ksmt.expr.rewrite.simplify.simplifyFpToIEEEBvExpr
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.utils.asExpr
import org.ksmt.utils.cast

class FpToBvTransformer(ctx: KContext) : KNonRecursiveTransformer(ctx) {
    private val mapFpToBvImpl = mutableMapOf<KExpr<KFpSort>, UnpackedFp<KFpSort>>()
    val mapFpToBv: Map<KExpr<KFpSort>, UnpackedFp<KFpSort>> get() = mapFpToBvImpl
    // use this function instead of apply as it may return UnpackedFp wrapper

    fun <T : KSort> applyAndGetExpr(expr: KExpr<T>): KExpr<T> {
        val applied = apply(expr)
        // it might have UnpackedFp inside, so
        // transform them to bvs
        return AdapterTermsRewriter().apply(applied)
    }

    inner class AdapterTermsRewriter : KNonRecursiveTransformer(ctx) {
        fun <T : KFpSort> transform(expr: UnpackedFp<T>): KExpr<KBvSort> = with(ctx) {
            return packToBv(expr)
        }
    }

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
            multiply(left, right, args[0].cast())
        }
    }

    override fun <Fp : KFpSort> transform(expr: KFpAddExpr<Fp>): KExpr<Fp> = with(ctx) {
        val args1: List<KExpr<Fp>> = expr.args.cast()
        transformExprAfterTransformed(expr, args1) { args ->
            val (left, right) = argsToTypedPair(args.drop(1))
            add(left, right, args[0].cast())
        }
    }

    override fun <Fp : KFpSort> transform(expr: KFpFusedMulAddExpr<Fp>): KExpr<Fp> = with(ctx) {
        transformExprAfterTransformed(
            expr,
            expr.arg0,
            expr.arg1,
            expr.arg2,
            expr.roundingMode
        ) { arg0, arg1, arg2, roundingMode ->
            fma(arg0.cast(), arg1.cast(), arg2.cast(), roundingMode.cast())
        }
    }

    override fun <Fp : KFpSort> transform(expr: KFpSqrtExpr<Fp>): KExpr<Fp> = with(ctx) {
        transformExprAfterTransformed(expr, expr.value, expr.roundingMode) { value, roundingMode ->
            sqrt(roundingMode, value.cast())
        }
    }

    override fun <Fp : KFpSort> transform(expr: KFpRemExpr<Fp>): KExpr<Fp> =
        transformExprAfterTransformed(expr, expr.arg0, expr.arg1) { arg0, arg1 ->
            remainder(arg0.cast(), arg1.cast())
        }


    override fun <Fp : KFpSort> transform(expr: KFpSubExpr<Fp>): KExpr<Fp> = with(ctx) {
        val args1: List<KExpr<Fp>> = expr.args.cast()
        transformExprAfterTransformed(expr, args1) { args ->
            val (left, right) = argsToTypedPair(args.drop(1))
            sub(left, right, args[0].cast())
        }
    }

    override fun <Fp : KFpSort> transform(expr: KFpDivExpr<Fp>): KExpr<Fp> = with(ctx) {
        val args1: List<KExpr<Fp>> = expr.args.cast()
        transformExprAfterTransformed(expr, args1) { args ->
            val (left, right) = argsToTypedPair(args.drop(1))
            divide(left, right, args[0].cast())
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

    override fun <Fp : KFpSort> transform(expr: KFpNegationExpr<Fp>): KExpr<Fp> =
        transformExprAfterTransformed(expr, expr.value) { value ->
            (value as UnpackedFp<Fp>).negate()
        }

    override fun <Fp : KFpSort> transform(expr: KFpAbsExpr<Fp>): KExpr<Fp> =
        transformExprAfterTransformed(expr, expr.value) { value ->
            (value as UnpackedFp<Fp>).absolute()
        }


    override fun <Fp : KFpSort> transform(expr: KFpRoundToIntegralExpr<Fp>): KExpr<Fp> =
        transformExprAfterTransformed(expr, expr.roundingMode, expr.value) { roundingMode, value ->
            roundToIntegral(roundingMode, (value as UnpackedFp<Fp>))
        }

    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = with(ctx) {
        return if (expr.sort is KFpSort) {
            val asFp: KConst<KFpSort> = expr.cast()
            mapFpToBvImpl.getOrPut(asFp) {
                unpack(asFp.sort,
                    mkConst(asFp.decl.name + "!tobv!", mkBvSort(asFp.sort.exponentBits + asFp.sort.significandBits)))
            }.cast()
        } else expr
    }

    override fun <Fp : KFpSort> transformFpValue(expr: KFpValue<Fp>): KExpr<Fp> = with(ctx) {
        return unpack(
            expr.sort,
            expr.signBit.expr,
            expr.biasedExponent.asExpr(mkBvSort(expr.sort.exponentBits)),
            expr.significand.asExpr(mkBvSort(expr.sort.significandBits - 1u)),
            simplifyFpToIEEEBvExpr(expr)
        )
    }


    override fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>) = transformHelper(expr, ::isNormal)
    override fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>) = transformHelper(expr, ::isSubnormal)
    override fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>) = transformHelper(expr, UnpackedFp<T>::isZero)
    override fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>) = transformHelper(expr, UnpackedFp<T>::isInf)
    override fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>) = transformHelper(expr, UnpackedFp<T>::isNaN)
    override fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>) = transformHelper(expr, ::isNegative)
    override fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>) = transformHelper(expr, ::isPositive)
    override fun <T : KFpSort> transform(expr: KFpToFpExpr<T>) =
        transformExprAfterTransformed(expr, expr.roundingMode, expr.value) { roundingMode, value ->
            fpToFp(expr.sort, roundingMode, (value as UnpackedFp<*>))
        }

    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>) =
        transformExprAfterTransformed(expr, expr.roundingMode, expr.value) { roundingMode, value ->
            fpToBv(roundingMode, (value as UnpackedFp<*>), expr.bvSize, expr.isSigned)
        }

    override fun <T : KFpSort> transform(expr: KBvToFpExpr<T>) =
        transformExprAfterTransformed(expr, expr.roundingMode, expr.value) { roundingMode, value ->
            bvToFp(roundingMode, value, expr.sort, expr.signed)
        }

    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>) =
        transformExprAfterTransformed(expr, expr.value) { value ->
            (value as UnpackedFp<T>).let {
                it.packedBv ?: ctx.packToBv(it)
            }
        }

    override fun <T : KFpSort> transform(expr: KFpFromBvExpr<T>) =
        transformExprAfterTransformed(expr, expr.sign, expr.biasedExponent, expr.significand) { s, e, sig ->
            ctx.unpack(expr.sort, ctx.bvToBool(s.cast()), e.cast(), sig.cast())
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

    private fun <Fp : KFpSort, R : KSort> transformHelper(
        expr: KApp<R, Fp>, f: (UnpackedFp<Fp>) -> KExpr<R>
    ): KExpr<R> =
        transformExprAfterTransformed(expr, expr.args) { args ->
            val value = args[0] as UnpackedFp<Fp>
            f(value)
        }

}
