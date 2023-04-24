package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KApp
import org.ksmt.expr.KArray2Lambda
import org.ksmt.expr.KArray2Select
import org.ksmt.expr.KArray2Store
import org.ksmt.expr.KArray3Lambda
import org.ksmt.expr.KArray3Select
import org.ksmt.expr.KArray3Store
import org.ksmt.expr.KArrayConst
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KArrayLambdaBase
import org.ksmt.expr.KArrayNLambda
import org.ksmt.expr.KArrayNSelect
import org.ksmt.expr.KArrayNStore
import org.ksmt.expr.KArraySelect
import org.ksmt.expr.KArrayStore
import org.ksmt.expr.KBvToFpExpr
import org.ksmt.expr.KConst
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExistentialQuantifier
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
import org.ksmt.expr.KFunctionApp
import org.ksmt.expr.KIteExpr
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.symfpu.ArraysTransform.Companion.packToBvIfUnpacked
import org.ksmt.symfpu.ArraysTransform.Companion.transformedArraySort
import org.ksmt.symfpu.UnpackedFp.Companion.iteOp
import org.ksmt.utils.asExpr
import org.ksmt.utils.cast

class FpToBvTransformer(ctx: KContext) : KNonRecursiveTransformer(ctx) {
    val arraysTransform = ArraysTransform(ctx)
    private val mapFpToUnpackedFpImpl =
        mutableMapOf<KDecl<KFpSort>, UnpackedFp<KFpSort>>()
    val mapFpToUnpackedFp: Map<KDecl<KFpSort>, UnpackedFp<KFpSort>> get() = mapFpToUnpackedFpImpl


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


    override fun <T : KSort> transform(expr: KEqExpr<T>): KExpr<KBoolSort> = with(ctx) {
        transformExprAfterTransformed(expr, expr.lhs, expr.rhs) { l, r ->
            if (l is UnpackedFp<*> && r is UnpackedFp<*>) {
                val flags = mkAnd(l.isNaN eq r.isNaN, l.isInf eq r.isInf, l.isZero eq r.isZero)
                if (l.packedBv is UnpackedFp.PackedFp.Exists && r.packedBv is UnpackedFp.PackedFp.Exists)
                    flags and (l.packedBv eq r.packedBv)
                else mkAnd(
                    flags,
                    l.sign eq r.sign,
                    l.unbiasedExponent eq r.unbiasedExponent,
                    l.normalizedSignificand eq r.normalizedSignificand,
                )
            } else {
                l eq r
            }
        }
    }

    override fun <T : KSort> transform(expr: KIteExpr<T>): KExpr<T> = with(ctx) {
        transformExprAfterTransformed(expr, expr.condition, expr.trueBranch, expr.falseBranch) { c, l, r ->
            if (l is UnpackedFp<*> && r is UnpackedFp<*>) {
                val lTyped: UnpackedFp<KFpSort> = l.cast()
                iteOp(c, lTyped, r.cast()).cast()
            } else {
                mkIte(c, l, r).cast()
            }
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

    override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KArrayConst<A, R>): KExpr<A> = with(ctx) {
        transformExprAfterTransformed(expr, expr.value) { value ->
            println("transforming arrayConst")
            val resSort = transformedArraySort(expr.cast())
            mkArrayConst(resSort, packToBvIfUnpacked(value)).cast()
        }
    }



    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> {
        return transformExprAfterTransformed(expr, expr.array, expr.index) { array, index ->
            with(ctx) {
                arraysTransform.arraySelectUnpacked(expr.sort, array.select(packToBvIfUnpacked(index).cast()))
            }
        }
    }


    override fun <D : KSort, D1 : KSort, R : KSort> transform(expr: KArray2Select<D, D1, R>): KExpr<R> {
        return transformExprAfterTransformed(expr, expr.array, expr.index0, expr.index1) { array, index0, index1 ->
            with(ctx) {
                arraysTransform.arraySelectUnpacked(expr.sort,
                    array.select(packToBvIfUnpacked(index0).cast(), packToBvIfUnpacked(index1).cast()))
            }
        }
    }

    override fun <D : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(expr: KArray3Select<D, D1, D2, R>) =
        transformExprAfterTransformed(
            expr, expr.array, expr.index0, expr.index1, expr.index2) { array, index0, index1, index2 ->
            with(ctx) {
                arraysTransform.arraySelectUnpacked(expr.sort,
                    array.select(packToBvIfUnpacked(index0).cast(), packToBvIfUnpacked(index1).cast(), index2.cast()))
            }
        }

    override fun <R : KSort> transform(expr: KArrayNSelect<R>) =
        transformExprAfterTransformed(
            expr, expr.args) { args ->
            val array: KExpr<KArrayNSort<R>> = args[0].cast()
            val indices = args.drop(1)
            arraysTransform.arraySelectUnpacked(expr.sort, ctx.mkArrayNSelect(array, indices.map(::packToBvIfUnpacked)))
        }

    private fun <D : KArraySortBase<R>, R : KSort> transformLambda(
        expr: KArrayLambdaBase<D, R>,
    ): KArrayLambdaBase<D, R> =
        transformExprAfterTransformed(expr, expr.body) { body ->
            val newDecl = arraysTransform.transformDeclList(expr.indexVarDeclarations)
            arraysTransform.mkArrayAnyLambda(newDecl, packToBvIfUnpacked(body)).cast()
        }.cast()


    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>> =
        transformLambda(expr)


    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Lambda<D0, D1, R>,
    ): KExpr<KArray2Sort<D0, D1, R>> = transformLambda(expr)


    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Lambda<D0, D1, D2, R>,
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = transformLambda(expr)


    override fun <R : KSort> transform(
        expr: KArrayNLambda<R>,
    ): KExpr<KArrayNSort<R>> = transformLambda(expr)


    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>): KExpr<KArraySort<D, R>> {
        return transformExprAfterTransformed(expr, expr.array, expr.index, expr.value) { array, index, value ->
            with(ctx) {
                array.store(packToBvIfUnpacked(index).cast(), packToBvIfUnpacked(value).cast())
            }
        }
    }

    override fun <D : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Store<D, D1, R>,
    ): KExpr<KArray2Sort<D, D1, R>> = with(ctx) {
        transformExprAfterTransformed(expr, expr.array, expr.index0, expr.index1,
            expr.value) { array, index0, index1, value ->
            array.store(
                packToBvIfUnpacked(index0).cast(), packToBvIfUnpacked(index1).cast(),
                packToBvIfUnpacked(value).cast())
        }
    }

    override fun <D : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Store<D, D1, D2, R>,
    ): KExpr<KArray3Sort<D, D1, D2, R>> = with(ctx) {
        transformExprAfterTransformed(expr, expr.array,
            expr.index0, expr.index1, expr.index2, expr.value) { array, index0, index1, index2, value ->
            array.store(
                packToBvIfUnpacked(index0).cast(), packToBvIfUnpacked(index1).cast(),
                packToBvIfUnpacked(index2).cast(), packToBvIfUnpacked(value).cast())
        }
    }

    override fun <R : KSort> transform(
        expr: KArrayNStore<R>,
    ): KExpr<KArrayNSort<R>> = with(ctx) {
        transformExprAfterTransformed(expr, expr.args) { args ->
            val array: KExpr<KArrayNSort<R>> = args.first().cast()
            val indices = args.subList(fromIndex = 1, toIndex = args.size - 1)
            val value = args.last()

            mkArrayNStore(array, indices.map(::packToBvIfUnpacked), packToBvIfUnpacked(value).cast())
        }
    }

    // quantified expressions
    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> = with(ctx) {
        transformExprAfterTransformed(expr, expr.body) { body ->
            val bounds = arraysTransform.transformDeclList(expr.bounds)
            mkExistentialQuantifier(body, bounds)
        }
    }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> = with(ctx) {
        transformExprAfterTransformed(expr, expr.body) { body ->
            val bounds = arraysTransform.transformDeclList(expr.bounds)
            mkUniversalQuantifier(body, bounds)
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

    // function transformers
    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> =
        transformExprAfterTransformed(expr, expr.args) { args ->
            val decl = arraysTransform.transformDecl(expr.decl)
            val argsTransformed = args.map(::packToBvIfUnpacked)
            decl.apply(argsTransformed).cast() // todo
        }

    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = with(ctx) {
        when (expr.sort) {
            is KFpSort -> {
                val asFp: KConst<KFpSort> = expr.cast()

                mapFpToUnpackedFpImpl.getOrPut(asFp.decl) {
                    unpack(asFp.sort,
                        mkConst(asFp.decl.name + "!tobv!", mkBvSort(
                            asFp.sort.exponentBits + asFp.sort.significandBits)).also {
                            arraysTransform.mapFpToBvDeclImpl[asFp.decl] = (it as KConst<KBvSort>)
                        }
                    )
                }.cast()
            }

            is KArraySortBase<*> -> {
                println("in transform KConst<KArraySortBase<*>>")
                val asArray: KConst<KArraySortBase<*>> = expr.cast()
                val resSort = transformedArraySort(asArray)
                arraysTransform.mapFpToBvDeclImpl.getOrPut(asArray.decl) {
                    mkFreshConst(asArray.decl.name + "!tobvArr!", resSort).cast()
                }.cast()
            }

            else -> expr
        }
    }

    override fun <Fp : KFpSort> transformFpValue(expr: KFpValue<Fp>): KExpr<Fp> = with(ctx) {
        return unpack(
            expr.sort,
            expr.signBit.expr,
            expr.biasedExponent.asExpr(mkBvSort(expr.sort.exponentBits)),
            expr.significand.asExpr(mkBvSort(expr.sort.significandBits - 1u)),
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
                it.packedBv.toIEEE() ?: ctx.packToBv(it)
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
        expr: KApp<R, Fp>, f: (UnpackedFp<Fp>, UnpackedFp<Fp>) -> KExpr<R>,
    ): KExpr<R> =
        transformExprAfterTransformed(expr, expr.args) { args ->
            val (left, right) = argsToTypedPair(args)
            f(left, right)
        }

    private fun <Fp : KFpSort, R : KSort> transformHelper(
        expr: KApp<R, Fp>, f: (UnpackedFp<Fp>) -> KExpr<R>,
    ): KExpr<R> =
        transformExprAfterTransformed(expr, expr.args) { args ->
            val value = args[0] as UnpackedFp<Fp>
            f(value)
        }

}
