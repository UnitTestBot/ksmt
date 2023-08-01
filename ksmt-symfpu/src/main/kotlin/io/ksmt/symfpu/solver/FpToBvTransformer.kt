package io.ksmt.symfpu.solver

import io.ksmt.KContext
import io.ksmt.decl.KConstDecl
import io.ksmt.decl.KDecl
import io.ksmt.decl.KFuncDecl
import io.ksmt.expr.KApp
import io.ksmt.expr.KArray2Lambda
import io.ksmt.expr.KArray2Select
import io.ksmt.expr.KArray2Store
import io.ksmt.expr.KArray3Lambda
import io.ksmt.expr.KArray3Select
import io.ksmt.expr.KArray3Store
import io.ksmt.expr.KArrayConst
import io.ksmt.expr.KArrayLambda
import io.ksmt.expr.KArrayLambdaBase
import io.ksmt.expr.KArrayNLambda
import io.ksmt.expr.KArrayNSelect
import io.ksmt.expr.KArrayNStore
import io.ksmt.expr.KArraySelect
import io.ksmt.expr.KArrayStore
import io.ksmt.expr.KBvToFpExpr
import io.ksmt.expr.KConst
import io.ksmt.expr.KEqExpr
import io.ksmt.expr.KExistentialQuantifier
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFpAbsExpr
import io.ksmt.expr.KFpAddExpr
import io.ksmt.expr.KFpDivExpr
import io.ksmt.expr.KFpEqualExpr
import io.ksmt.expr.KFpFromBvExpr
import io.ksmt.expr.KFpFusedMulAddExpr
import io.ksmt.expr.KFpGreaterExpr
import io.ksmt.expr.KFpGreaterOrEqualExpr
import io.ksmt.expr.KFpIsInfiniteExpr
import io.ksmt.expr.KFpIsNaNExpr
import io.ksmt.expr.KFpIsNegativeExpr
import io.ksmt.expr.KFpIsNormalExpr
import io.ksmt.expr.KFpIsPositiveExpr
import io.ksmt.expr.KFpIsSubnormalExpr
import io.ksmt.expr.KFpIsZeroExpr
import io.ksmt.expr.KFpLessExpr
import io.ksmt.expr.KFpLessOrEqualExpr
import io.ksmt.expr.KFpMaxExpr
import io.ksmt.expr.KFpMinExpr
import io.ksmt.expr.KFpMulExpr
import io.ksmt.expr.KFpNegationExpr
import io.ksmt.expr.KFpRemExpr
import io.ksmt.expr.KFpRoundToIntegralExpr
import io.ksmt.expr.KFpSqrtExpr
import io.ksmt.expr.KFpSubExpr
import io.ksmt.expr.KFpToBvExpr
import io.ksmt.expr.KFpToFpExpr
import io.ksmt.expr.KFpToIEEEBvExpr
import io.ksmt.expr.KFpToRealExpr
import io.ksmt.expr.KFpValue
import io.ksmt.expr.KFunctionApp
import io.ksmt.expr.KIteExpr
import io.ksmt.expr.KRealToFpExpr
import io.ksmt.expr.KUniversalQuantifier
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.solver.KSolverUnsupportedFeatureException
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KSort
import io.ksmt.symfpu.operations.UnpackedFp
import io.ksmt.symfpu.operations.UnpackedFp.Companion.iteOp
import io.ksmt.symfpu.operations.add
import io.ksmt.symfpu.operations.bvToBool
import io.ksmt.symfpu.operations.bvToFp
import io.ksmt.symfpu.operations.divide
import io.ksmt.symfpu.operations.equal
import io.ksmt.symfpu.operations.fma
import io.ksmt.symfpu.operations.fpToBv
import io.ksmt.symfpu.operations.fpToFp
import io.ksmt.symfpu.operations.greater
import io.ksmt.symfpu.operations.greaterOrEqual
import io.ksmt.symfpu.operations.isNegative
import io.ksmt.symfpu.operations.isNormal
import io.ksmt.symfpu.operations.isPositive
import io.ksmt.symfpu.operations.isSubnormal
import io.ksmt.symfpu.operations.less
import io.ksmt.symfpu.operations.lessOrEqual
import io.ksmt.symfpu.operations.max
import io.ksmt.symfpu.operations.min
import io.ksmt.symfpu.operations.multiply
import io.ksmt.symfpu.operations.packToBv
import io.ksmt.symfpu.operations.remainder
import io.ksmt.symfpu.operations.roundToIntegral
import io.ksmt.symfpu.operations.sqrt
import io.ksmt.symfpu.operations.sub
import io.ksmt.symfpu.operations.unpack
import io.ksmt.symfpu.solver.ArraysTransform.Companion.packToBvIfUnpacked
import io.ksmt.symfpu.solver.ArraysTransform.Companion.transformSortRemoveFP
import io.ksmt.symfpu.solver.ArraysTransform.Companion.transformedArraySort
import io.ksmt.symfpu.solver.SymFPUModel.Companion.declContainsFp
import io.ksmt.utils.asExpr
import io.ksmt.utils.uncheckedCast

class FpToBvTransformer(ctx: KContext, private val packedBvOptimization: Boolean) : KNonRecursiveTransformer(ctx) {
    private val mapFpToBvDeclImpl = mutableMapOf<KDecl<*>, KConst<*>>()
    val mapFpToBvDecl: Map<KDecl<*>, KConst<*>> get() = mapFpToBvDeclImpl

    private val arraysTransform = ArraysTransform(ctx, packedBvOptimization)

    private val mapFpToUnpackedFpImpl =
        mutableMapOf<KDecl<KFpSort>, UnpackedFp<KFpSort>>()

    // for tests
    val mapFpToUnpackedFp: Map<KDecl<KFpSort>, UnpackedFp<KFpSort>> get() = mapFpToUnpackedFpImpl

    private val adapterTermsRewriter by lazy { AdapterTermsRewriter() }

    // use this function instead of apply as it may return UnpackedFp wrapper
    fun <T : KSort> applyAndGetExpr(expr: KExpr<T>): KExpr<T> {
        val applied = apply(expr)
        // it might have UnpackedFp inside, so
        // transform them to bvs
        return adapterTermsRewriter.apply(applied)
    }

    inner class AdapterTermsRewriter : KNonRecursiveTransformer(ctx) {
        fun <T : KFpSort> transform(expr: UnpackedFp<T>): KExpr<KBvSort> = with(ctx) {
            return packToBv(expr)
        }
    }


    // non-fp operations
    override fun <T : KSort> transform(expr: KIteExpr<T>): KExpr<T> = with(ctx) {
        transformExprAfterTransformed(expr, expr.condition, expr.trueBranch, expr.falseBranch) { c, l, r ->
            if (l is UnpackedFp<*> && r is UnpackedFp<*>) {
                val lTyped: UnpackedFp<KFpSort> = l.uncheckedCast()
                val rTyped: UnpackedFp<KFpSort> = r.uncheckedCast()
                iteOp(c, lTyped, rTyped).uncheckedCast()
            } else {
                mkIte(c, l, r)
            }
        }
    }

    override fun <T : KSort> transform(expr: KEqExpr<T>): KExpr<KBoolSort> = with(ctx) {
        transformExprAfterTransformed(expr, expr.lhs, expr.rhs) { l, r ->
            if (l is UnpackedFp<*> && r is UnpackedFp<*>) {
                val flags = mkAnd(l.isNaN eq r.isNaN, l.isInf eq r.isInf, l.isZero eq r.isZero)
                if (l.packedBv is UnpackedFp.Companion.PackedFp.Exists &&
                    r.packedBv is UnpackedFp.Companion.PackedFp.Exists
                ) {
                    flags and (l.packedBv eq r.packedBv)
                } else {
                    mkAnd(
                        flags,
                        l.sign eq r.sign,
                        l.unbiasedExponent eq r.unbiasedExponent,
                        l.normalizedSignificand eq r.normalizedSignificand,
                    )
                }
            } else {
                l eq r
            }
        }
    }

    // fp-comparisons
    override fun <Fp : KFpSort> transform(expr: KFpEqualExpr<Fp>): KExpr<KBoolSort> = with(ctx) {
        transformHelper(expr, ::equal)
    }

    override fun <Fp : KFpSort> transform(expr: KFpLessExpr<Fp>): KExpr<KBoolSort> = with(ctx) {
        transformHelper(expr, ::less)
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


    // fp arithmetics
    override fun <Fp : KFpSort> transform(expr: KFpMulExpr<Fp>): KExpr<Fp> = with(ctx) {
        transformExprAfterTransformed(expr, expr.arg0, expr.arg1, expr.roundingMode) { arg0, arg1, rm ->
            multiply(arg0 as UnpackedFp<Fp>, arg1 as UnpackedFp<Fp>, rm)
        }
    }

    override fun <Fp : KFpSort> transform(expr: KFpAddExpr<Fp>): KExpr<Fp> = with(ctx) {
        transformExprAfterTransformed(expr, expr.arg0, expr.arg1, expr.roundingMode) { arg0, arg1, rm ->
            add(arg0 as UnpackedFp<Fp>, arg1 as UnpackedFp<Fp>, rm)
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
            fma(
                arg0 as UnpackedFp<Fp>,
                arg1 as UnpackedFp<Fp>,
                arg2 as UnpackedFp<Fp>,
                roundingMode
            )
        }
    }

    override fun <Fp : KFpSort> transform(expr: KFpSqrtExpr<Fp>): KExpr<Fp> = with(ctx) {
        transformExprAfterTransformed(expr, expr.value, expr.roundingMode) { value, roundingMode ->
            sqrt(roundingMode, value as UnpackedFp<Fp>)
        }
    }

    override fun <Fp : KFpSort> transform(expr: KFpRemExpr<Fp>): KExpr<Fp> =
        transformExprAfterTransformed(expr, expr.arg0, expr.arg1) { arg0, arg1 ->
            remainder(arg0 as UnpackedFp<Fp>, arg1 as UnpackedFp<Fp>)
        }


    override fun <Fp : KFpSort> transform(expr: KFpSubExpr<Fp>): KExpr<Fp> = with(ctx) {
        transformExprAfterTransformed(expr, expr.arg0, expr.arg1, expr.roundingMode) { arg0, arg1, rm ->
            sub(arg0 as UnpackedFp<Fp>, arg1 as UnpackedFp<Fp>, rm)
        }
    }

    override fun <Fp : KFpSort> transform(expr: KFpDivExpr<Fp>): KExpr<Fp> = with(ctx) {
        transformExprAfterTransformed(expr, expr.arg0, expr.arg1, expr.roundingMode) { arg0, arg1, rm ->
            divide(arg0 as UnpackedFp<Fp>, arg1 as UnpackedFp<Fp>, rm)
        }
    }

    // consts
    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = with(ctx) {
        when {
            !declContainsFp(expr.decl) -> expr

            expr.sort is KFpSort -> {
                val asFp: KConst<KFpSort> = expr.uncheckedCast()

                val unpacked = mapFpToUnpackedFpImpl.getOrPut(asFp.decl) {
                    unpack(
                        asFp.sort,
                        mkFreshConst(
                            asFp.decl.name + "!tobv!",
                            mkBvSort(asFp.sort.exponentBits + asFp.sort.significandBits)
                        ).also {
                            mapFpToBvDeclImpl[asFp.decl] = (it as KConst<KBvSort>)
                        },
                        packedBvOptimization
                    )
                }

                unpacked.uncheckedCast<_, KExpr<T>>()
            }

            expr.sort is KArraySortBase<*> -> {
                val asArray: KConst<KArraySortBase<*>> = expr.uncheckedCast()
                val resSort = transformedArraySort(asArray)
                val transformed = mapFpToBvDeclImpl.getOrPut(asArray.decl) {
                    mkFreshConst(asArray.decl.name + "!tobvArr!", resSort).uncheckedCast()
                }

                transformed.uncheckedCast<_, KExpr<T>>()
            }

            else -> expr
        }
    }

    // fp values
    override fun <Fp : KFpSort> transformFpValue(expr: KFpValue<Fp>): KExpr<Fp> = with(ctx) {
        return unpack(
            expr.sort,
            expr.signBit.expr,
            expr.biasedExponent.asExpr(mkBvSort(expr.sort.exponentBits)),
            expr.significand.asExpr(mkBvSort(expr.sort.significandBits - 1u)),
            packedBvOptimization
        )
    }


    // fp predicates
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

    // fp conversions
    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>) =
        transformExprAfterTransformed(expr, expr.roundingMode, expr.value) { roundingMode, value ->
            fpToBv(roundingMode, (value as UnpackedFp<*>), expr.bvSize, expr.isSigned)
        }

    override fun <T : KFpSort> transform(expr: KBvToFpExpr<T>) =
        transformExprAfterTransformed(expr, expr.roundingMode, expr.value) { roundingMode, value ->
            val res = bvToFp(roundingMode, value, expr.sort, expr.signed)
            res
        }

    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>) =
        transformExprAfterTransformed(expr, expr.value) { value ->
            (value as UnpackedFp<T>).let {
                it.packedBv.toIEEE() ?: ctx.packToBv(it)
            }
        }

    override fun <T : KFpSort> transform(expr: KFpFromBvExpr<T>) =
        transformExprAfterTransformed(expr, expr.sign, expr.biasedExponent, expr.significand) { s, e, sig ->
            ctx.unpack(
                expr.sort,
                ctx.bvToBool(s.uncheckedCast()),
                e.uncheckedCast(),
                sig.uncheckedCast(),
                packedBvOptimization
            )
        }

    override fun <Fp : KFpSort> transform(expr: KFpRoundToIntegralExpr<Fp>): KExpr<Fp> =
        transformExprAfterTransformed(expr, expr.roundingMode, expr.value) { roundingMode, value ->
            roundToIntegral(roundingMode, (value as UnpackedFp<Fp>))
        }

    override fun <Fp : KFpSort> transform(expr: KFpToRealExpr<Fp>) =
        throw KSolverUnsupportedFeatureException("FpToRealExpr is not supported")

    override fun <Fp : KFpSort> transform(expr: KRealToFpExpr<Fp>) =
        throw KSolverUnsupportedFeatureException("KRealToFpExpr is not supported")

    // arrays
    override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KArrayConst<A, R>): KExpr<A> = with(ctx) {
        transformExprAfterTransformed(expr, expr.value) { value: KExpr<R> ->
            val resSort = transformedArraySort(expr)
            mkArrayConst(resSort, packToBvIfUnpacked(value))
        }
    }


    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> {
        return transformExprAfterTransformed(expr, expr.array, expr.index) { array, index ->
            with(ctx) {
                arraysTransform.arraySelectUnpacked(
                    expr.sort,
                    array.select(packToBvIfUnpacked(index))
                )
            }
        }
    }


    override fun <D : KSort, D1 : KSort, R : KSort> transform(expr: KArray2Select<D, D1, R>): KExpr<R> {
        return transformExprAfterTransformed(expr, expr.array, expr.index0, expr.index1) { array, index0, index1 ->
            with(ctx) {
                arraysTransform.arraySelectUnpacked(
                    expr.sort,
                    array.select(packToBvIfUnpacked(index0), packToBvIfUnpacked(index1))
                )
            }
        }
    }

    override fun <D : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(expr: KArray3Select<D, D1, D2, R>) =
        transformExprAfterTransformed(
            expr, expr.array, expr.index0, expr.index1, expr.index2
        ) { array, index0, index1, index2 ->
            with(ctx) {
                arraysTransform.arraySelectUnpacked(
                    expr.sort,
                    array.select(packToBvIfUnpacked(index0), packToBvIfUnpacked(index1), index2)
                )
            }
        }

    override fun <R : KSort> transform(expr: KArrayNSelect<R>) =
        transformExprAfterTransformed(
            expr, expr.args
        ) { args ->
            val array: KExpr<KArrayNSort<R>> = args[0].uncheckedCast()
            val indices = args.drop(1)
            arraysTransform.arraySelectUnpacked(
                expr.sort,
                ctx.mkArrayNSelect(array, indices.map(::packToBvIfUnpacked))
            )
        }

    private fun <D : KArraySortBase<R>, R : KSort> transformLambda(
        expr: KArrayLambdaBase<D, R>,
    ): KArrayLambdaBase<D, R> =
        transformExprAfterTransformed(expr, expr.body) { body ->
            val newDecl = transformDeclList(expr.indexVarDeclarations)
            arraysTransform.mkArrayAnyLambda(
                newDecl,
                packToBvIfUnpacked(body)
            ).uncheckedCast()
        }.uncheckedCast()


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
                array.store(packToBvIfUnpacked(index), packToBvIfUnpacked(value))
            }
        }
    }

    override fun <D : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Store<D, D1, R>,
    ): KExpr<KArray2Sort<D, D1, R>> = with(ctx) {
        transformExprAfterTransformed(
            expr, expr.array, expr.index0, expr.index1,
            expr.value
        ) { array, index0, index1, value ->
            array.store(
                packToBvIfUnpacked(index0),
                packToBvIfUnpacked(index1),
                packToBvIfUnpacked(value)
            )
        }
    }

    override fun <D : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Store<D, D1, D2, R>,
    ): KExpr<KArray3Sort<D, D1, D2, R>> = with(ctx) {
        transformExprAfterTransformed(
            expr, expr.array,
            expr.index0, expr.index1, expr.index2, expr.value
        ) { array, index0, index1, index2, value ->
            array.store(
                packToBvIfUnpacked(index0), packToBvIfUnpacked(index1),
                packToBvIfUnpacked(index2), packToBvIfUnpacked(value)
            )
        }
    }

    override fun <R : KSort> transform(
        expr: KArrayNStore<R>,
    ): KExpr<KArrayNSort<R>> = with(ctx) {
        transformExprAfterTransformed(expr, expr.args) { args ->
            val array: KExpr<KArrayNSort<R>> = args.first().uncheckedCast()
            val indices = args.subList(fromIndex = 1, toIndex = args.size - 1)
            val value: KExpr<R> = args.last().uncheckedCast()

            mkArrayNStore(array, indices.map(::packToBvIfUnpacked), packToBvIfUnpacked(value))
        }
    }

    // quantified expressions
    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> = with(ctx) {
        transformExprAfterTransformed(expr, expr.body) { body ->
            val bounds = transformDeclList(expr.bounds)
            mkExistentialQuantifier(body, bounds)
        }
    }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> = with(ctx) {
        transformExprAfterTransformed(expr, expr.body) { body ->
            val bounds = transformDeclList(expr.bounds)
            mkUniversalQuantifier(body, bounds)
        }
    }


    // function transformers
    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> =
        transformExprAfterTransformed(expr, expr.args) { args ->
            val decl = transformDecl(expr.decl)
            val argsTransformed = args.map(::packToBvIfUnpacked)
            decl.apply(argsTransformed).uncheckedCast()
        }


    private fun <Fp : KFpSort, R : KSort> transformHelper(
        expr: KApp<R, Fp>, f: (UnpackedFp<Fp>, UnpackedFp<Fp>) -> KExpr<R>,
    ): KExpr<R> =
        transformExprAfterTransformed(expr, expr.args) { args ->
            val (left, right) = args
            f(left as UnpackedFp<Fp>, right as UnpackedFp<Fp>)
        }

    private fun <Fp : KFpSort, R : KSort> transformHelper(
        expr: KApp<R, Fp>, f: (UnpackedFp<Fp>) -> KExpr<R>,
    ): KExpr<R> =
        transformExprAfterTransformed(expr, expr.args) { args ->
            val value = args[0] as UnpackedFp<Fp>
            f(value)
        }

    private fun transformDeclList(
        decls: List<KDecl<*>>,
    ): List<KDecl<*>> = decls.map {
        transformDecl(it)
    }


    private fun transformDecl(it: KDecl<*>) = with(ctx) {
        val sort = it.sort
        when {
            !declContainsFp(it) -> {
                it
            }

            it is KConstDecl<*> -> {
                val newSort = transformSortRemoveFP(sort)
                mapFpToBvDeclImpl.getOrPut(it) {
                    mkFreshConst(it.name + "!tobv!", newSort).uncheckedCast()
                }.decl
            }

            it is KFuncDecl -> {
                mkFreshFuncDecl(it.name, transformSortRemoveFP(it.sort),
                    it.argSorts.map { ctx.transformSortRemoveFP(it) })
            }

            else -> throw IllegalStateException("Unexpected decl type: $it")
        }
    }
}
