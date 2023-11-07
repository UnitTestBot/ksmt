package io.ksmt.symfpu.solver

import io.ksmt.KContext
import io.ksmt.decl.KConstDecl
import io.ksmt.decl.KDecl
import io.ksmt.decl.KFuncDecl
import io.ksmt.expr.KArray2Lambda
import io.ksmt.expr.KArray2Select
import io.ksmt.expr.KArray2Store
import io.ksmt.expr.KArray3Lambda
import io.ksmt.expr.KArray3Select
import io.ksmt.expr.KArray3Store
import io.ksmt.expr.KArrayConst
import io.ksmt.expr.KArrayLambda
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
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KSortVisitor
import io.ksmt.sort.KUninterpretedSort
import io.ksmt.symfpu.operations.PackedFp
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
import io.ksmt.utils.asExpr
import io.ksmt.utils.uncheckedCast

class FpToBvTransformer(ctx: KContext, private val packedBvOptimization: Boolean) : KNonRecursiveTransformer(ctx) {
    private val mapFpToBvDecl = mutableMapOf<KDecl<*>, KDecl<*>>()
    private val reverseMapBvToFpDecl = mutableMapOf<KDecl<*>, KDecl<*>>()

    private val mapFpToUnpackedFpImpl = mutableMapOf<KDecl<KFpSort>, UnpackedFp<KFpSort>>()

    val mapFpToUnpackedFp: Map<KDecl<KFpSort>, UnpackedFp<KFpSort>>
        get() = mapFpToUnpackedFpImpl

    private val fpSortDetector = FpSortDetector()
    private val fpSortRewriter = FpSortRewriter(ctx)

    private val unpackedFpRewriter by lazy { UnpackedFpRewriter() }

    // use this function instead of apply as it may return UnpackedFp wrapper
    fun <T : KSort> applyAndGetExpr(expr: KExpr<T>): KExpr<T> {
        val applied = apply(expr)
        // it might have UnpackedFp inside, so
        // transform them to bvs
        return unpackedFpRewriter.apply(applied)
    }

    fun findMappedDeclForFpDecl(fpDecl: KDecl<*>): KDecl<*>? = mapFpToBvDecl[fpDecl]

    fun findFpDeclByMappedDecl(bvDecl: KDecl<*>): KDecl<*>? = reverseMapBvToFpDecl[bvDecl]

    inner class UnpackedFpRewriter : KNonRecursiveTransformer(ctx) {
        fun <T : KFpSort> transform(expr: UnpackedFp<T>): KExpr<KBvSort> = ctx.packToBv(expr)
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
                if (l.packedFp is PackedFp && r.packedFp is PackedFp) {
                    flags and (l.packedFp eq r.packedFp)
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

    // consts
    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = with(ctx) {
        when {
            !declContainsFp(expr.decl) -> expr

            expr.sort is KFpSort -> {
                val decl: KDecl<KFpSort> = expr.decl.uncheckedCast()
                val transformedDecl = transformDecl(decl)

                val unpacked = mapFpToUnpackedFpImpl.getOrPut(decl) {
                    unpack(
                        decl.sort,
                        transformedDecl.apply(emptyList()).uncheckedCast(),
                        packedBvOptimization
                    )
                }

                unpacked.uncheckedCast<_, KExpr<T>>()
            }

            expr.sort is KArraySortBase<*> -> {
                transformDecl(expr.decl).apply(emptyList())
            }

            else -> expr
        }
    }

    // fp-comparisons
    override fun <Fp : KFpSort> transform(expr: KFpEqualExpr<Fp>): KExpr<KBoolSort> =
        transformExprAfterTransformed(expr, expr.arg0, expr.arg1) { arg0, arg1 ->
            ctx.equal(arg0 as UnpackedFp<Fp>, arg1 as UnpackedFp<Fp>)
        }

    override fun <Fp : KFpSort> transform(expr: KFpLessExpr<Fp>): KExpr<KBoolSort> =
        transformExprAfterTransformed(expr, expr.arg0, expr.arg1) { arg0, arg1 ->
            ctx.less(arg0 as UnpackedFp<Fp>, arg1 as UnpackedFp<Fp>)
        }

    override fun <Fp : KFpSort> transform(expr: KFpLessOrEqualExpr<Fp>): KExpr<KBoolSort> =
        transformExprAfterTransformed(expr, expr.arg0, expr.arg1) { arg0, arg1 ->
            ctx.lessOrEqual(arg0 as UnpackedFp<Fp>, arg1 as UnpackedFp<Fp>)
        }

    override fun <Fp : KFpSort> transform(expr: KFpGreaterExpr<Fp>): KExpr<KBoolSort> =
        transformExprAfterTransformed(expr, expr.arg0, expr.arg1) { arg0, arg1 ->
            ctx.greater(arg0 as UnpackedFp<Fp>, arg1 as UnpackedFp<Fp>)
        }

    override fun <Fp : KFpSort> transform(expr: KFpGreaterOrEqualExpr<Fp>): KExpr<KBoolSort> =
        transformExprAfterTransformed(expr, expr.arg0, expr.arg1) { arg0, arg1 ->
            ctx.greaterOrEqual(arg0 as UnpackedFp<Fp>, arg1 as UnpackedFp<Fp>)
        }

    override fun <Fp : KFpSort> transform(expr: KFpMinExpr<Fp>): KExpr<Fp> =
        transformExprAfterTransformed(expr, expr.arg0, expr.arg1) { arg0, arg1 ->
            ctx.min(arg0 as UnpackedFp<Fp>, arg1 as UnpackedFp<Fp>)
        }

    override fun <Fp : KFpSort> transform(expr: KFpMaxExpr<Fp>): KExpr<Fp> =
        transformExprAfterTransformed(expr, expr.arg0, expr.arg1) { arg0, arg1 ->
            ctx.max(arg0 as UnpackedFp<Fp>, arg1 as UnpackedFp<Fp>)
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
    override fun <Fp : KFpSort> transform(expr: KFpMulExpr<Fp>): KExpr<Fp> =
        transformExprAfterTransformed(expr, expr.arg0, expr.arg1, expr.roundingMode) { arg0, arg1, rm ->
            ctx.multiply(arg0 as UnpackedFp<Fp>, arg1 as UnpackedFp<Fp>, rm)
        }

    override fun <Fp : KFpSort> transform(expr: KFpAddExpr<Fp>): KExpr<Fp> =
        transformExprAfterTransformed(expr, expr.arg0, expr.arg1, expr.roundingMode) { arg0, arg1, rm ->
            ctx.add(arg0 as UnpackedFp<Fp>, arg1 as UnpackedFp<Fp>, rm)
        }

    override fun <Fp : KFpSort> transform(expr: KFpFusedMulAddExpr<Fp>): KExpr<Fp> =
        transformExprAfterTransformed(
            expr,
            expr.arg0,
            expr.arg1,
            expr.arg2,
            expr.roundingMode
        ) { arg0, arg1, arg2, roundingMode ->
            ctx.fma(
                arg0 as UnpackedFp<Fp>,
                arg1 as UnpackedFp<Fp>,
                arg2 as UnpackedFp<Fp>,
                roundingMode
            )
        }

    override fun <Fp : KFpSort> transform(expr: KFpSqrtExpr<Fp>): KExpr<Fp> =
        transformExprAfterTransformed(expr, expr.value, expr.roundingMode) { value, roundingMode ->
            ctx.sqrt(roundingMode, value as UnpackedFp<Fp>)
        }

    override fun <Fp : KFpSort> transform(expr: KFpRemExpr<Fp>): KExpr<Fp> =
        transformExprAfterTransformed(expr, expr.arg0, expr.arg1) { arg0, arg1 ->
            remainder(arg0 as UnpackedFp<Fp>, arg1 as UnpackedFp<Fp>)
        }

    override fun <Fp : KFpSort> transform(expr: KFpSubExpr<Fp>): KExpr<Fp> =
        transformExprAfterTransformed(expr, expr.arg0, expr.arg1, expr.roundingMode) { arg0, arg1, rm ->
            ctx.sub(arg0 as UnpackedFp<Fp>, arg1 as UnpackedFp<Fp>, rm)
        }

    override fun <Fp : KFpSort> transform(expr: KFpDivExpr<Fp>): KExpr<Fp> =
        transformExprAfterTransformed(expr, expr.arg0, expr.arg1, expr.roundingMode) { arg0, arg1, rm ->
            ctx.divide(arg0 as UnpackedFp<Fp>, arg1 as UnpackedFp<Fp>, rm)
        }

    // fp values
    override fun <Fp : KFpSort> transformFpValue(expr: KFpValue<Fp>): KExpr<Fp> = with(ctx) {
        unpack(
            expr.sort,
            expr.signBit.expr,
            expr.biasedExponent.asExpr(mkBvSort(expr.sort.exponentBits)),
            expr.significand.asExpr(mkBvSort(expr.sort.significandBits - 1u)),
            packedBvOptimization
        )
    }

    // fp predicates
    override fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>) =
        transformPredicate(expr, expr.value, ::isNormal)

    override fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>) =
        transformPredicate(expr, expr.value, ::isSubnormal)

    override fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>) =
        transformPredicate(expr, expr.value, UnpackedFp<T>::isZero)

    override fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>) =
        transformPredicate(expr, expr.value, UnpackedFp<T>::isInf)

    override fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>) =
        transformPredicate(expr, expr.value, UnpackedFp<T>::isNaN)

    override fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>) =
        transformPredicate(expr, expr.value, ::isNegative)

    override fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>) =
        transformPredicate(expr, expr.value, ::isPositive)

    private inline fun <Fp : KFpSort> transformPredicate(
        expr: KExpr<KBoolSort>, value: KExpr<Fp>,
        mkPredicate: (UnpackedFp<Fp>) -> KExpr<KBoolSort>,
    ): KExpr<KBoolSort> = transformExprAfterTransformed(expr, value) { transformedValue ->
        mkPredicate(transformedValue as UnpackedFp<Fp>)
    }

    // fp conversions
    override fun <T : KFpSort> transform(expr: KFpToFpExpr<T>) =
        transformExprAfterTransformed(expr, expr.roundingMode, expr.value) { roundingMode, value ->
            fpToFp(expr.sort, roundingMode, value as UnpackedFp<*>)
        }

    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>) =
        transformExprAfterTransformed(expr, expr.roundingMode, expr.value) { roundingMode, value ->
            fpToBv(roundingMode, value as UnpackedFp<T>, expr.bvSize, expr.isSigned)
        }

    override fun <T : KFpSort> transform(expr: KBvToFpExpr<T>) =
        transformExprAfterTransformed(expr, expr.roundingMode, expr.value) { roundingMode, value ->
            bvToFp(roundingMode, value, expr.sort, expr.signed)
        }

    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>) =
        transformExprAfterTransformed(expr, expr.value) { value ->
            val unpackedValue = value as UnpackedFp<T>
            if (unpackedValue.packedFp is PackedFp) {
                unpackedValue.packedFp.toIEEEBv()
            } else {
                ctx.packToBv(unpackedValue)
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
            roundToIntegral(roundingMode, value as UnpackedFp<Fp>)
        }

    override fun <Fp : KFpSort> transform(expr: KFpToRealExpr<Fp>) =
        throw KSolverUnsupportedFeatureException("FpToRealExpr is not supported")

    override fun <Fp : KFpSort> transform(expr: KRealToFpExpr<Fp>) =
        throw KSolverUnsupportedFeatureException("KRealToFpExpr is not supported")

    // quantified expressions
    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> = with(ctx) {
        transformExprAfterTransformed(expr, expr.body) { body ->
            val bounds = expr.bounds.map { transformDecl(it) }
            mkExistentialQuantifier(body, bounds)
        }
    }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> = with(ctx) {
        transformExprAfterTransformed(expr, expr.body) { body ->
            val bounds = expr.bounds.map { transformDecl(it) }
            mkUniversalQuantifier(body, bounds)
        }
    }

    // function transformers
    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> =
        transformExprAfterTransformed(expr, expr.args) { args ->
            val decl = transformDecl(expr.decl)
            val argsTransformed = args.map(::packToBvIfUnpacked)
            val res = decl.apply(argsTransformed)
            unpackFromBvIfPacked(expr.sort, res.uncheckedCast())
        }

    // arrays
    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>): KExpr<KArraySort<D, R>> =
        transformExprAfterTransformed(
            expr, expr.array, expr.index, expr.value
        ) { array, index, value ->
            ctx.mkArrayStore(
                array,
                packToBvIfUnpacked(index),
                packToBvIfUnpacked(value)
            )
        }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Store<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = transformExprAfterTransformed(
        expr, expr.array, expr.index0, expr.index1, expr.value
    ) { array, index0, index1, value ->
        ctx.mkArrayStore(
            array,
            packToBvIfUnpacked(index0),
            packToBvIfUnpacked(index1),
            packToBvIfUnpacked(value)
        )
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Store<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = transformExprAfterTransformed(
        expr, expr.array, expr.index0, expr.index1, expr.index2, expr.value
    ) { array, index0, index1, index2, value ->
        ctx.mkArrayStore(
            array,
            packToBvIfUnpacked(index0),
            packToBvIfUnpacked(index1),
            packToBvIfUnpacked(index2),
            packToBvIfUnpacked(value)
        )
    }

    override fun <R : KSort> transform(expr: KArrayNStore<R>): KExpr<KArrayNSort<R>> =
        transformExprAfterTransformed(expr, expr.args) { args ->
            ctx.mkArrayNStore(
                array = args.first().uncheckedCast(),
                indices = args.subList(fromIndex = 1, toIndex = args.size - 1).map(::packToBvIfUnpacked),
                value = packToBvIfUnpacked(args.last()).uncheckedCast()
            )
        }

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> =
        transformExprAfterTransformed(expr, expr.array, expr.index) { array, index ->
            val res = ctx.mkArraySelect(array, packToBvIfUnpacked(index))
            unpackFromBvIfPacked(expr.sort, res)
        }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Select<D0, D1, R>
    ): KExpr<R> = transformExprAfterTransformed(expr, expr.array, expr.index0, expr.index1) { array, i0, i1 ->
        val res = ctx.mkArraySelect(array, packToBvIfUnpacked(i0), packToBvIfUnpacked(i1))
        unpackFromBvIfPacked(expr.sort, res)
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Select<D0, D1, D2, R>
    ): KExpr<R> = transformExprAfterTransformed(
        expr, expr.array, expr.index0, expr.index1, expr.index2
    ) { array, i0, i1, i2 ->
        val res = ctx.mkArraySelect(array, packToBvIfUnpacked(i0), packToBvIfUnpacked(i1), packToBvIfUnpacked(i2))
        unpackFromBvIfPacked(expr.sort, res)
    }

    override fun <R : KSort> transform(expr: KArrayNSelect<R>): KExpr<R> =
        transformExprAfterTransformed(expr, expr.args) { args ->
            val res: KExpr<R> = ctx.mkArrayNSelect(
                array = args.first().uncheckedCast(),
                indices = args.drop(1).map { packToBvIfUnpacked(it) }
            )
            unpackFromBvIfPacked(expr.sort, res)
        }

    override fun <D : KSort, R : KSort> transform(
        expr: KArrayLambda<D, R>
    ): KExpr<KArraySort<D, R>> = transformExprAfterTransformed(expr, expr.body) { body ->
        ctx.mkArrayLambda(
            transformDecl(expr.indexVarDecl),
            packToBvIfUnpacked(body)
        )
    }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Lambda<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = transformExprAfterTransformed(expr, expr.body) { body ->
        ctx.mkArrayLambda(
            transformDecl(expr.indexVar0Decl),
            transformDecl(expr.indexVar1Decl),
            packToBvIfUnpacked(body)
        )
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Lambda<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = transformExprAfterTransformed(expr, expr.body) { body ->
        ctx.mkArrayLambda(
            transformDecl(expr.indexVar0Decl),
            transformDecl(expr.indexVar1Decl),
            transformDecl(expr.indexVar2Decl),
            packToBvIfUnpacked(body)
        )
    }

    override fun <R : KSort> transform(
        expr: KArrayNLambda<R>
    ): KExpr<KArrayNSort<R>> = transformExprAfterTransformed(expr, expr.body) { body ->
        ctx.mkArrayNLambda(
            expr.indexVarDeclarations.map { transformDecl(it) },
            packToBvIfUnpacked(body)
        )
    }

    override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KArrayConst<A, R>): KExpr<A> =
        transformExprAfterTransformed(expr, expr.value) { value ->
            val sort = transformSortRemoveFP(expr.sort)
            ctx.mkArrayConst(sort.uncheckedCast(), packToBvIfUnpacked(value))
        }

    private fun <T : KSort> transformDecl(decl: KDecl<T>): KDecl<T> = with(ctx) {
        when {
            !declContainsFp(decl) -> decl

            decl is KConstDecl<*> -> mapFpToBvDecl.getOrPut(decl) {
                val newSort = transformSortRemoveFP(decl.sort)
                mkFreshConstDecl(decl.name + "!tobv!", newSort).also {
                    reverseMapBvToFpDecl[it] = decl
                }
            }.uncheckedCast()

            decl is KFuncDecl<*> -> mapFpToBvDecl.getOrPut(decl) {
                mkFreshFuncDecl(
                    decl.name + "!toBv!",
                    transformSortRemoveFP(decl.sort),
                    decl.argSorts.map { transformSortRemoveFP(it) }
                ).also {
                    reverseMapBvToFpDecl[it] = decl
                }
            }.uncheckedCast()

            else -> throw IllegalStateException("Unexpected decl type: $decl")
        }
    }

    private fun <T : KSort> packToBvIfUnpacked(expr: KExpr<T>): KExpr<T> = when (expr.sort) {
        is KFpSort -> {
            val unpacked = expr as? UnpackedFp<*> ?: error("Unexpected fp expr: $expr")
            ctx.packToBv(unpacked).uncheckedCast()
        }

        else -> expr
    }

    private fun <T : KSort> unpackFromBvIfPacked(sort: T, expr: KExpr<T>): KExpr<T> = when (sort) {
        is KFpSort -> ctx.unpack(sort, expr.uncheckedCast(), packedBvOptimization).uncheckedCast()
        else -> expr
    }

    private fun transformSortRemoveFP(sort: KSort) = sort.accept(fpSortRewriter)
    private fun sortContainsFP(curSort: KSort): Boolean = curSort.accept(fpSortDetector)

    private fun <T : KSort> declContainsFp(decl: KDecl<T>) =
        sortContainsFP(decl.sort) || decl.argSorts.any { sortContainsFP(it) }

    private class FpSortDetector : KSortVisitor<Boolean> {
        override fun <S : KFpSort> visit(sort: S): Boolean = true

        override fun visit(sort: KFpRoundingModeSort): Boolean =
            throw KSolverUnsupportedFeatureException("Rounding mode expressions are not supported")

        override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): Boolean =
            sort.range.accept(this) || sort.domain.accept(this)

        override fun <D0 : KSort, D1 : KSort, R : KSort> visit(sort: KArray2Sort<D0, D1, R>): Boolean =
            sort.range.accept(this)
                    || sort.domain0.accept(this)
                    || sort.domain1.accept(this)

        override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(sort: KArray3Sort<D0, D1, D2, R>): Boolean =
            sort.range.accept(this)
                    || sort.domain0.accept(this)
                    || sort.domain1.accept(this)
                    || sort.domain2.accept(this)

        override fun <R : KSort> visit(sort: KArrayNSort<R>): Boolean =
            sort.range.accept(this) || sort.domainSorts.any { it.accept(this) }

        override fun visit(sort: KUninterpretedSort): Boolean = false
        override fun visit(sort: KBoolSort): Boolean = false
        override fun visit(sort: KIntSort): Boolean = false
        override fun visit(sort: KRealSort): Boolean = false
        override fun <S : KBvSort> visit(sort: S): Boolean = false
    }

    private class FpSortRewriter(val ctx: KContext) : KSortVisitor<KSort> {
        override fun <S : KFpSort> visit(sort: S): KSort =
            ctx.mkBvSort(sort.exponentBits + sort.significandBits)

        override fun visit(sort: KFpRoundingModeSort): KSort =
            TODO("Fp rounding mode transformer")

        override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): KSort {
            val range = sort.range.accept(this)
            val domain = sort.domain.accept(this)
            return ctx.mkArraySort(domain, range)
        }

        override fun <D0 : KSort, D1 : KSort, R : KSort> visit(sort: KArray2Sort<D0, D1, R>): KSort {
            val range = sort.range.accept(this)
            val domain0 = sort.domain0.accept(this)
            val domain1 = sort.domain1.accept(this)
            return ctx.mkArraySort(domain0, domain1, range)
        }

        override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(sort: KArray3Sort<D0, D1, D2, R>): KSort {
            val range = sort.range.accept(this)
            val domain0 = sort.domain0.accept(this)
            val domain1 = sort.domain1.accept(this)
            val domain2 = sort.domain2.accept(this)
            return ctx.mkArraySort(domain0, domain1, domain2, range)
        }

        override fun <R : KSort> visit(sort: KArrayNSort<R>): KSort {
            val range = sort.range.accept(this)
            val domain = sort.domainSorts.map { it.accept(this) }
            return ctx.mkArrayNSort(domain, range)
        }

        override fun visit(sort: KUninterpretedSort): KSort = sort
        override fun visit(sort: KBoolSort): KSort = sort
        override fun visit(sort: KIntSort): KSort = sort
        override fun visit(sort: KRealSort): KSort = sort
        override fun <S : KBvSort> visit(sort: S): KSort = sort
    }
}
