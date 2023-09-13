package io.ksmt.solver.yices

import com.sri.yices.Constructor
import com.sri.yices.Terms
import io.ksmt.decl.KDecl
import io.ksmt.expr.KAddArithExpr
import io.ksmt.expr.KAndBinaryExpr
import io.ksmt.expr.KAndExpr
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
import io.ksmt.expr.KBitVec16Value
import io.ksmt.expr.KBitVec1Value
import io.ksmt.expr.KBitVec32Value
import io.ksmt.expr.KBitVec64Value
import io.ksmt.expr.KBitVec8Value
import io.ksmt.expr.KBitVecCustomValue
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
import io.ksmt.expr.KBvToFpExpr
import io.ksmt.expr.KBvUnsignedDivExpr
import io.ksmt.expr.KBvUnsignedGreaterExpr
import io.ksmt.expr.KBvUnsignedGreaterOrEqualExpr
import io.ksmt.expr.KBvUnsignedLessExpr
import io.ksmt.expr.KBvUnsignedLessOrEqualExpr
import io.ksmt.expr.KBvUnsignedRemExpr
import io.ksmt.expr.KBvXNorExpr
import io.ksmt.expr.KBvXorExpr
import io.ksmt.expr.KBvZeroExtensionExpr
import io.ksmt.expr.KConst
import io.ksmt.expr.KDistinctExpr
import io.ksmt.expr.KDivArithExpr
import io.ksmt.expr.KEqExpr
import io.ksmt.expr.KExistentialQuantifier
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFalse
import io.ksmt.expr.KFp128Value
import io.ksmt.expr.KFp16Value
import io.ksmt.expr.KFp32Value
import io.ksmt.expr.KFp64Value
import io.ksmt.expr.KFpAbsExpr
import io.ksmt.expr.KFpAddExpr
import io.ksmt.expr.KFpCustomSizeValue
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
import io.ksmt.expr.KFpRoundingModeExpr
import io.ksmt.expr.KFpSqrtExpr
import io.ksmt.expr.KFpSubExpr
import io.ksmt.expr.KFpToBvExpr
import io.ksmt.expr.KFpToFpExpr
import io.ksmt.expr.KFpToIEEEBvExpr
import io.ksmt.expr.KFpToRealExpr
import io.ksmt.expr.KFunctionApp
import io.ksmt.expr.KFunctionAsArray
import io.ksmt.expr.KGeArithExpr
import io.ksmt.expr.KGtArithExpr
import io.ksmt.expr.KImpliesExpr
import io.ksmt.expr.KInt32NumExpr
import io.ksmt.expr.KInt64NumExpr
import io.ksmt.expr.KIntBigNumExpr
import io.ksmt.expr.KIsIntRealExpr
import io.ksmt.expr.KIteExpr
import io.ksmt.expr.KLeArithExpr
import io.ksmt.expr.KLtArithExpr
import io.ksmt.expr.KModIntExpr
import io.ksmt.expr.KMulArithExpr
import io.ksmt.expr.KNotExpr
import io.ksmt.expr.KOrBinaryExpr
import io.ksmt.expr.KOrExpr
import io.ksmt.expr.KPowerArithExpr
import io.ksmt.expr.KRealNumExpr
import io.ksmt.expr.KRealToFpExpr
import io.ksmt.expr.KRemIntExpr
import io.ksmt.expr.KSubArithExpr
import io.ksmt.expr.KToIntRealExpr
import io.ksmt.expr.KToRealIntExpr
import io.ksmt.expr.KTrue
import io.ksmt.expr.KUnaryMinusArithExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.expr.KUniversalQuantifier
import io.ksmt.expr.KXorExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvAddNoOverflowExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvAddNoUnderflowExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvDivNoOverflowExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvMulNoOverflowExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvMulNoUnderflowExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvNegNoOverflowExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvSubNoOverflowExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvSubNoUnderflowExpr
import io.ksmt.solver.KSolverUnsupportedFeatureException
import io.ksmt.solver.util.KExprIntInternalizerBase
import io.ksmt.sort.KArithSort
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBv16Sort
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBv32Sort
import io.ksmt.sort.KBv64Sort
import io.ksmt.sort.KBv8Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFp128Sort
import io.ksmt.sort.KFp16Sort
import io.ksmt.sort.KFp32Sort
import io.ksmt.sort.KFp64Sort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import java.math.BigInteger

@Suppress("LargeClass")
open class KYicesExprInternalizer(
    private val yicesCtx: KYicesContext,
) : KExprIntInternalizerBase() {

    private val sortInternalizer: KYicesSortInternalizer by lazy {
        KYicesSortInternalizer(yicesCtx)
    }

    private val declSortInternalizer: KYicesDeclSortInternalizer by lazy {
        KYicesDeclSortInternalizer(yicesCtx, sortInternalizer)
    }

    override fun findInternalizedExpr(expr: KExpr<*>) = yicesCtx.findInternalizedExpr(expr)

    override fun saveInternalizedExpr(expr: KExpr<*>, internalized: YicesTerm) {
        yicesCtx.saveInternalizedExpr(expr, internalized)
    }

    fun <T : KSort> KExpr<T>.internalize(): YicesTerm = try {
        internalizeExpr()
    } finally {
        resetInternalizer()
    }

    fun <T : KDecl<*>> T.internalizeDecl(): YicesTerm = yicesCtx.internalizeDecl(this) {
        val sort = declSortInternalizer.internalizeYicesDeclSort(this)
        yicesCtx.newUninterpretedTerm(name, sort)
    }

    private fun <T : KDecl<*>> T.internalizeVariable(): YicesTerm = yicesCtx.internalizeVar(this) {
        val sort = declSortInternalizer.internalizeYicesDeclSort(this)
        yicesCtx.newVariable(name, sort)
    }

    private fun <T : KSort> T.internalizeSort(): YicesSort =
        sortInternalizer.internalizeYicesSort(this)

    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> = with(expr) {
        transformList(args) { args: YicesTermArray ->
            if (args.isNotEmpty()) {
                yicesCtx.funApplication(decl.internalizeDecl(), args)
            } else {
                decl.internalizeDecl()
            }
        }
    }

    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = with(expr) {
        transform { decl.internalizeDecl() }
    }

    override fun transform(expr: KAndExpr): KExpr<KBoolSort> = with(expr) {
        transformList(args) { args: YicesTermArray ->
            yicesCtx.and(args)
        }
    }

    override fun transform(expr: KAndBinaryExpr): KExpr<KBoolSort> = with(expr) {
        transform(lhs, rhs) { l: YicesTerm, r: YicesTerm ->
            yicesCtx.and(intArrayOf(l, r))
        }
    }

    override fun transform(expr: KOrExpr): KExpr<KBoolSort> = with(expr) {
        transformList(args) { args: YicesTermArray -> yicesCtx.or(args) }
    }

    override fun transform(expr: KOrBinaryExpr): KExpr<KBoolSort> = with(expr) {
        transform(lhs, rhs) { l: YicesTerm, r: YicesTerm ->
            yicesCtx.or(intArrayOf(l, r))
        }
    }

    override fun transform(expr: KNotExpr): KExpr<KBoolSort> = with(expr) {
        transform(arg, yicesCtx::not)
    }

    override fun transform(expr: KImpliesExpr): KExpr<KBoolSort> = with(expr) {
        transform(p, q, yicesCtx::implies)
    }

    override fun transform(expr: KXorExpr): KExpr<KBoolSort> = with(expr) {
        transform(a, b) { a: YicesTerm, b: YicesTerm -> yicesCtx.xor(a, b) }
    }

    override fun transform(expr: KTrue): KExpr<KBoolSort> = expr.transform(yicesCtx::mkTrue)

    override fun transform(expr: KFalse): KExpr<KBoolSort> = expr.transform(yicesCtx::mkFalse)

    override fun <T : KSort> transform(expr: KEqExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(lhs, rhs) { l: YicesTerm, r: YicesTerm ->
            internalizeEqExpr(lhs.sort, l, r)
        }
    }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>): KExpr<KBoolSort> = with(expr) {
        transformList(args) { args: YicesTermArray ->
            if (args.isEmpty()) {
                yicesCtx.mkTrue()
            } else {
                internalizeDistinctExpr(expr.args.first().sort, args)
            }
        }
    }

    override fun <T : KSort> transform(expr: KIteExpr<T>): KExpr<T> = with(expr) {
        transform(condition, trueBranch, falseBranch) { c: YicesTerm, t: YicesTerm, f: YicesTerm ->
            internalizeIteExpr(sort, c, t, f)
        }
    }

    private fun internalizeIteExpr(
        sort: KSort,
        condition: YicesTerm,
        trueBranch: YicesTerm,
        falseBranch: YicesTerm
    ): YicesTerm = if (sort is KArraySortBase<*>) {
        mkArrayIteTerm(sort, condition, trueBranch, falseBranch)
    } else {
        yicesCtx.ifThenElse(condition, trueBranch, falseBranch)
    }

    private fun internalizeEqExpr(
        sort: KSort,
        lhs: YicesTerm,
        rhs: YicesTerm
    ): YicesTerm = if (sort is KArraySortBase<*>) {
        mkArrayEqTerm(lhs, rhs)
    } else {
        yicesCtx.eq(lhs, rhs)
    }

    private fun internalizeDistinctExpr(
        sort: KSort,
        args: YicesTermArray
    ): YicesTerm = if (sort is KArraySortBase<*>) {
        mkArrayDistinctTerm(args)
    } else {
        yicesCtx.distinct(args)
    }

    override fun transform(expr: KBitVec1Value): KExpr<KBv1Sort> = with(expr) {
        transform { yicesCtx.bvConst(sort.sizeBits, if (value) 1L else 0L) }
    }

    override fun transform(expr: KBitVec8Value): KExpr<KBv8Sort> = with(expr) {
        transform { yicesCtx.bvConst(sort.sizeBits, numberValue.toLong()) }
    }

    override fun transform(expr: KBitVec16Value): KExpr<KBv16Sort> = with(expr) {
        transform { yicesCtx.bvConst(sort.sizeBits, numberValue.toLong()) }
    }

    override fun transform(expr: KBitVec32Value): KExpr<KBv32Sort> = with(expr) {
        transform { yicesCtx.bvConst(sort.sizeBits, numberValue.toLong()) }
    }

    override fun transform(expr: KBitVec64Value): KExpr<KBv64Sort> = with(expr) {
        transform { yicesCtx.bvConst(sort.sizeBits, numberValue) }
    }

    override fun transform(expr: KBitVecCustomValue): KExpr<KBvSort> = with(expr) {
        transform { yicesCtx.parseBvBin(stringValue) }
    }

    override fun <T : KBvSort> transform(expr: KBvNotExpr<T>): KExpr<T> = with(expr) {
        transform(value, yicesCtx::bvNot)
    }

    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>): KExpr<KBv1Sort> = with(expr) {
        transform(value, yicesCtx::bvRedAnd)
    }

    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>): KExpr<KBv1Sort> = with(expr) {
        transform(value, yicesCtx::bvRedOr)
    }

    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvAnd)
    }

    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvOr)
    }

    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvXor)
    }

    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvNand)
    }

    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvNor)
    }

    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvXNor)
    }

    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>): KExpr<T> = with(expr) {
        transform(value, yicesCtx::bvNeg)
    }

    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvAdd)
    }

    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvSub)
    }

    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvMul)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvDiv)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvSDiv)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvRem)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvSRem)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvSMod)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvLt)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvSLt)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvLe)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvSLe)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvGe)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvSGe)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvGt)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvSGt)
    }

    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> = with(expr) {
        transform(arg0, arg1, yicesCtx::bvConcat)
    }

    override fun transform(expr: KBvExtractExpr): KExpr<KBvSort> = with(expr) {
        transform(value) { value: YicesTerm -> yicesCtx.bvExtract(value, low, high) }
    }

    override fun transform(expr: KBvSignExtensionExpr): KExpr<KBvSort> = with(expr) {
        transform(value) { value: YicesTerm -> yicesCtx.bvSignExtend(value, extensionSize) }
    }

    override fun transform(expr: KBvZeroExtensionExpr): KExpr<KBvSort> = with(expr) {
        transform(value) { value: YicesTerm -> yicesCtx.bvZeroExtend(value, extensionSize) }
    }

    override fun transform(expr: KBvRepeatExpr): KExpr<KBvSort> = with(expr) {
        transform(value) { value: YicesTerm -> yicesCtx.bvRepeat(value, repeatNumber) }
    }

    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>): KExpr<T> = with(expr) {
        transform(arg, shift, yicesCtx::bvShl)
    }

    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>): KExpr<T> = with(expr) {
        transform(arg, shift, yicesCtx::bvLshr)
    }

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>): KExpr<T> = with(expr) {
        transform(arg, shift, yicesCtx::bvAshr)
    }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>): KExpr<T> = with(expr) {
        transform(arg, rotation) { arg0: YicesTerm, arg1: YicesTerm ->
            val size = expr.sort.sizeBits
            val bvSize = yicesCtx.bvConst(size, size.toLong())
            val rotationNumber = yicesCtx.bvRem(arg1, bvSize)

            val left = yicesCtx.bvShl(arg0, rotationNumber)
            val right = yicesCtx.bvLshr(arg0, yicesCtx.bvSub(bvSize, rotationNumber))

            yicesCtx.bvOr(left, right)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>): KExpr<T> = with(expr) {
        transform(value) { value: YicesTerm -> yicesCtx.bvRotateLeft(value, rotationNumber) }
    }

    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>): KExpr<T> = with(expr) {
        transform(arg, rotation) { arg0: YicesTerm, arg1: YicesTerm ->
            val size = expr.sort.sizeBits
            val bvSize = yicesCtx.bvConst(size, size.toLong())
            val rotationNumber = yicesCtx.bvRem(arg1, bvSize)

            val left = yicesCtx.bvShl(arg0, yicesCtx.bvSub(bvSize, rotationNumber))
            val right = yicesCtx.bvLshr(arg0, rotationNumber)

            yicesCtx.bvOr(left, right)
        }
    }


    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>): KExpr<T> = with(expr) {
        transform(value) { value: YicesTerm -> yicesCtx.bvRotateRight(value, rotationNumber) }
    }

    override fun transform(expr: KBv2IntExpr): KExpr<KIntSort> = with(expr) {
        transform(value) { value: YicesTerm ->
            val size = expr.value.sort.sizeBits.toInt()

            val args = (0 until size - 1).map {
                yicesCtx.ifThenElse(
                    yicesCtx.bvExtractBit(value, it),
                    yicesCtx.intConst(BigInteger.valueOf(2).pow(it)),
                    yicesCtx.zero
                )
            }

            var sign = yicesCtx.ifThenElse(
                yicesCtx.bvExtractBit(value, size - 1),
                yicesCtx.intConst(BigInteger.valueOf(2).pow(size - 1)),
                yicesCtx.zero
            )

            if (isSigned)
                sign = yicesCtx.neg(sign)

            yicesCtx.add((args + sign).toIntArray())
        }
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform {
            ctx.rewriteBvAddNoOverflowExpr(arg0, arg1, isSigned).internalizeExpr()
        }
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform {
            ctx.rewriteBvAddNoUnderflowExpr(arg0, arg1).internalizeExpr()
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform {
            ctx.rewriteBvSubNoOverflowExpr(arg0, arg1).internalizeExpr()
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform {
            ctx.rewriteBvSubNoUnderflowExpr(arg0, arg1, isSigned).internalizeExpr()
        }
    }

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform {
            ctx.rewriteBvDivNoOverflowExpr(arg0, arg1).internalizeExpr()
        }
    }

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform {
            ctx.rewriteBvNegNoOverflowExpr(value).internalizeExpr()
        }
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform {
            ctx.rewriteBvMulNoOverflowExpr(arg0, arg1, isSigned).internalizeExpr()
        }
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform {
            ctx.rewriteBvMulNoUnderflowExpr(arg0, arg1).internalizeExpr()
        }
    }

    override fun transform(expr: KFp16Value): KExpr<KFp16Sort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun transform(expr: KFp32Value): KExpr<KFp32Sort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun transform(expr: KFp64Value): KExpr<KFp64Sort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun transform(expr: KFp128Value): KExpr<KFp128Sort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun transform(expr: KFpCustomSizeValue): KExpr<KFpSort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun transform(expr: KFpRoundingModeExpr): KExpr<KFpRoundingModeSort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpAbsExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpNegationExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpAddExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpSubExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpMulExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpDivExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpFusedMulAddExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpSqrtExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpRemExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpRoundToIntegralExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpMinExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpMaxExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpLessOrEqualExpr<T>): KExpr<KBoolSort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpLessExpr<T>): KExpr<KBoolSort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpGreaterOrEqualExpr<T>): KExpr<KBoolSort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpGreaterExpr<T>): KExpr<KBoolSort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpEqualExpr<T>): KExpr<KBoolSort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>): KExpr<KBoolSort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>): KExpr<KBoolSort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>): KExpr<KBoolSort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>): KExpr<KBoolSort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>): KExpr<KBoolSort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>): KExpr<KBoolSort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>): KExpr<KBoolSort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpFromBvExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KFpToFpExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KRealToFpExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <T : KFpSort> transform(expr: KBvToFpExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("Unsupported expr $expr")
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>): KExpr<KArraySort<D, R>> = with(expr) {
        transform(array, index, value) { a: YicesTerm, index: YicesTerm, v: YicesTerm ->
            mkArrayStoreTerm(a, intArrayOf(index), v)
        }
    }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Store<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = with(expr) {
        transform(array, index0, index1, value) { a: YicesTerm, i0: YicesTerm, i1: YicesTerm, v: YicesTerm ->
            mkArrayStoreTerm(a, intArrayOf(i0, i1), v)
        }
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Store<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = with(expr) {
        transformList(listOf(array, value, index0, index1, index2)) { args: YicesTermArray ->
            mkArrayStoreTerm(
                array = args[0],
                indices = args.copyOfRange(fromIndex = 2, toIndex = args.size),
                value = args[1]
            )
        }
    }

    override fun <R : KSort> transform(
        expr: KArrayNStore<R>
    ): KExpr<KArrayNSort<R>> = with(expr) {
        transformList(listOf(array, value) + indices) { args: YicesTermArray ->
            mkArrayStoreTerm(
                array = args[0],
                indices = args.copyOfRange(fromIndex = 2, toIndex = args.size),
                value = args[1]
            )
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> = with(expr) {
        transform(array, index) { array: YicesTerm, index: YicesTerm ->
            yicesCtx.funApplication(array, index)
        }
    }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Select<D0, D1, R>
    ): KExpr<R> = with(expr) {
        transform(array, index0, index1) { a: YicesTerm, i0: YicesTerm, i1: YicesTerm ->
            yicesCtx.funApplication(a, intArrayOf(i0, i1))
        }
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Select<D0, D1, D2, R>
    ): KExpr<R> = with(expr) {
        transform(array, index0, index1, index2) { a: YicesTerm, i0: YicesTerm, i1: YicesTerm, i2: YicesTerm ->
            yicesCtx.funApplication(a, intArrayOf(i0, i1, i2))
        }
    }

    override fun <R : KSort> transform(expr: KArrayNSelect<R>): KExpr<R> = with(expr) {
        transformList(args) { args: YicesTermArray ->
            yicesCtx.funApplication(
                func = args[0],
                args = args.copyOfRange(fromIndex = 1, toIndex = args.size)
            )
        }
    }

    override fun <A : KArraySortBase<R>, R : KSort> transform(
        expr: KArrayConst<A, R>
    ): KExpr<A> = with(expr) {
        transform(value) { value: YicesTerm ->
            val bounds = sort.domainSorts.let { domain ->
                IntArray(domain.size) { yicesCtx.newVariable(domain[it].internalizeSort()) }
            }
            yicesCtx.lambda(bounds, value)
        }
    }

    override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KFunctionAsArray<A, R>): KExpr<A> = with(expr) {
        transform { function.internalizeDecl() }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>> =
        expr.transformArrayLambda()

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Lambda<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> =
        expr.transformArrayLambda()

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Lambda<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> =
        expr.transformArrayLambda()

    override fun <R : KSort> transform(expr: KArrayNLambda<R>): KExpr<KArrayNSort<R>> =
        expr.transformArrayLambda()

    private fun <L : KArrayLambdaBase<*, *>> L.transformArrayLambda(): L =
        internalizeQuantifiedBody(indexVarDeclarations, body) { vars, body ->
            yicesCtx.lambda(vars, body)
        }

    override fun <T : KArithSort> transform(expr: KAddArithExpr<T>): KExpr<T> = with(expr) {
        transformList(args) { args: YicesTermArray -> yicesCtx.add(args) }
    }

    override fun <T : KArithSort> transform(expr: KMulArithExpr<T>): KExpr<T> = with(expr) {
        transformList(args) { args: YicesTermArray -> yicesCtx.mul(args) }
    }

    override fun <T : KArithSort> transform(expr: KSubArithExpr<T>): KExpr<T> = with(expr) {
        transformList(args) { args: YicesTermArray ->
            if (args.size == 1) {
                args.first()
            } else {
                val argsToAdd = args.copyOfRange(fromIndex = 1, toIndex = args.size)
                yicesCtx.sub(args[0], yicesCtx.add(argsToAdd))
            }
        }
    }

    override fun <T : KArithSort> transform(expr: KUnaryMinusArithExpr<T>): KExpr<T> = with(expr) {
        transform(arg, yicesCtx::neg)
    }

    override fun <T : KArithSort> transform(expr: KDivArithExpr<T>): KExpr<T> = with(expr) {
        transform(lhs, rhs) { lhs: YicesTerm, rhs: YicesTerm ->
            when (sort) {
                is KIntSort -> yicesCtx.idiv(lhs, rhs)
                else -> yicesCtx.div(lhs, rhs)
            }
        }
    }

    override fun <T : KArithSort> transform(expr: KPowerArithExpr<T>): KExpr<T> = with(expr) {
        transform(lhs, rhs, yicesCtx::power)
    }

    override fun <T : KArithSort> transform(expr: KLtArithExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(lhs, rhs, yicesCtx::arithLt)
    }

    override fun <T : KArithSort> transform(expr: KLeArithExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(lhs, rhs, yicesCtx::arithLeq)
    }

    override fun <T : KArithSort> transform(expr: KGtArithExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(lhs, rhs, yicesCtx::arithGt)
    }

    override fun <T : KArithSort> transform(expr: KGeArithExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(lhs, rhs, yicesCtx::arithGeq)
    }

    override fun transform(expr: KModIntExpr): KExpr<KIntSort> = with(expr) {
        transform(lhs, rhs, yicesCtx::imod)
    }

    override fun transform(expr: KRemIntExpr): KExpr<KIntSort> = with(expr) {
        transform(lhs, rhs) { lhs: YicesTerm, rhs: YicesTerm ->
            val sign = yicesCtx.ifThenElse(yicesCtx.arithLeq0(rhs), yicesCtx.minusOne, yicesCtx.one)
            val mod = yicesCtx.imod(lhs, rhs)

            yicesCtx.mul(mod, sign)
        }
    }

    override fun transform(expr: KToRealIntExpr): KExpr<KRealSort> = with(expr) {
        /**
         * Yices doesn't distinguish between IntSort and RealSort
         */
        transform(arg) { arg: YicesTerm ->
            arg
        }
    }

    override fun transform(expr: KInt32NumExpr): KExpr<KIntSort> = with(expr) {
        transform { yicesCtx.intConst(value.toLong()) }
    }

    override fun transform(expr: KInt64NumExpr): KExpr<KIntSort> = with(expr) {
        transform { yicesCtx.intConst(value) }
    }

    override fun transform(expr: KIntBigNumExpr): KExpr<KIntSort> = with(expr) {
        transform { yicesCtx.intConst(value) }
    }

    override fun transform(expr: KToIntRealExpr): KExpr<KIntSort> = with(expr) {
        transform(arg, yicesCtx::floor)
    }

    override fun transform(expr: KIsIntRealExpr): KExpr<KBoolSort> = with(expr) {
        transform(arg, yicesCtx::isInt)
    }

    override fun transform(expr: KRealNumExpr): KExpr<KRealSort> = with(expr) {
        transform(numerator, denominator) { numerator: YicesTerm, denominator: YicesTerm ->
            yicesCtx.div(numerator, denominator)
        }
    }

    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> = with(expr) {
        internalizeQuantifiedBody(bounds, body) { vars, body ->
            yicesCtx.exists(vars, body)
        }
    }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> = with(expr) {
        internalizeQuantifiedBody(bounds, body) { vars, body ->
            yicesCtx.forall(vars, body)
        }
    }

    override fun transform(expr: KUninterpretedSortValue): KExpr<KUninterpretedSort> = with(expr) {
        transform {
            yicesCtx.uninterpretedSortConst(
                sort.internalizeSort(),
                yicesCtx.uninterpretedSortValueIndex(valueIdx)
            )
        }
    }

    private inline fun <E : KExpr<*>> E.internalizeQuantifiedBody(
        quantifiedDecls: List<KDecl<*>>,
        quantifierBody: KExpr<*>,
        internalizer: (YicesTermArray, YicesTerm) -> YicesTerm
    ): E = transform(quantifierBody) { body: YicesTerm ->
        val consts = IntArray(quantifiedDecls.size)
        val vars = IntArray(quantifiedDecls.size)

        for (i in quantifiedDecls.indices) {
            val decl = quantifiedDecls[i]
            consts[i] = decl.internalizeDecl()
            vars[i] = decl.internalizeVariable()
        }

        val bodyWithVars = yicesCtx.substitute(
            term = body,
            substituteFrom = consts,
            substituteTo = vars
        )

        internalizer(vars, bodyWithVars)
    }

    private fun mkArrayStoreTerm(
        array: YicesTerm,
        indices: YicesTermArray,
        value: YicesTerm
    ): YicesTerm {
        if (!array.isLambda()) {
            return yicesCtx.functionUpdate(array, indices, value)
        }

        val indicesSorts = IntArray(indices.size) { Terms.typeOf(indices[it]) }
        return mkArrayIteLambdaTerm(
            indicesSorts = indicesSorts,
            mkCondition = { boundVars ->
                val indexEqualities = IntArray(indices.size) {
                    yicesCtx.eq(indices[it], boundVars[it])
                }
                yicesCtx.and(indexEqualities)
            },
            mkTrueBranch = { value },
            mkFalseBranch = { boundVars -> yicesCtx.funApplication(array, boundVars) }
        )
    }

    private fun mkArrayIteTerm(
        sort: KArraySortBase<*>,
        condition: YicesTerm,
        trueBranch: YicesTerm,
        falseBranch: YicesTerm
    ): YicesTerm {
        if (!trueBranch.isLambda() && !falseBranch.isLambda()) {
            return yicesCtx.ifThenElse(condition, trueBranch, falseBranch)
        }

        val indicesSorts = sort.domainSorts.let { domain ->
            IntArray(domain.size) { domain[it].internalizeSort() }
        }
        return mkArrayIteLambdaTerm(
            indicesSorts = indicesSorts,
            mkCondition = { condition },
            mkTrueBranch = { boundVars -> yicesCtx.funApplication(trueBranch, boundVars) },
            mkFalseBranch = { boundVars -> yicesCtx.funApplication(falseBranch, boundVars) }
        )
    }

    private fun mkArrayEqTerm(lhs: YicesTerm, rhs: YicesTerm): YicesTerm {
        if (!lhs.isLambda() && !rhs.isLambda()) {
            return yicesCtx.eq(lhs, rhs)
        }

        throw KSolverUnsupportedFeatureException("Yices doesn't support equalities with lambda expressions")
    }

    private fun mkArrayDistinctTerm(args: YicesTermArray): YicesTerm {
        if (args.all { !it.isLambda() }) {
            return yicesCtx.distinct(args)
        }

        // Blast array distinct
        val inequalities = mutableListOf<YicesTerm>()
        for (i in args.indices) {
            for (j in (i + 1) until args.size) {
                val equality = mkArrayEqTerm(args[i], args[j])
                inequalities += yicesCtx.not(equality)
            }
        }

        return yicesCtx.and(inequalities.toIntArray())
    }

    private fun YicesTerm.isLambda(): Boolean =
        Terms.constructor(this) == Constructor.LAMBDA_TERM

    private inline fun mkArrayIteLambdaTerm(
        indicesSorts: YicesSortArray,
        mkCondition: (YicesTermArray) -> YicesTerm,
        mkTrueBranch: (YicesTermArray) -> YicesTerm,
        mkFalseBranch: (YicesTermArray) -> YicesTerm
    ): YicesTerm {
        val lambdaBoundVars = IntArray(indicesSorts.size) {
            yicesCtx.newVariable(indicesSorts[it])
        }
        val condition = mkCondition(lambdaBoundVars)
        val trueBranch = mkTrueBranch(lambdaBoundVars)
        val falseBranch = mkFalseBranch(lambdaBoundVars)

        val lambdaBody = yicesCtx.ifThenElse(condition, trueBranch, falseBranch)
        return yicesCtx.lambda(lambdaBoundVars, lambdaBody)
    }

    private fun resetInternalizer() {
        exprStack.clear()
    }
}
