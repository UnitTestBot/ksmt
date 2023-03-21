package org.ksmt.solver.yices

import com.sri.yices.Constructor
import com.sri.yices.Terms
import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KAddArithExpr
import org.ksmt.expr.KAndBinaryExpr
import org.ksmt.expr.KAndExpr
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
import org.ksmt.expr.KBitVec16Value
import org.ksmt.expr.KBitVec1Value
import org.ksmt.expr.KBitVec32Value
import org.ksmt.expr.KBitVec64Value
import org.ksmt.expr.KBitVec8Value
import org.ksmt.expr.KBitVecCustomValue
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
import org.ksmt.expr.KBvToFpExpr
import org.ksmt.expr.KBvUnsignedDivExpr
import org.ksmt.expr.KBvUnsignedGreaterExpr
import org.ksmt.expr.KBvUnsignedGreaterOrEqualExpr
import org.ksmt.expr.KBvUnsignedLessExpr
import org.ksmt.expr.KBvUnsignedLessOrEqualExpr
import org.ksmt.expr.KBvUnsignedRemExpr
import org.ksmt.expr.KBvXNorExpr
import org.ksmt.expr.KBvXorExpr
import org.ksmt.expr.KBvZeroExtensionExpr
import org.ksmt.expr.KConst
import org.ksmt.expr.KDistinctExpr
import org.ksmt.expr.KDivArithExpr
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFalse
import org.ksmt.expr.KFp128Value
import org.ksmt.expr.KFp16Value
import org.ksmt.expr.KFp32Value
import org.ksmt.expr.KFp64Value
import org.ksmt.expr.KFpAbsExpr
import org.ksmt.expr.KFpAddExpr
import org.ksmt.expr.KFpCustomSizeValue
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
import org.ksmt.expr.KFpRoundingModeExpr
import org.ksmt.expr.KFpSqrtExpr
import org.ksmt.expr.KFpSubExpr
import org.ksmt.expr.KFpToBvExpr
import org.ksmt.expr.KFpToFpExpr
import org.ksmt.expr.KFpToIEEEBvExpr
import org.ksmt.expr.KFpToRealExpr
import org.ksmt.expr.KFunctionApp
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.expr.KGeArithExpr
import org.ksmt.expr.KGtArithExpr
import org.ksmt.expr.KImpliesExpr
import org.ksmt.expr.KInt32NumExpr
import org.ksmt.expr.KInt64NumExpr
import org.ksmt.expr.KIntBigNumExpr
import org.ksmt.expr.KIsIntRealExpr
import org.ksmt.expr.KIteExpr
import org.ksmt.expr.KLeArithExpr
import org.ksmt.expr.KLtArithExpr
import org.ksmt.expr.KModIntExpr
import org.ksmt.expr.KMulArithExpr
import org.ksmt.expr.KNotExpr
import org.ksmt.expr.KOrBinaryExpr
import org.ksmt.expr.KOrExpr
import org.ksmt.expr.KPowerArithExpr
import org.ksmt.expr.KRealNumExpr
import org.ksmt.expr.KRealToFpExpr
import org.ksmt.expr.KRemIntExpr
import org.ksmt.expr.KSubArithExpr
import org.ksmt.expr.KToIntRealExpr
import org.ksmt.expr.KToRealIntExpr
import org.ksmt.expr.KTrue
import org.ksmt.expr.KUnaryMinusArithExpr
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.KXorExpr
import org.ksmt.solver.KSolverUnsupportedFeatureException
import org.ksmt.solver.util.KExprInternalizerBase
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv16Sort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBv64Sort
import org.ksmt.sort.KBv8Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFp128Sort
import org.ksmt.sort.KFp16Sort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import java.math.BigInteger

open class KYicesExprInternalizer(
    private val ctx: KContext,
    private val yicesCtx: KYicesContext,
) : KExprInternalizerBase<YicesTerm>() {

    private val sortInternalizer: KYicesSortInternalizer by lazy {
        KYicesSortInternalizer(yicesCtx)
    }

    private val declSortInternalizer: KYicesDeclSortInternalizer by lazy {
        KYicesDeclSortInternalizer(yicesCtx, sortInternalizer)
    }

    override fun findInternalizedExpr(expr: KExpr<*>): YicesTerm? = yicesCtx.findInternalizedExpr(expr)

    override fun saveInternalizedExpr(expr: KExpr<*>, internalized: YicesTerm) {
        yicesCtx.saveInternalizedExpr(expr, internalized)
    }

    fun <T : KSort> KExpr<T>.internalize(): YicesTerm = internalizeExpr(this)

    fun <T : KDecl<*>> T.internalizeDecl(): YicesTerm = yicesCtx.internalizeDecl(this) {
        val sort = accept(declSortInternalizer)
        yicesCtx.newUninterpretedTerm(name, sort)
    }

    private fun <T : KDecl<*>> T.internalizeVariable(): YicesTerm = yicesCtx.internalizeVar(this) {
        val sort = accept(declSortInternalizer)
        yicesCtx.newVariable(name, sort)
    }

    private fun <T : KSort> T.internalizeSort(): YicesSort = accept(sortInternalizer)

    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> = with(expr) {
        transformList(args) { args: Array<YicesTerm> ->
            if (args.isNotEmpty()) {
                yicesCtx.funApplication(decl.internalizeDecl(), args.toIntArray())
            } else {
                decl.internalizeDecl()
            }
        }
    }

    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = with(expr) {
        transform { decl.internalizeDecl() }
    }

    override fun transform(expr: KAndExpr): KExpr<KBoolSort> = with(expr) {
        transformList(args) { args: Array<YicesTerm> ->
            yicesCtx.and(args.toIntArray())
        }
    }

    override fun transform(expr: KAndBinaryExpr): KExpr<KBoolSort> = with(expr) {
        transform(lhs, rhs) { l: YicesTerm, r: YicesTerm ->
            yicesCtx.and(intArrayOf(l, r))
        }
    }

    override fun transform(expr: KOrExpr): KExpr<KBoolSort> = with(expr) {
        transformList(args) { args: Array<YicesTerm> -> yicesCtx.or(args.toIntArray()) }
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
        transformList(args) { args: Array<YicesTerm> ->
            if (args.isEmpty()) {
                yicesCtx.mkTrue()
            } else {
                internalizeDistinctExpr(expr.args.first().sort, args.toIntArray())
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

    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>): KExpr<KBoolSort> {
        TODO("Not yet implemented")
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
        transformList(listOf(array, value, index0, index1, index2)) { args: Array<YicesTerm> ->
            mkArrayStoreTerm(
                array = args[0],
                indices = args.copyOfRange(fromIndex = 2, toIndex = args.size).toIntArray(),
                value = args[1]
            )
        }
    }

    override fun <R : KSort> transform(
        expr: KArrayNStore<R>
    ): KExpr<KArrayNSort<R>> = with(expr) {
        transformList(listOf(array, value) + indices) { args: Array<YicesTerm> ->
            mkArrayStoreTerm(
                array = args[0],
                indices = args.copyOfRange(fromIndex = 2, toIndex = args.size).toIntArray(),
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
        transformList(args) { args: Array<YicesTerm> ->
            val a = args[0]
            val indices = IntArray(expr.indices.size) { idx -> args[idx + 1] }

            yicesCtx.funApplication(a, indices)
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
        transformList(args) { args: Array<YicesTerm> -> yicesCtx.add(args.toIntArray()) }
    }

    override fun <T : KArithSort> transform(expr: KMulArithExpr<T>): KExpr<T> = with(expr) {
        transformList(args) { args: Array<YicesTerm> -> yicesCtx.mul(args.toIntArray()) }
    }

    override fun <T : KArithSort> transform(expr: KSubArithExpr<T>): KExpr<T> = with(expr) {
        transformList(args) { args: Array<YicesTerm> ->
            if (args.size == 1) {
                args.first()
            } else {
                val argsToAdd = args.copyOfRange(fromIndex = 1, toIndex = args.size)
                yicesCtx.sub(args[0], yicesCtx.add(argsToAdd.toIntArray()))
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

    fun internalizeExpr(expr: KExpr<*>) = try {
        expr.internalizeExpr()
    } finally {
        resetInternalizer()
    }

    private fun resetInternalizer() {
        exprStack.clear()
    }
}
