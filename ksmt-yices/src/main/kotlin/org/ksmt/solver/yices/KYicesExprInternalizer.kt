package org.ksmt.solver.yices

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KAddArithExpr
import org.ksmt.expr.KAndExpr
import org.ksmt.expr.KArrayConst
import org.ksmt.expr.KArrayLambda
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
import org.ksmt.expr.KOrExpr
import org.ksmt.expr.KPowerArithExpr
import org.ksmt.expr.KQuantifier
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
import org.ksmt.expr.rewrite.KExprSubstitutor
import org.ksmt.solver.util.KExprInternalizerBase
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFp128Sort
import org.ksmt.sort.KFp16Sort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.solver.KSolverUnsupportedFeatureException
import org.ksmt.sort.KBv16Sort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBv64Sort
import org.ksmt.sort.KBv8Sort
import org.ksmt.sort.KIntSort
import org.ksmt.utils.mkFreshConstDecl
import org.ksmt.utils.uncheckedCast
import java.math.BigInteger

open class KYicesExprInternalizer(
    private val ctx: KContext,
    private val yicesCtx: KYicesContext,
) : KExprInternalizerBase<YicesTerm>() {

    private val sortInternalizer: KYicesSortInternalizer by lazy { KYicesSortInternalizer(yicesCtx) }
    private val declInternalizer: KYicesDeclInternalizer by lazy {
        KYicesDeclInternalizer(yicesCtx, sortInternalizer)
    }
    private val variableInternalizer: KYicesVariableInternalizer by lazy {
        KYicesVariableInternalizer(yicesCtx, sortInternalizer)
    }

    override fun findInternalizedExpr(expr: KExpr<*>): YicesTerm? = yicesCtx.findInternalizedExpr(expr)

    override fun saveInternalizedExpr(expr: KExpr<*>, internalized: YicesTerm) {
        yicesCtx.internalizeExpr(expr) { internalized }
    }

    fun <T : KSort> KExpr<T>.internalize(): YicesTerm = internalizeExpr()

    fun <T : KDecl<*>> T.internalizeDecl(): YicesTerm = accept(declInternalizer)

    private fun <T : KSort> T.internalizeSort(): YicesSort = accept(sortInternalizer)

    private fun <T : KDecl<*>> T.internalizeVariable(): YicesTerm = accept(variableInternalizer)

    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> = with(expr) {
        transformList(args) { args: Array<YicesTerm> ->
            if (args.isNotEmpty())
                yicesCtx.funApplication(decl.internalizeDecl(), args.toList())
            else
                decl.internalizeDecl()
        }
    }

    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = with(expr) {
        transform { decl.internalizeDecl() }
    }

    override fun transform(expr: KAndExpr): KExpr<KBoolSort> = with(expr) {
        transformList(args) { args: Array<YicesTerm> ->
            yicesCtx.and(args.toList())
        }
    }

    override fun transform(expr: KOrExpr): KExpr<KBoolSort> = with(expr) {
        transformList(args) { args: Array<YicesTerm> -> yicesCtx.or(args.toList()) }
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
        transform(lhs, rhs, yicesCtx::eq)
    }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>): KExpr<KBoolSort> = with(expr) {
        transformList(args) { args: Array<YicesTerm> -> yicesCtx.distinct(args.toList()) }
    }

    override fun <T : KSort> transform(expr: KIteExpr<T>): KExpr<T> = with(expr) {
        transform(condition, trueBranch, falseBranch, yicesCtx::ifThenElse)
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

            yicesCtx.add(args + sign)
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
        transform(array, index, value, yicesCtx::functionUpdate1)
    }

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> = with(expr) {
        transform(array, index) { array: YicesTerm, index: YicesTerm ->
            yicesCtx.funApplication(array, index)
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayConst<D, R>): KExpr<KArraySort<D, R>> = with(expr) {
        transform(value) { value: YicesTerm ->
            yicesCtx.lambda(listOf(yicesCtx.newVariable(sort.internalizeSort())), value)
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KFunctionAsArray<D, R>): KExpr<KArraySort<D, R>> = with(expr) {
        transform { function.internalizeDecl() }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>>  {
        val transformedExpr = yicesCtx.substituteDecls(expr) { term ->
            with(term) {
                val newIndex = indexVarDecl.sort.mkFreshConstDecl(indexVarDecl.name)
                val transformer = KExprSubstitutor(ctx).apply {
                    substitute(indexVarDecl, newIndex)
                }
                ctx.mkArrayLambda(newIndex, transformer.apply(body))
            }
        }

        return with(transformedExpr) {
            val variable = indexVarDecl.internalizeVariable()

            expr.transform(body) { body: YicesTerm ->
                yicesCtx.lambda(listOf(variable), body)
            }
        }
    }

    override fun <T : KArithSort> transform(expr: KAddArithExpr<T>): KExpr<T> = with(expr) {
        transformList(args) { args: Array<YicesTerm> -> yicesCtx.add(args.toList()) }
    }

    override fun <T : KArithSort> transform(expr: KMulArithExpr<T>): KExpr<T> = with(expr) {
        transformList(args) { args: Array<YicesTerm> -> yicesCtx.mul(args.toList()) }
    }

    override fun <T : KArithSort> transform(expr: KSubArithExpr<T>): KExpr<T> = with(expr) {
        transformList(args) { args: Array<YicesTerm> ->
            if (args.size == 1)
                args.first()
            else
                yicesCtx.sub(args[0], yicesCtx.add(args.drop(1)))
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
        transform(arg) {arg: YicesTerm ->
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

    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> = with(ctx) {
        internalizeQuantifier(expr, ::mkExistentialQuantifier) { body, variables ->
            yicesCtx.exists(variables, body)
        }
    }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> = with(ctx) {
        internalizeQuantifier(expr, ::mkUniversalQuantifier) { body, variables ->
            yicesCtx.forall(variables, body)
        }
    }

    private inline fun <T: KQuantifier> internalizeQuantifier(
        expr: T,
        crossinline constructor: (KExpr<KBoolSort>, List<KDecl<*>>) -> T,
        internalizer: (YicesTerm, List<YicesTerm>) -> YicesTerm
    ): T
    {
        val transformedExpr = yicesCtx.substituteDecls(expr) { term: T ->
            with(term) {
                val newBounds = bounds.map { it.sort.mkFreshConstDecl(it.name) }
                val transformer = KExprSubstitutor(ctx).apply {
                    bounds.zip(newBounds).forEach { (bound, newBound) ->
                        substitute(bound.uncheckedCast(), newBound)
                    }
                }

                constructor(transformer.apply(body), newBounds)
            }
        }

        return with(transformedExpr) {
            val variables = bounds.map { it.internalizeVariable() }

            expr.transform(body) { body: YicesTerm ->
                internalizer(body, variables)
            }
        }
    }
}
