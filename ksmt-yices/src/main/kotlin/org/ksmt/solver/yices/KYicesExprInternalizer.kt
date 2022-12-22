package org.ksmt.solver.yices

import com.sri.yices.Terms
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
import java.math.BigInteger

open class KYicesExprInternalizer(
    private val ctx: KContext,
    private val yicesCtx: KYicesContext,
    private val sortInternalizer: KYicesSortInternalizer,
    private val declInternalizer: KYicesDeclInternalizer,
    private val variableInternalizer: KYicesVariableInternalizer
) : KExprInternalizerBase<YicesTerm>() {
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
                Terms.funApplication(decl.internalizeDecl(), args.toList())
            else
                decl.internalizeDecl()
        }
    }

    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = with(expr) {
        transform { decl.internalizeDecl() }
    }

    override fun transform(expr: KAndExpr): KExpr<KBoolSort> = with(expr) {
        transformList(args) { args: Array<YicesTerm> ->
            Terms.and(args.toList())
        }
    }

    override fun transform(expr: KOrExpr): KExpr<KBoolSort> = with(expr) {
        transformList(args) { args: Array<YicesTerm> -> Terms.or(args.toList()) }
    }

    override fun transform(expr: KNotExpr): KExpr<KBoolSort> = with(expr) {
        transform(arg, Terms::not)
    }

    override fun transform(expr: KImpliesExpr): KExpr<KBoolSort> = with(expr) {
        transform(p, q, Terms::implies)
    }

    override fun transform(expr: KXorExpr): KExpr<KBoolSort> = with(expr) {
        transform(a, b) { a: YicesTerm, b: YicesTerm -> Terms.xor(a, b) }
    }

    override fun transform(expr: KTrue): KExpr<KBoolSort> = expr.transform(Terms::mkTrue)

    override fun transform(expr: KFalse): KExpr<KBoolSort> = expr.transform(Terms::mkFalse)

    override fun <T : KSort> transform(expr: KEqExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(lhs, rhs, Terms::eq)
    }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>): KExpr<KBoolSort> = with(expr) {
        transformList(args) { args: Array<YicesTerm> -> Terms.distinct(args.toList()) }
    }

    override fun <T : KSort> transform(expr: KIteExpr<T>): KExpr<T> = with(expr) {
        transform(condition, trueBranch, falseBranch, Terms::ifThenElse)
    }

    override fun transform(expr: KBitVec1Value): KExpr<KBv1Sort> = with(expr) {
        transform { Terms.bvConst(sort.sizeBits.toInt(), if (value) 1L else 0L) }
    }

    override fun transform(expr: KBitVec8Value): KExpr<KBv8Sort> = with(expr) {
        transform { Terms.bvConst(sort.sizeBits.toInt(), numberValue.toLong()) }
    }

    override fun transform(expr: KBitVec16Value): KExpr<KBv16Sort> = with(expr) {
        transform { Terms.bvConst(sort.sizeBits.toInt(), numberValue.toLong()) }
    }

    override fun transform(expr: KBitVec32Value): KExpr<KBv32Sort> = with(expr) {
        transform { Terms.bvConst(sort.sizeBits.toInt(), numberValue.toLong()) }
    }

    override fun transform(expr: KBitVec64Value): KExpr<KBv64Sort> = with(expr) {
        transform { Terms.bvConst(sort.sizeBits.toInt(), numberValue) }
    }

    override fun transform(expr: KBitVecCustomValue): KExpr<KBvSort> = with(expr) {
        transform { Terms.parseBvBin(binaryStringValue) }
    }

    override fun <T : KBvSort> transform(expr: KBvNotExpr<T>): KExpr<T> = with(expr) {
        transform(value, Terms::bvNot)
    }

    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>): KExpr<KBv1Sort> = with(expr) {
        transform(value, Terms::bvRedAnd)
    }

    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>): KExpr<KBv1Sort> = with(expr) {
        transform(value, Terms::bvRedOr)
    }

    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, Terms::bvAnd)
    }

    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, Terms::bvOr)
    }

    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, Terms::bvXor)
    }

    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, Terms::bvNand)
    }

    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, Terms::bvNor)
    }

    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, Terms::bvXNor)
    }

    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>): KExpr<T> = with(expr) {
        transform(value, Terms::bvNeg)
    }

    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, Terms::bvAdd)
    }

    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, Terms::bvSub)
    }

    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, Terms::bvMul)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, Terms::bvDiv)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, Terms::bvSDiv)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, Terms::bvRem)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, Terms::bvSRem)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, Terms::bvSMod)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, Terms::bvLt)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, Terms::bvSLt)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, Terms::bvLe)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, Terms::bvSLe)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, Terms::bvGe)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, Terms::bvSGe)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, Terms::bvGt)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, Terms::bvSGt)
    }

    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> = with(expr) {
        transform(arg0, arg1, Terms::bvConcat)
    }

    override fun transform(expr: KBvExtractExpr): KExpr<KBvSort> = with(expr) {
        transform(value) { value: YicesTerm -> Terms.bvExtract(value, low, high) }
    }

    override fun transform(expr: KBvSignExtensionExpr): KExpr<KBvSort> = with(expr) {
        transform(value) { value: YicesTerm -> Terms.bvSignExtend(value, extensionSize) }
    }

    override fun transform(expr: KBvZeroExtensionExpr): KExpr<KBvSort> = with(expr) {
        transform(value) { value: YicesTerm -> Terms.bvZeroExtend(value, extensionSize) }
    }

    override fun transform(expr: KBvRepeatExpr): KExpr<KBvSort> = with(expr) {
        transform(value) { value: YicesTerm -> Terms.bvRepeat(value, repeatNumber) }
    }

    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>): KExpr<T> = with(expr) {
        transform(arg, shift, Terms::bvShl)
    }

    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>): KExpr<T> = with(expr) {
        transform(arg, shift, Terms::bvLshr)
    }

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>): KExpr<T> = with(expr) {
        transform(arg, shift, Terms::bvAshr)
    }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>): KExpr<T> = with(expr) {
        transform(arg, rotation) { arg0: YicesTerm, arg1: YicesTerm ->
            val size = expr.sort.sizeBits
            val bvSize = Terms.bvConst(size.toInt(), size.toLong())
            val rotationNumber = Terms.bvRem(arg1, bvSize)

            val left = Terms.bvShl(arg0, rotationNumber)
            val right = Terms.bvLshr(arg0, Terms.bvSub(bvSize, rotationNumber))

            Terms.bvOr(left, right)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>): KExpr<T> = with(expr) {
        transform(value) { value: YicesTerm -> Terms.bvRotateLeft(value, rotationNumber) }
    }

    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>): KExpr<T> = with(expr) {
        transform(arg, rotation) { arg0: YicesTerm, arg1: YicesTerm ->
            val size = expr.sort.sizeBits
            val bvSize = Terms.bvConst(size.toInt(), size.toLong())
            val rotationNumber = Terms.bvRem(arg1, bvSize)

            val left = Terms.bvShl(arg0, Terms.bvSub(bvSize, rotationNumber))
            val right = Terms.bvLshr(arg0, rotationNumber)

            Terms.bvOr(left, right)
        }
    }


    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>): KExpr<T> = with(expr) {
        transform(value) { value: YicesTerm -> Terms.bvRotateRight(value, rotationNumber) }
    }

    override fun transform(expr: KBv2IntExpr): KExpr<KIntSort> = with(expr) {
        transform(value) { value: YicesTerm ->
            val size = expr.value.sort.sizeBits.toInt()

            val args = (0 until size - 1).map {
                Terms.ifThenElse(
                    Terms.bvExtractBit(value, it),
                    Terms.intConst(BigInteger.valueOf(2).pow(it)),
                    Terms.ZERO
                )
            }

            var sign = Terms.ifThenElse(
                Terms.bvExtractBit(value, size - 1),
                Terms.intConst(BigInteger.valueOf(2).pow(size - 1)),
                Terms.ZERO
            )

            if (isSigned)
                sign = Terms.neg(sign)

            Terms.add(args + sign)
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
        transform(array, index, value, Terms::functionUpdate1)
    }

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> = with(expr) {
        transform(array, index) { array: YicesTerm, index: YicesTerm ->
            Terms.funApplication(array, index)
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayConst<D, R>): KExpr<KArraySort<D, R>> = with(expr) {
        transform(value) { value: YicesTerm ->
            Terms.lambda(listOf(Terms.newVariable(sort.internalizeSort())), value)
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KFunctionAsArray<D, R>): KExpr<KArraySort<D, R>> = with(expr) {
        transform { function.internalizeDecl() }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>>  {
        val transformedExpr = yicesCtx.substituteDecls(expr) { term ->
            with(term) {
                val newIndex = indexVarDecl.sort.mkFreshConstDecl(indexVarDecl.name)
                val transformer = KDeclSubstitutor(ctx).apply {
                    substitute(indexVarDecl, newIndex)
                }
                ctx.mkArrayLambda(newIndex, transformer.apply(body))
            }
        }

        return with(transformedExpr) {
            val variable = indexVarDecl.internalizeVariable()

            expr.transform(body) { body: YicesTerm ->
                Terms.lambda(listOf(variable), body)
            }
        }
    }

    override fun <T : KArithSort<T>> transform(expr: KAddArithExpr<T>): KExpr<T> = with(expr) {
        transformList(args) { args: Array<YicesTerm> -> Terms.add(args.toList()) }
    }

    override fun <T : KArithSort<T>> transform(expr: KMulArithExpr<T>): KExpr<T> = with(expr) {
        transformList(args) { args: Array<YicesTerm> -> Terms.mul(args.toList()) }
    }

    override fun <T : KArithSort<T>> transform(expr: KSubArithExpr<T>): KExpr<T> = with(expr) {
        transformList(args) { args: Array<YicesTerm> ->
            if (args.size == 1)
                args.first()
            else
                Terms.sub(args[0], Terms.add(args.drop(1)))
        }
    }

    override fun <T : KArithSort<T>> transform(expr: KUnaryMinusArithExpr<T>): KExpr<T> = with(expr) {
        transform(arg, Terms::neg)
    }

    override fun <T : KArithSort<T>> transform(expr: KDivArithExpr<T>): KExpr<T> = with(expr) {
        transform(lhs, rhs) { lhs: YicesTerm, rhs: YicesTerm ->
            when (sort) {
                is KIntSort -> Terms.idiv(lhs, rhs)
                else -> Terms.div(lhs, rhs)
            }
        }
    }

    override fun <T : KArithSort<T>> transform(expr: KPowerArithExpr<T>): KExpr<T> = with(expr) {
        transform(lhs, rhs, Terms::power)
    }

    override fun <T : KArithSort<T>> transform(expr: KLtArithExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(lhs, rhs, Terms::arithLt)
    }

    override fun <T : KArithSort<T>> transform(expr: KLeArithExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(lhs, rhs, Terms::arithLeq)
    }

    override fun <T : KArithSort<T>> transform(expr: KGtArithExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(lhs, rhs, Terms::arithGt)
    }

    override fun <T : KArithSort<T>> transform(expr: KGeArithExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(lhs, rhs, Terms::arithGeq)
    }

    override fun transform(expr: KModIntExpr): KExpr<KIntSort> = with(expr) {
        transform(lhs, rhs, Terms::imod)
    }

    override fun transform(expr: KRemIntExpr): KExpr<KIntSort> = with(expr) {
        transform(lhs, rhs) { lhs: YicesTerm, rhs: YicesTerm ->
            val sign = Terms.ifThenElse(Terms.arithLeq0(rhs), Terms.MINUS_ONE, Terms.ONE)
            val mod = Terms.imod(lhs, rhs)

            Terms.mul(mod, sign)
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
        transform{ Terms.intConst(value.toLong()) }
    }

    override fun transform(expr: KInt64NumExpr): KExpr<KIntSort> = with(expr) {
        transform { Terms.intConst(value) }
    }

    override fun transform(expr: KIntBigNumExpr): KExpr<KIntSort> = with(expr) {
        transform { Terms.intConst(value) }
    }

    override fun transform(expr: KToIntRealExpr): KExpr<KIntSort> = with(expr) {
        transform(arg, Terms::floor)
    }

    override fun transform(expr: KIsIntRealExpr): KExpr<KBoolSort> = with(expr) {
        transform(arg, Terms::isInt)
    }

    override fun transform(expr: KRealNumExpr): KExpr<KRealSort> = with(expr) {
        transform(numerator, denominator) { numerator: YicesTerm, denominator: YicesTerm ->
            Terms.div(numerator, denominator)
        }
    }

    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> = with (ctx) {
        internalizeQuantifier(expr, ::mkExistentialQuantifier) { body, variables ->
            Terms.exists(variables, body)
        }
    }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> = with (ctx) {
        internalizeQuantifier(expr, ::mkUniversalQuantifier) { body, variables ->
            Terms.forall(variables, body)
        }
    }

    @Suppress("UNCHECKED_CAST")
    private inline fun <T: KQuantifier> internalizeQuantifier(
        expr: T,
        crossinline constructor: (KExpr<KBoolSort>, List<KDecl<*>>) -> T,
        internalizer: (YicesTerm, List<YicesTerm>) -> YicesTerm
    ): T
    {
        val transformedExpr = yicesCtx.substituteDecls(expr) { term: T ->
            with(term) {
                val newBounds = bounds.map { it.sort.mkFreshConstDecl(it.name) }
                val transformer = KDeclSubstitutor(ctx).apply {
                    bounds.zip(newBounds).forEach { (bound, newBound) ->
                        substitute(bound as KDecl<KSort>, newBound)
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
