package org.ksmt.solver.z3

import com.microsoft.z3.ArithExpr
import com.microsoft.z3.ArithSort
import com.microsoft.z3.ArrayExpr
import com.microsoft.z3.BitVecExpr
import com.microsoft.z3.BoolExpr
import com.microsoft.z3.BoolSort
import com.microsoft.z3.Context
import com.microsoft.z3.Expr
import com.microsoft.z3.FPRMExpr
import com.microsoft.z3.FPSort
import com.microsoft.z3.FuncDecl
import com.microsoft.z3.IntExpr
import com.microsoft.z3.Sort
import com.microsoft.z3.mkBvNumeral
import com.microsoft.z3.mkExistsQuantifier
import com.microsoft.z3.mkForallQuantifier
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
import org.ksmt.expr.KBitVecNumberValue
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
import org.ksmt.expr.KConst
import org.ksmt.expr.KDistinctExpr
import org.ksmt.expr.KDivArithExpr
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFalse
import org.ksmt.expr.KFunctionApp
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
import org.ksmt.expr.KRealNumExpr
import org.ksmt.expr.KRemIntExpr
import org.ksmt.expr.KSubArithExpr
import org.ksmt.expr.KToIntRealExpr
import org.ksmt.expr.KToRealIntExpr
import org.ksmt.expr.KTrue
import org.ksmt.expr.KUnaryMinusArithExpr
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.KBvZeroExtensionExpr
import org.ksmt.expr.KFp128Value
import org.ksmt.expr.KFp16Value
import org.ksmt.expr.KFp32Value
import org.ksmt.expr.KFp64Value
import org.ksmt.expr.KFpAbsExpr
import org.ksmt.expr.KFpAddExpr
import org.ksmt.expr.KFpCustomSizeValue
import org.ksmt.expr.KFpDivExpr
import org.ksmt.expr.KFpEqualExpr
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
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.expr.KFpRoundingModeExpr
import org.ksmt.expr.KFpSqrtExpr
import org.ksmt.expr.KFpSubExpr
import org.ksmt.expr.KFpToBvExpr
import org.ksmt.expr.KFpToIEEEBvExpr
import org.ksmt.expr.KFpToRealExpr
import org.ksmt.expr.KFpValue
import org.ksmt.expr.KXorExpr
import org.ksmt.solver.util.KExprInternalizerBase
import org.ksmt.sort.KArithSort
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

@Suppress("SpreadOperator")
open class KZ3ExprInternalizer(
    override val ctx: KContext,
    private val z3Ctx: Context,
    private val z3InternCtx: KZ3InternalizationContext,
    private val sortInternalizer: KZ3SortInternalizer,
    private val declInternalizer: KZ3DeclInternalizer
) : KExprInternalizerBase<Expr<*>>() {

    override fun findInternalizedExpr(expr: KExpr<*>): Expr<*>? = z3InternCtx.findInternalizedExpr(expr)
    override fun saveInternalizedExpr(expr: KExpr<*>, internalized: Expr<*>) {
        z3InternCtx.internalizeExpr(expr) { internalized }
    }

    fun <T : KSort> KExpr<T>.internalize(): Expr<*> = internalizeExpr()

    fun <T : KDecl<*>> T.internalizeDecl(): FuncDecl<*> = accept(declInternalizer)

    fun <T : KSort> T.internalizeSort(): Sort = accept(sortInternalizer)

    override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> =
        error("Unexpected expr $expr")

    override fun <T : KSort> transform(expr: KFunctionApp<T>) = with(expr) {
        transformList(args) { args: Array<Expr<*>> ->
            z3Ctx.mkApp(decl.internalizeDecl(), *args)
        }
    }

    override fun <T : KSort> transform(expr: KConst<T>) = with(expr) {
        transform { z3Ctx.mkConst(decl.internalizeDecl()) }
    }

    override fun transform(expr: KAndExpr) = with(expr) {
        transformList(args) { args: Array<BoolExpr> -> z3Ctx.mkAnd(*args) }
    }

    override fun transform(expr: KOrExpr) = with(expr) {
        transformList(args) { args: Array<BoolExpr> -> z3Ctx.mkOr(*args) }
    }

    override fun transform(expr: KNotExpr) = with(expr) { transform(arg, z3Ctx::mkNot) }

    override fun transform(expr: KImpliesExpr) = with(expr) { transform(p, q, z3Ctx::mkImplies) }

    override fun transform(expr: KXorExpr) = with(expr) { transform(a, b, z3Ctx::mkXor) }

    override fun transform(expr: KTrue) = expr.transform { z3Ctx.mkTrue() }

    override fun transform(expr: KFalse) = expr.transform { z3Ctx.mkFalse() }

    override fun <T : KSort> transform(expr: KEqExpr<T>) = with(expr) { transform(lhs, rhs, z3Ctx::mkEq) }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>) = with(expr) {
        transformList(args) { args: Array<Expr<*>> -> z3Ctx.mkDistinct(*args) }
    }

    override fun <T : KSort> transform(expr: KIteExpr<T>) = with(expr) {
        transform<Expr<BoolSort>, Expr<Sort>, Expr<Sort>, KIteExpr<T>>(
            condition, trueBranch, falseBranch, z3Ctx::mkITE
        )
    }

    override fun <T : KBvSort> transformBitVecValue(expr: KBitVecValue<T>) = expr.transform {
        val sizeBits = expr.sort().sizeBits.toInt()
        when (expr) {
            is KBitVec1Value -> z3Ctx.mkBvNumeral(booleanArrayOf(expr.value))
            is KBitVec8Value, is KBitVec16Value, is KBitVec32Value -> {
                z3Ctx.mkBV((expr as KBitVecNumberValue<*, *>).numberValue.toInt(), sizeBits)
            }
            is KBitVec64Value -> z3Ctx.mkBV(expr.numberValue, sizeBits)
            is KBitVecCustomValue -> {
                val bits = expr.binaryStringValue.reversed().let { value ->
                    BooleanArray(value.length) { value[it] == '1' }
                }
                check(bits.size == sizeBits) { "bv bits size mismatch" }
                z3Ctx.mkBvNumeral(bits)
            }
            else -> error("Unknown bv expression class ${expr::class} in transformation method: $expr")
        }
    }

    override fun transform(expr: KBitVec1Value) = transformBitVecValue(expr)

    override fun transform(expr: KBitVec8Value) = transformBitVecValue(expr)

    override fun transform(expr: KBitVec16Value) = transformBitVecValue(expr)

    override fun transform(expr: KBitVec32Value) = transformBitVecValue(expr)

    override fun transform(expr: KBitVec64Value) = transformBitVecValue(expr)

    override fun transform(expr: KBitVecCustomValue) = transformBitVecValue(expr)

    override fun <T : KBvSort> transform(expr: KBvNotExpr<T>) =
        with(expr) { transform(value, z3Ctx::mkBVNot) }

    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>) =
        with(expr) { transform(value, z3Ctx::mkBVRedAND) }

    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>) =
        with(expr) { transform(value, z3Ctx::mkBVRedOR) }

    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVAND) }

    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVOR) }

    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVXOR) }

    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVNAND) }

    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVNOR) }

    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVXNOR) }

    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>) =
        with(expr) { transform(value, z3Ctx::mkBVNeg) }

    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVAdd) }

    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVSub) }

    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVMul) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVUDiv) }

    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVSDiv) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVURem) }

    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVSRem) }

    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVSMod) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVULT) }

    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVSLT) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVULE) }

    override fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVSLE) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVUGE) }

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVSGE) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVUGT) }

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVSGT) }

    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> =
        with(expr) { transform(arg0, arg1, z3Ctx::mkConcat) }

    override fun transform(expr: KBvExtractExpr) = with(expr) {
        transform(value) { value: BitVecExpr -> z3Ctx.mkExtract(high, low, value) }
    }

    override fun transform(expr: KBvSignExtensionExpr) = with(expr) {
        transform(value) { value: BitVecExpr -> z3Ctx.mkSignExt(i, value) }
    }

    override fun transform(expr: KBvZeroExtensionExpr) = with(expr) {
        transform(value) { value: BitVecExpr -> z3Ctx.mkZeroExt(i, value) }
    }

    override fun transform(expr: KBvRepeatExpr) = with(expr) {
        transform(value) { value: BitVecExpr -> z3Ctx.mkRepeat(i, value) }
    }

    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVSHL) }

    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVLSHR) }

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVASHR) }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVRotateLeft) }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>) = with(expr) {
        transform(value) { value: BitVecExpr -> z3Ctx.mkBVRotateLeft(i, value) }
    }

    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVRotateRight) }

    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>) = with(expr) {
        transform(value) { value: BitVecExpr -> z3Ctx.mkBVRotateRight(i, value) }
    }

    override fun transform(expr: KBv2IntExpr) = with(expr) {
        transform(value) { value: BitVecExpr -> z3Ctx.mkBV2Int(value, isSigned) }
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>) = with(expr) {
        transform(arg0, arg1) { a0: BitVecExpr, a1: BitVecExpr ->
            z3Ctx.mkBVAddNoOverflow(a0, a1, isSigned)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVAddNoUnderflow) }

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVSubNoOverflow) }

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1) { a0: BitVecExpr, a1: BitVecExpr ->
            z3Ctx.mkBVSubNoUnderflow(a0, a1, isSigned)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVSDivNoOverflow) }

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>) =
        with(expr) { transform(value, z3Ctx::mkBVNegNoOverflow) }

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>) = with(expr) {
        transform(arg0, arg1) { a0: BitVecExpr, a1: BitVecExpr ->
            z3Ctx.mkBVMulNoOverflow(a0, a1, isSigned)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>) =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVMulNoUnderflow) }

    override fun <T : KFpSort> transformFpValue(expr: KFpValue<T>): KExpr<T> = expr.transform {
        with(expr) {
            when (this) {
                is KFp16Value -> z3Ctx.mkFP(value, z3Ctx.mkFPSort16())
                is KFp32Value -> z3Ctx.mkFP(value, z3Ctx.mkFPSort32())
                is KFp64Value -> z3Ctx.mkFP(value, z3Ctx.mkFPSort64())
                is KFp128Value -> z3Ctx.mkFP(signBit, exponentValue, significandValue, z3Ctx.mkFPSort128())
                is KFpCustomSizeValue -> z3Ctx.mkFP(
                    signBit,
                    exponentValue,
                    significandValue,
                    z3Ctx.mkFPSort(sort().exponentBits.toInt(), sort().significandBits.toInt())
                )
                else -> error("Unexpected KFpValue type: ${this::class.qualifiedName}")
            }
        }
    }

    override fun transform(expr: KFp16Value): KExpr<KFp16Sort> = transformFpValue(expr)

    override fun transform(expr: KFp32Value): KExpr<KFp32Sort> = transformFpValue(expr)

    override fun transform(expr: KFp64Value): KExpr<KFp64Sort> = transformFpValue(expr)

    override fun transform(expr: KFp128Value): KExpr<KFp128Sort> = transformFpValue(expr)

    override fun transform(expr: KFpCustomSizeValue): KExpr<KFpSort> = transformFpValue(expr)

    override fun transform(expr: KFpRoundingModeExpr): KExpr<KFpRoundingModeSort> = with(expr) {
        transform { transformRoundingModeNumeral(value) }
    }

    override fun <T : KFpSort> transform(expr: KFpAbsExpr<T>): KExpr<T> = with(expr) {
        transform(value, z3Ctx::mkFPAbs)
    }

    override fun <T : KFpSort> transform(expr: KFpNegationExpr<T>): KExpr<T> = with(expr) {
        transform(value, z3Ctx::mkFPNeg)
    }

    override fun <T : KFpSort> transform(expr: KFpAddExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1) { left: Expr<FPSort>, right: Expr<FPSort> ->
            z3Ctx.mkFPAdd(transformRoundingModeNumeral(roundingMode), left, right)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpSubExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1) { left: Expr<FPSort>, right: Expr<FPSort> ->
            z3Ctx.mkFPSub(transformRoundingModeNumeral(roundingMode), left, right)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpMulExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1) { left: Expr<FPSort>, right: Expr<FPSort> ->
            z3Ctx.mkFPMul(transformRoundingModeNumeral(roundingMode), left, right)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpDivExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1) { left: Expr<FPSort>, right: Expr<FPSort> ->
            z3Ctx.mkFPDiv(transformRoundingModeNumeral(roundingMode), left, right)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpFusedMulAddExpr<T>): KExpr<T> =
        with(expr) {
            transform(arg0, arg1, arg2) { first: Expr<FPSort>, second: Expr<FPSort>, third: Expr<FPSort> ->
                z3Ctx.mkFPFMA(transformRoundingModeNumeral(roundingMode), first, second, third)
            }
        }

    override fun <T : KFpSort> transform(expr: KFpSqrtExpr<T>): KExpr<T> =
        with(expr) {
            transform(value) { z3Value: Expr<FPSort> ->
                z3Ctx.mkFPSqrt(transformRoundingModeNumeral(roundingMode), z3Value)
            }
        }

    override fun <T : KFpSort> transform(expr: KFpRemExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, z3Ctx::mkFPRem)
    }

    override fun <T : KFpSort> transform(expr: KFpRoundToIntegralExpr<T>): KExpr<T> =
        with(expr) {
            transform(value) { z3Value: Expr<FPSort> ->
                z3Ctx.mkFPRoundToIntegral(transformRoundingModeNumeral(roundingMode), z3Value)
            }
        }

    override fun <T : KFpSort> transform(expr: KFpMinExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, z3Ctx::mkFPMin)
    }

    override fun <T : KFpSort> transform(expr: KFpMaxExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, z3Ctx::mkFPMax)
    }

    override fun <T : KFpSort> transform(expr: KFpLessOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, z3Ctx::mkFPLEq)
    }

    override fun <T : KFpSort> transform(expr: KFpLessExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, z3Ctx::mkFPLt)
    }

    override fun <T : KFpSort> transform(expr: KFpGreaterOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, z3Ctx::mkFPGEq)
    }

    override fun <T : KFpSort> transform(expr: KFpGreaterExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, z3Ctx::mkFPGt)
    }

    override fun <T : KFpSort> transform(expr: KFpEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, z3Ctx::mkFPEq)
    }

    override fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, z3Ctx::mkFPIsNormal)
    }

    override fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, z3Ctx::mkFPIsSubnormal)
    }

    override fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, z3Ctx::mkFPIsZero)
    }

    override fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, z3Ctx::mkFPIsInfinite)
    }

    override fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, z3Ctx::mkFPIsNaN)
    }

    override fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, z3Ctx::mkFPIsNegative)
    }

    override fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, z3Ctx::mkFPIsPositive)
    }

    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort> =
        with(expr) {
            transform(value) { value: Expr<FPSort> ->
                z3Ctx.mkFPToBV(transformRoundingModeNumeral(expr.roundingMode), value, bvSize, isSigned)
            }
        }

    private fun transformRoundingModeNumeral(roundingMode: KFpRoundingMode): FPRMExpr =
        with(z3Ctx) {
            when (roundingMode) {
                KFpRoundingMode.RoundNearestTiesToEven -> mkFPRoundNearestTiesToEven()
                KFpRoundingMode.RoundNearestTiesToAway -> mkFPRoundNearestTiesToAway()
                KFpRoundingMode.RoundTowardNegative -> mkFPRoundTowardNegative()
                KFpRoundingMode.RoundTowardPositive -> mkFPRoundTowardPositive()
                KFpRoundingMode.RoundTowardZero -> mkFPRoundTowardZero()
            }
        }

    override fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort> = with(expr) {
        transform(value, z3Ctx::mkFPToReal)
    }

    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> = with(expr) {
        transform(value, z3Ctx::mkFPToIEEEBV)
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>) = with(expr) {
        transform<ArrayExpr<Sort, Sort>, Expr<Sort>, Expr<Sort>, KArrayStore<D, R>>(
            array, index, value, z3Ctx::mkStore
        )
    }

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>) = with(expr) {
        transform<ArrayExpr<Sort, Sort>, Expr<Sort>, KArraySelect<D, R>>(array, index, z3Ctx::mkSelect)
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayConst<D, R>) = with(expr) {
        transform(value) { value: Expr<*> -> z3Ctx.mkConstArray(sort.domain.internalizeSort(), value) }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>) = with(expr) {
        transform(body) { body: Expr<*> ->
            val internalizedIndex = indexVarDecl.internalizeDecl()
            z3Ctx.mkLambda(arrayOf(internalizedIndex.range), arrayOf(internalizedIndex.name), body)
        }
    }

    override fun <T : KArithSort<T>> transform(expr: KAddArithExpr<T>) = with(expr) {
        transformList(args) { args: Array<ArithExpr<*>> -> z3Ctx.mkAdd(*args) }
    }

    override fun <T : KArithSort<T>> transform(expr: KSubArithExpr<T>) = with(expr) {
        transformList(args) { args: Array<ArithExpr<*>> -> z3Ctx.mkSub(*args) }
    }

    override fun <T : KArithSort<T>> transform(expr: KMulArithExpr<T>) = with(expr) {
        transformList(args) { args: Array<ArithExpr<*>> -> z3Ctx.mkMul(*args) }
    }

    override fun <T : KArithSort<T>> transform(expr: KUnaryMinusArithExpr<T>) = with(expr) {
        transform<ArithExpr<ArithSort>, KUnaryMinusArithExpr<T>>(arg, z3Ctx::mkUnaryMinus)
    }

    override fun <T : KArithSort<T>> transform(expr: KDivArithExpr<T>) = with(expr) {
        transform<ArithExpr<ArithSort>, ArithExpr<ArithSort>, KDivArithExpr<T>>(lhs, rhs, z3Ctx::mkDiv)
    }

    override fun <T : KArithSort<T>> transform(expr: KPowerArithExpr<T>) = with(expr) {
        transform<ArithExpr<ArithSort>, ArithExpr<ArithSort>, KPowerArithExpr<T>>(lhs, rhs, z3Ctx::mkPower)
    }

    override fun <T : KArithSort<T>> transform(expr: KLtArithExpr<T>) = with(expr) { transform(lhs, rhs, z3Ctx::mkLt) }

    override fun <T : KArithSort<T>> transform(expr: KLeArithExpr<T>) = with(expr) { transform(lhs, rhs, z3Ctx::mkLe) }

    override fun <T : KArithSort<T>> transform(expr: KGtArithExpr<T>) = with(expr) { transform(lhs, rhs, z3Ctx::mkGt) }

    override fun <T : KArithSort<T>> transform(expr: KGeArithExpr<T>) = with(expr) { transform(lhs, rhs, z3Ctx::mkGe) }

    override fun transform(expr: KModIntExpr) = with(expr) { transform(lhs, rhs, z3Ctx::mkMod) }

    override fun transform(expr: KRemIntExpr) = with(expr) { transform(lhs, rhs, z3Ctx::mkRem) }

    override fun transform(expr: KToRealIntExpr) = with(expr) { transform(arg, z3Ctx::mkInt2Real) }

    override fun transform(expr: KInt32NumExpr) = with(expr) { transform { z3Ctx.mkInt(value) } }

    override fun transform(expr: KInt64NumExpr) = with(expr) { transform { z3Ctx.mkInt(value) } }

    override fun transform(expr: KIntBigNumExpr) = with(expr) { transform { z3Ctx.mkInt(value.toString()) } }

    override fun transform(expr: KToIntRealExpr) = with(expr) { transform(arg, z3Ctx::mkReal2Int) }

    override fun transform(expr: KIsIntRealExpr) = with(expr) { transform(arg, z3Ctx::mkIsInteger) }

    override fun transform(expr: KRealNumExpr) = with(expr) {
        transform(numerator, denominator) { numerator: IntExpr, denominator: IntExpr ->
            z3Ctx.mkDiv(z3Ctx.mkInt2Real(numerator), z3Ctx.mkInt2Real(denominator))
        }
    }

    override fun transform(expr: KExistentialQuantifier) = with(expr) {
        transform(body) { body: BoolExpr ->
            z3Ctx.mkExistsQuantifier(
                boundConstants = bounds.map { z3Ctx.mkConst(it.internalizeDecl()) }.toTypedArray(),
                body = body,
                weight = 0,
                patterns = arrayOf(),
                noPatterns = arrayOf(),
                quantifierId = null,
                skolemId = null
            )
        }
    }

    override fun transform(expr: KUniversalQuantifier) = with(expr) {
        transform(body) { body: BoolExpr ->
            z3Ctx.mkForallQuantifier(
                boundConstants = bounds.map { z3Ctx.mkConst(it.internalizeDecl()) }.toTypedArray(),
                body = body,
                weight = 0,
                patterns = arrayOf(),
                noPatterns = arrayOf(),
                quantifierId = null,
                skolemId = null
            )
        }
    }
}
