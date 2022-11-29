package org.ksmt.solver.z3

import com.microsoft.z3.Expr
import com.microsoft.z3.FuncDecl
import com.microsoft.z3.Native
import com.microsoft.z3.mkQuantifier
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
import org.ksmt.expr.KFpRoundingMode
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

@Suppress("SpreadOperator")
open class KZ3ExprInternalizer(
    val ctx: KContext,
    private val z3InternCtx: KZ3Context
) : KExprInternalizerBase<Long>() {

    val nCtx: Long = z3InternCtx.nCtx

    private val sortInternalizer = KZ3SortInternalizer(z3InternCtx)
    private val declInternalizer = KZ3DeclInternalizer(z3InternCtx, sortInternalizer)

    override fun findInternalizedExpr(expr: KExpr<*>): Long? = z3InternCtx.findInternalizedExpr(expr)

    override fun saveInternalizedExpr(expr: KExpr<*>, internalized: Long) {
        z3InternCtx.saveInternalizedExpr(expr, internalized)
    }

    fun <T : KSort> KExpr<T>.internalizeExprWrapped(): Expr<*> =
        z3InternCtx.nativeContext.wrapAST(internalizeExpr()) as Expr<*>

    fun <T : KDecl<*>> T.internalizeDeclWrapped(): FuncDecl<*> =
        z3InternCtx.nativeContext.wrapAST(internalizeDecl()) as FuncDecl<*>

    fun <T : KDecl<*>> T.internalizeDecl(): Long = accept(declInternalizer)

    fun <T : KSort> T.internalizeSort(): Long = accept(sortInternalizer)

    override fun <T : KSort> transform(expr: KFunctionApp<T>) = with(expr) {
        transformArray(args) { args ->
            Native.mkApp(nCtx, decl.internalizeDecl(), args.size, args)
        }
    }

    override fun <T : KSort> transform(expr: KConst<T>) = with(expr) {
        transform { Native.mkApp(nCtx, decl.internalizeDecl(), 0, null) }
    }

    override fun transform(expr: KAndExpr) = with(expr) {
        transformArray(args) { args -> Native.mkAnd(nCtx, args.size, args) }
    }

    override fun transform(expr: KOrExpr) = with(expr) {
        transformArray(args) { args -> Native.mkOr(nCtx, args.size, args) }
    }

    override fun transform(expr: KNotExpr) = with(expr) { transform(arg, Native::mkNot) }

    override fun transform(expr: KImpliesExpr) = with(expr) { transform(p, q, Native::mkImplies) }

    override fun transform(expr: KXorExpr) = with(expr) { transform(a, b, Native::mkXor) }

    override fun transform(expr: KTrue) = expr.transform { Native.mkTrue(nCtx) }

    override fun transform(expr: KFalse) = expr.transform { Native.mkFalse(nCtx) }

    override fun <T : KSort> transform(expr: KEqExpr<T>) = with(expr) { transform(lhs, rhs, Native::mkEq) }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>) = with(expr) {
        transformArray(args) { args -> Native.mkDistinct(nCtx, args.size, args) }
    }

    override fun <T : KSort> transform(expr: KIteExpr<T>) = with(expr) {
        transform(condition, trueBranch, falseBranch, Native::mkIte)
    }

    fun <T : KBvSort> transformBitVecValue(expr: KBitVecValue<T>) = expr.transform {
        when (expr) {
            is KBitVec1Value -> {
                val bits = booleanArrayOf(expr.value)
                Native.mkBvNumeral(nCtx, bits.size, bits)
            }

            is KBitVec8Value, is KBitVec16Value, is KBitVec32Value -> {
                val sort = expr.sort.internalizeSort()
                val intValue = (expr as KBitVecNumberValue<*, *>).numberValue.toInt()
                Native.mkInt(nCtx, intValue, sort)
            }

            is KBitVec64Value -> {
                val sort = expr.sort.internalizeSort()
                Native.mkInt64(nCtx, expr.numberValue, sort)
            }

            is KBitVecCustomValue -> {
                val bits = expr.binaryStringValue.reversed().let { value ->
                    BooleanArray(value.length) { value[it] == '1' }
                }
                Native.mkBvNumeral(nCtx, bits.size, bits)
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
        with(expr) { transform(value, Native::mkBvnot) }

    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>) =
        with(expr) { transform(value, Native::mkBvredand) }

    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>) =
        with(expr) { transform(value, Native::mkBvredor) }

    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvand) }

    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvor) }

    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvxor) }

    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvnand) }

    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvnor) }

    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvxnor) }

    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>) =
        with(expr) { transform(value, Native::mkBvneg) }

    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvadd) }

    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvsub) }

    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvmul) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvudiv) }

    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvsdiv) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvurem) }

    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvsrem) }

    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvsmod) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvult) }

    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvslt) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvule) }

    override fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvsle) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvuge) }

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvsge) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvugt) }

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvsgt) }

    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> =
        with(expr) { transform(arg0, arg1, Native::mkConcat) }

    override fun transform(expr: KBvExtractExpr) = with(expr) {
        transform(value) { value: Long -> Native.mkExtract(nCtx, high, low, value) }
    }

    override fun transform(expr: KBvSignExtensionExpr) = with(expr) {
        transform(value) { value: Long -> Native.mkSignExt(nCtx, extensionSize, value) }
    }

    override fun transform(expr: KBvZeroExtensionExpr) = with(expr) {
        transform(value) { value: Long -> Native.mkZeroExt(nCtx, extensionSize, value) }
    }

    override fun transform(expr: KBvRepeatExpr) = with(expr) {
        transform(value) { value: Long -> Native.mkRepeat(nCtx, repeatNumber, value) }
    }

    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvshl) }

    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvlshr) }

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvashr) }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkExtRotateLeft) }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>) = with(expr) {
        transform(value) { value: Long -> Native.mkRotateLeft(nCtx, rotationNumber, value) }
    }

    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkExtRotateRight) }

    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>) = with(expr) {
        transform(value) { value: Long -> Native.mkRotateRight(nCtx, rotationNumber, value) }
    }

    override fun transform(expr: KBv2IntExpr) = with(expr) {
        transform(value) { value: Long -> Native.mkBv2int(nCtx, value, isSigned) }
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>) = with(expr) {
        transform(arg0, arg1) { a0: Long, a1: Long ->
            Native.mkBvaddNoOverflow(nCtx, a0, a1, isSigned)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvaddNoUnderflow) }

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvsubNoOverflow) }

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>) = with(expr) {
        transform(arg0, arg1) { a0: Long, a1: Long ->
            Native.mkBvsubNoUnderflow(nCtx, a0, a1, isSigned)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvsdivNoOverflow) }

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>) =
        with(expr) { transform(value, Native::mkBvnegNoOverflow) }

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>) = with(expr) {
        transform(arg0, arg1) { a0: Long, a1: Long ->
            Native.mkBvmulNoOverflow(nCtx, a0, a1, isSigned)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>) =
        with(expr) { transform(arg0, arg1, Native::mkBvmulNoUnderflow) }

    override fun transform(expr: KFp16Value): KExpr<KFp16Sort> = expr.transform {
        val sort = expr.sort.internalizeSort()
        Native.mkFpaNumeralFloat(nCtx, expr.value, sort)
    }

    override fun transform(expr: KFp32Value): KExpr<KFp32Sort> = expr.transform {
        val sort = expr.sort.internalizeSort()
        Native.mkFpaNumeralFloat(nCtx, expr.value, sort)
    }

    override fun transform(expr: KFp64Value): KExpr<KFp64Sort> = expr.transform {
        val sort = expr.sort.internalizeSort()
        Native.mkFpaNumeralDouble(nCtx, expr.value, sort)
    }

    override fun transform(expr: KFp128Value): KExpr<KFp128Sort> = with(expr) {
        transform(ctx.mkBv(signBit), biasedExponent, significand, Native::mkFpaFp)
    }

    override fun transform(expr: KFpCustomSizeValue): KExpr<KFpSort> = with(expr) {
        transform(ctx.mkBv(signBit), biasedExponent, significand, Native::mkFpaFp)
    }

    override fun transform(expr: KFpRoundingModeExpr): KExpr<KFpRoundingModeSort> = with(expr) {
        transform { transformRoundingModeNumeral(value) }
    }

    override fun <T : KFpSort> transform(expr: KFpAbsExpr<T>): KExpr<T> = with(expr) {
        transform(value, Native::mkFpaAbs)
    }

    override fun <T : KFpSort> transform(expr: KFpNegationExpr<T>): KExpr<T> = with(expr) {
        transform(value, Native::mkFpaNeg)
    }

    override fun <T : KFpSort> transform(expr: KFpAddExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, arg0, arg1, Native::mkFpaAdd)
    }

    override fun <T : KFpSort> transform(expr: KFpSubExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, arg0, arg1, Native::mkFpaSub)
    }

    override fun <T : KFpSort> transform(expr: KFpMulExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, arg0, arg1, Native::mkFpaMul)
    }

    override fun <T : KFpSort> transform(expr: KFpDivExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, arg0, arg1, Native::mkFpaDiv)
    }

    override fun <T : KFpSort> transform(expr: KFpFusedMulAddExpr<T>): KExpr<T> =
        with(expr) {
            transform(roundingMode, arg0, arg1, arg2, Native::mkFpaFma)
        }

    override fun <T : KFpSort> transform(expr: KFpSqrtExpr<T>): KExpr<T> =
        with(expr) {
            transform(roundingMode, value, Native::mkFpaSqrt)
        }

    override fun <T : KFpSort> transform(expr: KFpRemExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, Native::mkFpaRem)
    }

    override fun <T : KFpSort> transform(expr: KFpRoundToIntegralExpr<T>): KExpr<T> =
        with(expr) {
            transform(roundingMode, value, Native::mkFpaRoundToIntegral)
        }

    override fun <T : KFpSort> transform(expr: KFpMinExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, Native::mkFpaMin)
    }

    override fun <T : KFpSort> transform(expr: KFpMaxExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, Native::mkFpaMax)
    }

    override fun <T : KFpSort> transform(expr: KFpLessOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, Native::mkFpaLeq)
    }

    override fun <T : KFpSort> transform(expr: KFpLessExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, Native::mkFpaLt)
    }

    override fun <T : KFpSort> transform(expr: KFpGreaterOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, Native::mkFpaGeq)
    }

    override fun <T : KFpSort> transform(expr: KFpGreaterExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, Native::mkFpaGt)
    }

    override fun <T : KFpSort> transform(expr: KFpEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, Native::mkFpaEq)
    }

    override fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, Native::mkFpaIsNormal)
    }

    override fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, Native::mkFpaIsSubnormal)
    }

    override fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, Native::mkFpaIsZero)
    }

    override fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, Native::mkFpaIsInfinite)
    }

    override fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, Native::mkFpaIsNan)
    }

    override fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, Native::mkFpaIsNegative)
    }

    override fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, Native::mkFpaIsPositive)
    }

    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort> =
        with(expr) {
            transform(roundingMode, value) { rm: Long, value: Long ->
                if (isSigned) {
                    Native.mkFpaToSbv(nCtx, rm, value, bvSize)
                } else {
                    Native.mkFpaToUbv(nCtx, rm, value, bvSize)
                }
            }
        }

    private fun transformRoundingModeNumeral(roundingMode: KFpRoundingMode): Long =
        when (roundingMode) {
            KFpRoundingMode.RoundNearestTiesToEven -> Native.mkFpaRoundNearestTiesToEven(nCtx)
            KFpRoundingMode.RoundNearestTiesToAway -> Native.mkFpaRoundNearestTiesToAway(nCtx)
            KFpRoundingMode.RoundTowardNegative -> Native.mkFpaRoundTowardNegative(nCtx)
            KFpRoundingMode.RoundTowardPositive -> Native.mkFpaRoundTowardPositive(nCtx)
            KFpRoundingMode.RoundTowardZero -> Native.mkFpaRoundTowardZero(nCtx)
        }

    override fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort> = with(expr) {
        transform(value, Native::mkFpaToReal)
    }

    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> = with(expr) {
        transform(value, Native::mkFpaToIeeeBv)
    }

    override fun <T : KFpSort> transform(expr: KFpFromBvExpr<T>): KExpr<T> = with(expr) {
        transform(sign, biasedExponent, significand, Native::mkFpaFp)
    }

    override fun <T : KFpSort> transform(expr: KFpToFpExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, value) { rm: Long, value: Long ->
            val fpSort = sort.internalizeSort()
            Native.mkFpaToFpFloat(nCtx, rm, value, fpSort)
        }
    }

    override fun <T : KFpSort> transform(expr: KRealToFpExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, value) { rm: Long, value: Long ->
            val fpSort = sort.internalizeSort()
            Native.mkFpaToFpReal(nCtx, rm, value, fpSort)
        }
    }

    override fun <T : KFpSort> transform(expr: KBvToFpExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, value) { rm: Long, value: Long ->
            val fpSort = sort.internalizeSort()
            if (signed) {
                Native.mkFpaToFpSigned(nCtx, rm, value, fpSort)
            } else {
                Native.mkFpaToFpUnsigned(nCtx, rm, value, fpSort)
            }
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>) = with(expr) {
        transform(array, index, value, Native::mkStore)
    }

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>) = with(expr) {
        transform(array, index, Native::mkSelect)
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayConst<D, R>) = with(expr) {
        transform(value) { value: Long -> Native.mkConstArray(nCtx, sort.domain.internalizeSort(), value) }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>) = with(expr) {
        transform(body) { body: Long ->
            val indexSort = indexVarDecl.sort.internalizeSort()
            val indexName = Native.mkStringSymbol(nCtx, indexVarDecl.name)
            Native.mkLambda(nCtx, 1, longArrayOf(indexSort), longArrayOf(indexName), body)
        }
    }

    override fun <T : KArithSort<T>> transform(expr: KAddArithExpr<T>) = with(expr) {
        transformArray(args) { args -> Native.mkAdd(nCtx, args.size, args) }
    }

    override fun <T : KArithSort<T>> transform(expr: KSubArithExpr<T>) = with(expr) {
        transformArray(args) { args -> Native.mkSub(nCtx, args.size, args) }
    }

    override fun <T : KArithSort<T>> transform(expr: KMulArithExpr<T>) = with(expr) {
        transformArray(args) { args -> Native.mkMul(nCtx, args.size, args) }
    }

    override fun <T : KArithSort<T>> transform(expr: KUnaryMinusArithExpr<T>) = with(expr) {
        transform(arg, Native::mkUnaryMinus)
    }

    override fun <T : KArithSort<T>> transform(expr: KDivArithExpr<T>) = with(expr) {
        transform(lhs, rhs, Native::mkDiv)
    }

    override fun <T : KArithSort<T>> transform(expr: KPowerArithExpr<T>) = with(expr) {
        transform(lhs, rhs, Native::mkPower)
    }

    override fun <T : KArithSort<T>> transform(expr: KLtArithExpr<T>) = with(expr) { transform(lhs, rhs, Native::mkLt) }

    override fun <T : KArithSort<T>> transform(expr: KLeArithExpr<T>) = with(expr) { transform(lhs, rhs, Native::mkLe) }

    override fun <T : KArithSort<T>> transform(expr: KGtArithExpr<T>) = with(expr) { transform(lhs, rhs, Native::mkGt) }

    override fun <T : KArithSort<T>> transform(expr: KGeArithExpr<T>) = with(expr) { transform(lhs, rhs, Native::mkGe) }

    override fun transform(expr: KModIntExpr) = with(expr) { transform(lhs, rhs, Native::mkMod) }

    override fun transform(expr: KRemIntExpr) = with(expr) { transform(lhs, rhs, Native::mkRem) }

    override fun transform(expr: KToRealIntExpr) = with(expr) { transform(arg, Native::mkInt2real) }

    override fun transform(expr: KInt32NumExpr) = with(expr) {
        transform { Native.mkInt(nCtx, value, ctx.intSort.internalizeSort()) }
    }

    override fun transform(expr: KInt64NumExpr) = with(expr) {
        transform { Native.mkInt64(nCtx, value, ctx.intSort.internalizeSort()) }
    }

    override fun transform(expr: KIntBigNumExpr) = with(expr) {
        transform { Native.mkNumeral(nCtx, value.toString(), ctx.intSort.internalizeSort()) }
    }

    override fun transform(expr: KToIntRealExpr) = with(expr) { transform(arg, Native::mkReal2int) }

    override fun transform(expr: KIsIntRealExpr) = with(expr) { transform(arg, Native::mkIsInt) }

    override fun transform(expr: KRealNumExpr) = with(expr) {
        transform(ctx.mkIntToReal(numerator), ctx.mkIntToReal(denominator), Native::mkDiv)
    }

    private fun transformQuantifier(expr: KQuantifier, isUniversal: Boolean) = with(expr) {
        val boundConstants = bounds.map { ctx.mkConstApp(it) }
        transformArray(boundConstants + body) { args ->
            val body = args.last()
            val bounds = args.copyOf(args.size - 1)
            mkQuantifier(
                ctx = nCtx,
                isUniversal = isUniversal,
                boundConsts = bounds,
                body = body,
                weight = 0,
                patterns = longArrayOf(),
            )
        }
    }

    override fun transform(expr: KExistentialQuantifier) = transformQuantifier(expr, isUniversal = false)

    override fun transform(expr: KUniversalQuantifier) = transformQuantifier(expr, isUniversal = true)

    override fun <D : KSort, R : KSort> transform(expr: KFunctionAsArray<D, R>): KExpr<KArraySort<D, R>> {
        TODO("KFunctionAsArray internalization is not implemented in z3")
    }

    inline fun <S : KExpr<*>> S.transform(
        arg: KExpr<*>,
        operation: (Long, Long) -> Long
    ): S = transform(arg) { a: Long ->
        operation(nCtx, a)
    }

    inline fun <S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        operation: (Long, Long, Long) -> Long
    ): S = transform(arg0, arg1) { a0: Long, a1: Long ->
        operation(nCtx, a0, a1)
    }

    inline fun <S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        arg2: KExpr<*>,
        operation: (Long, Long, Long, Long) -> Long
    ): S = transform(arg0, arg1, arg2) { a0: Long, a1: Long, a2: Long ->
        operation(nCtx, a0, a1, a2)
    }

    inline fun <S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        arg2: KExpr<*>,
        arg3: KExpr<*>,
        operation: (Long, Long, Long, Long, Long) -> Long
    ): S = transform(arg0, arg1, arg2, arg3) { a0: Long, a1: Long, a2: Long, a3: Long ->
        operation(nCtx, a0, a1, a2, a3)
    }

    @Suppress("ArrayPrimitive")
    inline fun <S : KExpr<*>> S.transformArray(
        args: List<KExpr<*>>,
        operation: (LongArray) -> Long
    ): S = transformList(args) { a: Array<Long> -> operation(a.toLongArray()) }

}
