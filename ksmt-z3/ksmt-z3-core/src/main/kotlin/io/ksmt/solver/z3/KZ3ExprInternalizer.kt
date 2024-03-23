package io.ksmt.solver.z3

import com.microsoft.z3.Native
import com.microsoft.z3.mkQuantifier
import io.ksmt.KContext
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
import io.ksmt.expr.KBitVecNumberValue
import io.ksmt.expr.KBitVecValue
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
import io.ksmt.expr.KFpRoundingMode
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
import io.ksmt.expr.KQuantifier
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
import io.ksmt.solver.util.KExprLongInternalizerBase
import io.ksmt.sort.KArithSort
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFp128Sort
import io.ksmt.sort.KFp16Sort
import io.ksmt.sort.KFp32Sort
import io.ksmt.sort.KFp64Sort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort

open class KZ3ExprInternalizer(
    val ctx: KContext,
    private val z3InternCtx: KZ3Context
) : KExprLongInternalizerBase() {

    @JvmField
    val nCtx: Long = z3InternCtx.nCtx

    private val sortInternalizer = KZ3SortInternalizer(z3InternCtx)
    private val declInternalizer = KZ3DeclInternalizer(z3InternCtx, sortInternalizer)

    override fun findInternalizedExpr(expr: KExpr<*>): Long =
        z3InternCtx.findInternalizedExpr(expr)

    override fun saveInternalizedExpr(expr: KExpr<*>, internalized: Long) {
        z3InternCtx.saveInternalizedExpr(expr, internalized)
    }

    fun <T : KDecl<*>> T.internalizeDecl(): Long = declInternalizer.internalizeZ3Decl(this)

    fun <T : KSort> T.internalizeSort(): Long = sortInternalizer.internalizeZ3Sort(this)

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

    override fun transform(expr: KAndBinaryExpr) = with(expr) {
        transform(lhs, rhs) { l, r -> Native.mkAnd(nCtx, 2, longArrayOf(l, r)) }
    }

    override fun transform(expr: KOrExpr) = with(expr) {
        transformArray(args) { args -> Native.mkOr(nCtx, args.size, args) }
    }

    override fun transform(expr: KOrBinaryExpr) = with(expr) {
        transform(lhs, rhs) { l, r -> Native.mkOr(nCtx, 2, longArrayOf(l, r)) }
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

            is KBitVec8Value -> {
                val sort = expr.sort.internalizeSort()
                Native.mkInt(nCtx, expr.byteValue.toInt(), sort)
            }

            is KBitVec16Value -> {
                val sort = expr.sort.internalizeSort()
                Native.mkInt(nCtx, expr.shortValue.toInt(), sort)
            }

            is KBitVec32Value -> {
                val sort = expr.sort.internalizeSort()
                Native.mkInt(nCtx, expr.intValue, sort)
            }

            is KBitVec64Value -> {
                val sort = expr.sort.internalizeSort()
                Native.mkInt64(nCtx, expr.longValue, sort)
            }

            is KBitVecCustomValue -> {
                val bits = BooleanArray(expr.sizeBits.toInt()) { expr.value.testBit(it) }
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
        with(expr) { transform(arg, shift, Native::mkBvshl) }

    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>) =
        with(expr) { transform(arg, shift, Native::mkBvlshr) }

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>) =
        with(expr) { transform(arg, shift, Native::mkBvashr) }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>) =
        with(expr) { transform(arg, rotation, Native::mkExtRotateLeft) }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>) = with(expr) {
        transform(value) { value: Long -> Native.mkRotateLeft(nCtx, rotationNumber, value) }
    }

    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>) =
        with(expr) { transform(arg, rotation, Native::mkExtRotateRight) }

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

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Store<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = with(expr) {
        transform(array, index0, index1, value) { array: Long, i0: Long, i1: Long, value: Long ->
            val indices = longArrayOf(i0, i1)
            Native.mkStoreN(nCtx, array, indices.size, indices, value)
        }
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Store<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = with(expr) {
        transformArray(listOf(array, index0, index1, index2, value)) { args ->
            val (array, i0, i1, i2, value) = args
            val indices = longArrayOf(i0, i1, i2)
            Native.mkStoreN(nCtx, array, indices.size, indices, value)
        }
    }

    override fun <R : KSort> transform(expr: KArrayNStore<R>): KExpr<KArrayNSort<R>> = with(expr) {
        transformArray(indices + listOf(array, value)) { args ->
            val value = args[args.lastIndex]
            val array = args[args.lastIndex - 1]
            val indices = args.copyOf(args.size - 2)
            Native.mkStoreN(nCtx, array, indices.size, indices, value)
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>) = with(expr) {
        transform(array, index, Native::mkSelect)
    }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Select<D0, D1, R>
    ): KExpr<R> = with(expr) {
        transform(array, index0, index1) { array: Long, i0: Long, i1: Long ->
            val indices = longArrayOf(i0, i1)
            Native.mkSelectN(nCtx, array, indices.size, indices)
        }
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Select<D0, D1, D2, R>
    ): KExpr<R>  = with(expr) {
        transform(array, index0, index1, index2) { array: Long, i0: Long, i1: Long, i2: Long ->
            val indices = longArrayOf(i0, i1, i2)
            Native.mkSelectN(nCtx, array, indices.size, indices)
        }
    }

    override fun <R : KSort> transform(expr: KArrayNSelect<R>): KExpr<R> = with(expr) {
        transformArray(indices + array) { args ->
            val array = args.last()
            val indices = args.copyOf(args.size - 1)
            Native.mkSelectN(nCtx, array, indices.size, indices)
        }
    }

    override fun <A: KArraySortBase<R>, R : KSort> transform(expr: KArrayConst<A, R>) = with(expr) {
        transform(value) { value: Long ->
            mkConstArray(sort, value)
        }
    }

    private fun mkConstArray(sort: KArraySortBase<*>, value: Long): Long =
        if (sort is KArraySort<*, *>) {
            Native.mkConstArray(nCtx, sort.domain.internalizeSort(), value)
        } else {
            ensureArraySortInternalized(sort)

            val domain = sort.domainSorts.let { domain ->
                LongArray(domain.size) { domain[it].internalizeSort() }
            }
            val domainNames = LongArray(sort.domainSorts.size) { Native.mkIntSymbol(nCtx, it) }
            Native.mkLambda(nCtx, domain.size, domain, domainNames, value)
        }

    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>) = with(expr) {
        transform(body, ctx.mkConstApp(indexVarDecl)) { body: Long, index: Long ->
            ensureArraySortInternalized(expr.sort)

            val indices = longArrayOf(index)
            Native.mkLambdaConst(nCtx, indices.size, indices, body)
        }
    }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Lambda<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = with(expr) {
        transform(
            body, ctx.mkConstApp(indexVar0Decl), ctx.mkConstApp(indexVar1Decl)
        ) { body: Long, index0: Long, index1: Long ->
            ensureArraySortInternalized(expr.sort)

            val indices = longArrayOf(index0, index1)
            Native.mkLambdaConst(nCtx, indices.size, indices, body)
        }
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Lambda<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>>  = with(expr) {
        transform(
            body, ctx.mkConstApp(indexVar0Decl), ctx.mkConstApp(indexVar1Decl), ctx.mkConstApp(indexVar2Decl)
        ) { body: Long, index0: Long, index1: Long, index2: Long ->
            ensureArraySortInternalized(expr.sort)

            val indices = longArrayOf(index0, index1, index2)
            Native.mkLambdaConst(nCtx, indices.size, indices, body)
        }
    }

    override fun <R : KSort> transform(expr: KArrayNLambda<R>): KExpr<KArrayNSort<R>> = with(expr) {
        transformArray(indexVarDeclarations.map { ctx.mkConstApp(it) } + body) { args ->
            ensureArraySortInternalized(expr.sort)

            val body = args.last()
            val indices = args.copyOf(args.size - 1)
            Native.mkLambdaConst(nCtx, indices.size, indices, body)
        }
    }

    /**
     * Z3 incorrectly manage array sort references on lambda expression creation.
     * If the created lambda expression is the only reference to the array sort,
     * it results in native error on context deletion. To bypass this issue
     * we ensure that array sort is referenced not only in the created lambda
     * */
    private fun ensureArraySortInternalized(sort: KArraySortBase<*>) {
        sort.internalizeSort()
    }

    override fun <T : KArithSort> transform(expr: KAddArithExpr<T>) = with(expr) {
        transformArray(args) { args -> Native.mkAdd(nCtx, args.size, args) }
    }

    override fun <T : KArithSort> transform(expr: KSubArithExpr<T>) = with(expr) {
        transformArray(args) { args -> Native.mkSub(nCtx, args.size, args) }
    }

    override fun <T : KArithSort> transform(expr: KMulArithExpr<T>) = with(expr) {
        transformArray(args) { args -> Native.mkMul(nCtx, args.size, args) }
    }

    override fun <T : KArithSort> transform(expr: KUnaryMinusArithExpr<T>) = with(expr) {
        transform(arg, Native::mkUnaryMinus)
    }

    override fun <T : KArithSort> transform(expr: KDivArithExpr<T>) = with(expr) {
        transform(lhs, rhs, Native::mkDiv)
    }

    override fun <T : KArithSort> transform(expr: KPowerArithExpr<T>) = with(expr) {
        transform(lhs, rhs, Native::mkPower)
    }

    override fun <T : KArithSort> transform(expr: KLtArithExpr<T>) = with(expr) { transform(lhs, rhs, Native::mkLt) }

    override fun <T : KArithSort> transform(expr: KLeArithExpr<T>) = with(expr) { transform(lhs, rhs, Native::mkLe) }

    override fun <T : KArithSort> transform(expr: KGtArithExpr<T>) = with(expr) { transform(lhs, rhs, Native::mkGt) }

    override fun <T : KArithSort> transform(expr: KGeArithExpr<T>) = with(expr) { transform(lhs, rhs, Native::mkGe) }

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
        transform(numerator, denominator) { numeratorExpr: Long, denominatorExpr: Long ->
            val realNumerator = z3InternCtx.temporaryAst(Native.mkInt2real(nCtx, numeratorExpr))
            val realDenominator = z3InternCtx.temporaryAst(Native.mkInt2real(nCtx, denominatorExpr))
            Native.mkDiv(nCtx, realNumerator, realDenominator).also {
                z3InternCtx.releaseTemporaryAst(realNumerator)
                z3InternCtx.releaseTemporaryAst(realDenominator)
            }
        }
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

    override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KFunctionAsArray<A, R>): KExpr<A> = with(expr) {
        transform { Native.mkAsArray(nCtx, function.internalizeDecl()) }
    }

    /**
     * There is no way in Z3 API to mark uninterpreted constant as value.
     *
     * To overcome this we apply the following scheme:
     * 1. Internalize value `x` of a sort T as normal constant.
     * 2. Associate unique interpreted value `i` with this constant.
     * Currently, we use integer values.
     * 3. Introduce `interpreter` function `F` of type T -> Int.
     * We introduce one function for each uninterpreted sort.
     * 4. Assert expression `(= i (F x))` to the solver.
     * Since all Int values are known to be distinct, this
     * assertion forces that all values of T are also distinct.
     * */
    override fun transform(expr: KUninterpretedSortValue): KExpr<KUninterpretedSort> = with(expr) {
        transform(ctx.mkIntNum(expr.valueIdx)) { intValueExpr ->
            val nativeSort = sort.internalizeSort()
            val valueDecl = z3InternCtx.saveUninterpretedSortValueDecl(
                Native.mkFreshFuncDecl(nCtx, "value", 0, null, nativeSort),
                expr
            )

            Native.mkApp(nCtx, valueDecl, 0, null).also {
                // Force expression save to perform `incRef` and prevent possible reference counting issues
                saveInternalizedExpr(expr, it)

                z3InternCtx.registerUninterpretedSortValue(expr, intValueExpr, it) {
                    val descriptorSort = ctx.intSort.internalizeSort()
                    z3InternCtx.saveUninterpretedSortValueInterpreter(
                        Native.mkFreshFuncDecl(nCtx, "interpreter", 1, longArrayOf(nativeSort), descriptorSort)
                    )
                }
            }
        }
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

    inline fun <S : KExpr<*>> S.transformArray(
        args: List<KExpr<*>>,
        operation: (LongArray) -> Long
    ): S = transformList(args) { a: LongArray -> operation(a) }

}
