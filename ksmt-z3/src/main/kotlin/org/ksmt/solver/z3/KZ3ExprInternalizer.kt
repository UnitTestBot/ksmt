package org.ksmt.solver.z3

import com.microsoft.z3.ArithExpr
import com.microsoft.z3.ArrayExpr
import com.microsoft.z3.BitVecExpr
import com.microsoft.z3.BoolExpr
import com.microsoft.z3.Context
import com.microsoft.z3.Expr
import com.microsoft.z3.FuncDecl
import com.microsoft.z3.IntExpr
import com.microsoft.z3.RealExpr
import com.microsoft.z3.Sort
import com.microsoft.z3.mkExistsQuantifier
import com.microsoft.z3.mkForallQuantifier
import java.math.BigInteger
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
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KBitVecNumberValue
import org.ksmt.expr.KBv2IntExpr
import org.ksmt.expr.KBvAddExpr
import org.ksmt.expr.KBvAddNoOverflowExpr
import org.ksmt.expr.KBvAddNoUnderflowExpr
import org.ksmt.expr.KBvAndExpr
import org.ksmt.expr.KBvArithShiftRightExpr
import org.ksmt.expr.KBvDivNoOverflowExpr
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
import org.ksmt.expr.KBvRotateLeftExpr
import org.ksmt.expr.KBvRotateLeftIndexedExpr
import org.ksmt.expr.KBvRotateRightExpr
import org.ksmt.expr.KBvRotateRightIndexedExpr
import org.ksmt.expr.KBvShiftLeftExpr
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
import org.ksmt.expr.KBvConcatExpr
import org.ksmt.expr.KConst
import org.ksmt.expr.KDistinctExpr
import org.ksmt.expr.KDivArithExpr
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KBvExtractExpr
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
import org.ksmt.expr.KBvRepeatExpr
import org.ksmt.expr.KBvSignExtensionExpr
import org.ksmt.expr.KSubArithExpr
import org.ksmt.expr.KToIntRealExpr
import org.ksmt.expr.KToRealIntExpr
import org.ksmt.expr.KTransformer
import org.ksmt.expr.KTrue
import org.ksmt.expr.KUnaryMinusArithExpr
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.KBvZeroExtensionExpr
import org.ksmt.expr.KXorExpr
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv16Sort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBv64Sort
import org.ksmt.sort.KBv8Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KSort

@Suppress("TooManyFunctions", "SpreadOperator")
open class KZ3ExprInternalizer(
    override val ctx: KContext,
    private val z3Ctx: Context,
    val z3InternCtx: KZ3InternalizationContext,
    private val sortInternalizer: KZ3SortInternalizer,
    private val declInternalizer: KZ3DeclInternalizer
) : KTransformer {

    fun <T : KDecl<*>> T.internalize(): FuncDecl = accept(declInternalizer)

    @Suppress("MemberVisibilityCanBePrivate")
    fun <T : KSort> T.internalize(): Sort = accept(sortInternalizer)

    fun <T : KSort> KExpr<T>.internalize(): Expr {
        accept(this@KZ3ExprInternalizer)
        return z3InternCtx[this].getOrError()
    }

    override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> =
        error("Unexpected expr $expr")

    override fun <T : KSort> transform(expr: KFunctionApp<T>) = expr.internalizeExpr {
        z3Ctx.mkApp(expr.decl.internalize(), *args.map { it.internalize() }.toTypedArray())
    }

    override fun <T : KSort> transform(expr: KConst<T>) = expr.internalizeExpr {
        z3Ctx.mkConst(decl.internalize())
    }

    override fun transform(expr: KAndExpr) = expr.internalizeExpr {
        z3Ctx.mkAnd(*args.map { it.internalize() as BoolExpr }.toTypedArray())
    }

    override fun transform(expr: KOrExpr) = expr.internalizeExpr {
        z3Ctx.mkOr(*args.map { it.internalize() as BoolExpr }.toTypedArray())
    }

    override fun transform(expr: KNotExpr) = expr.internalizeExpr {
        z3Ctx.mkNot(arg.internalize() as BoolExpr)
    }

    override fun transform(expr: KImpliesExpr) = expr.internalizeExpr {
        z3Ctx.mkImplies(p.internalize() as BoolExpr, q.internalize() as BoolExpr)
    }

    override fun transform(expr: KXorExpr) = expr.internalizeExpr {
        z3Ctx.mkXor(a.internalize() as BoolExpr, b.internalize() as BoolExpr)
    }

    override fun transform(expr: KTrue) = expr.internalizeExpr {
        z3Ctx.mkTrue()
    }

    override fun transform(expr: KFalse) = expr.internalizeExpr {
        z3Ctx.mkFalse()
    }

    override fun <T : KSort> transform(expr: KEqExpr<T>) = expr.internalizeExpr {
        z3Ctx.mkEq(lhs.internalize(), rhs.internalize())
    }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>) = expr.internalizeExpr {
        z3Ctx.mkDistinct(*args.map { it.internalize() }.toTypedArray())
    }

    override fun <T : KSort> transform(expr: KIteExpr<T>) = expr.internalizeExpr {
        z3Ctx.mkITE(
            condition.internalize() as BoolExpr,
            trueBranch.internalize(),
            falseBranch.internalize()
        )
    }

    override fun <T : KBvSort> transformBitVecValue(expr: KBitVecValue<T>): KExpr<T> = expr.internalizeExpr {
        val sizeBits = expr.sort().sizeBits.toInt()
        when (expr) {
            is KBitVec8Value, is KBitVec16Value, is KBitVec32Value -> {
                z3Ctx.mkBV((expr as KBitVecNumberValue<*, *>).numberValue.toInt(), sizeBits)
            }
            is KBitVec64Value -> z3Ctx.mkBV(expr.numberValue, sizeBits)
            is KBitVecCustomValue -> {
//              // TODO a problem with negative numbers. Here BigInteger treat `-15` as big positive number
                val decimalString = BigInteger(expr.binaryStringValue, 2).toString(10)
                z3Ctx.mkBV(decimalString, sizeBits)
            }
            else -> error("Unknown bv expression class ${expr::class} in transformation method: ${expr.print()}")
        }
    }

    override fun transform(expr: KBitVec1Value): KExpr<KBv1Sort> = transformBitVecValue(expr)

    override fun transform(expr: KBitVec8Value): KExpr<KBv8Sort> = transformBitVecValue(expr)

    override fun transform(expr: KBitVec16Value): KExpr<KBv16Sort> = transformBitVecValue(expr)

    override fun transform(expr: KBitVec32Value): KExpr<KBv32Sort> = transformBitVecValue(expr)

    override fun transform(expr: KBitVec64Value): KExpr<KBv64Sort> = transformBitVecValue(expr)

    override fun transform(expr: KBitVecCustomValue): KExpr<KBvSort> = transformBitVecValue(expr)

    private fun <T : KBvSort, R : Expr, S : KSort> KExpr<S>.transform(
        arg: KExpr<T>,
        operation: (BitVecExpr) -> R
    ): KExpr<S> = internalizeExpr {
        operation(arg.internalize() as BitVecExpr)
    }

    private fun <R : Expr, S : KSort, T1 : KBvSort, T2 : KBvSort> KExpr<S>.transform(
        arg0: KExpr<T1>,
        arg1: KExpr<T2>,
        operation: (BitVecExpr, BitVecExpr) -> R
    ): KExpr<S> = internalizeExpr {
        operation(arg0.internalize() as BitVecExpr, arg1.internalize() as BitVecExpr)
    }

    override fun <T : KBvSort> transform(expr: KBvNotExpr<T>): KExpr<T> =
        with(expr) { transform(value, z3Ctx::mkBVNot) }

    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>): KExpr<KBv1Sort> = with(expr)
    { transform(value, z3Ctx::mkBVRedAND) }

    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>): KExpr<KBv1Sort> = with(expr)
    { transform(value, z3Ctx::mkBVRedOR) }

    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>): KExpr<T> = with(expr)
    { transform(arg0, arg1, z3Ctx::mkBVAND) }

    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>): KExpr<T> = with(expr)
    { transform(arg0, arg1, z3Ctx::mkBVOR) }

    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>): KExpr<T> = with(expr)
    { transform(arg0, arg1, z3Ctx::mkBVXOR) }

    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>): KExpr<T> = with(expr)
    { transform(arg0, arg1, z3Ctx::mkBVNAND) }

    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>): KExpr<T> = with(expr)
    { transform(arg0, arg1, z3Ctx::mkBVNOR) }

    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>): KExpr<T> = with(expr)
    { transform(arg0, arg1, z3Ctx::mkBVXNOR) }

    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>): KExpr<T> = with(expr)
    { transform(value, z3Ctx::mkBVNeg) }

    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>): KExpr<T> = with(expr)
    { transform(arg0, arg1, z3Ctx::mkBVAdd) }

    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>): KExpr<T> = with(expr)
    { transform(arg0, arg1, z3Ctx::mkBVSub) }

    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>): KExpr<T> = with(expr)
    { transform(arg0, arg1, z3Ctx::mkBVMul) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>): KExpr<T> = with(expr)
    { transform(arg0, arg1, z3Ctx::mkBVUDiv) }

    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>): KExpr<T> = with(expr)
    { transform(arg0, arg1, z3Ctx::mkBVSDiv) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>): KExpr<T> = with(expr)
    { transform(arg0, arg1, z3Ctx::mkBVURem) }

    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>): KExpr<T> = with(expr)
    { transform(arg0, arg1, z3Ctx::mkBVSRem) }

    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>): KExpr<T> = with(expr)
    { transform(arg0, arg1, z3Ctx::mkBVSMod) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>): KExpr<KBoolSort> =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVULT) }

    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>): KExpr<KBoolSort> =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVSLT) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>): KExpr<KBoolSort> =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVULE) }

    override fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>): KExpr<KBoolSort> =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVSLE) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVUGE) }

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>): KExpr<KBoolSort> =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVSGE) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>): KExpr<KBoolSort> =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVUGT) }

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>): KExpr<KBoolSort> =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVSGT) }

    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> = with(expr) { transform(arg0, arg1, z3Ctx::mkConcat) }

    override fun transform(expr: KBvExtractExpr): KExpr<KBvSort> = with(expr) {
        internalizeExpr {
            z3Ctx.mkExtract(high, low, value.internalize() as BitVecExpr)
        }
    }

    override fun transform(expr: KBvSignExtensionExpr): KExpr<KBvSort> = with(expr) {
        internalizeExpr {
            z3Ctx.mkSignExt(i, value.internalize() as BitVecExpr)
        }
    }

    override fun transform(expr: KBvZeroExtensionExpr): KExpr<KBvSort> = with(expr) {
        internalizeExpr {
            z3Ctx.mkZeroExt(i, value.internalize() as BitVecExpr)
        }
    }

    override fun transform(expr: KBvRepeatExpr): KExpr<KBvSort> = with(expr) {
        internalizeExpr {
            z3Ctx.mkRepeat(expr.i, value.internalize() as BitVecExpr)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>): KExpr<T> = with(expr)
    { transform(expr.arg0, expr.arg1, z3Ctx::mkBVSHL) }

    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>): KExpr<T> = with(expr)
    { transform(expr.arg0, expr.arg1, z3Ctx::mkBVLSHR) }

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>): KExpr<T> = with(expr)
    { transform(expr.arg0, expr.arg1, z3Ctx::mkBVASHR) }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>): KExpr<T> = with(expr)
    { transform(expr.arg0, expr.arg1, z3Ctx::mkBVRotateLeft) }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>): KExpr<T> = with(expr)
    {
        internalizeExpr {
            z3Ctx.mkBVRotateLeft(i, value.internalize() as BitVecExpr)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>): KExpr<T> = with(expr)
    { transform(expr.arg0, expr.arg1, z3Ctx::mkBVRotateRight) }

    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>): KExpr<T> = with(expr)
    {
        internalizeExpr {
            z3Ctx.mkBVRotateRight(i, value.internalize() as BitVecExpr)
        }
    }

    override fun transform(expr: KBv2IntExpr): KExpr<KIntSort> = with(expr) {
        internalizeExpr {
            z3Ctx.mkBV2Int(expr.value.internalize() as BitVecExpr, expr.isSigned)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>): KExpr<KBoolSort> = with(expr) {
        internalizeExpr {
            z3Ctx.mkBVAddNoOverflow(arg0.internalize() as BitVecExpr, arg1.internalize() as BitVecExpr, isSigned)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>): KExpr<KBoolSort> =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVAddNoUnderflow) }

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>): KExpr<KBoolSort> =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVSubNoOverflow) }

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort> = with(expr) {
        internalizeExpr {
            z3Ctx.mkBVSubNoUnderflow(arg0.internalize() as BitVecExpr, arg1.internalize() as BitVecExpr, isSigned)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>): KExpr<KBoolSort> =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVSDivNoOverflow) }

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>): KExpr<KBoolSort> =
        with(expr) { transform(value, z3Ctx::mkBVNegNoOverflow) }

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>): KExpr<KBoolSort> = with(expr) {
        internalizeExpr {
            z3Ctx.mkBVMulNoOverflow(arg0.internalize() as BitVecExpr, arg1.internalize() as BitVecExpr, isSigned)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>): KExpr<KBoolSort> =
        with(expr) { transform(arg0, arg1, z3Ctx::mkBVMulNoUnderflow) }

    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>) = expr.internalizeExpr {
        z3Ctx.mkStore(array.internalize() as ArrayExpr, index.internalize(), value.internalize())
    }

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>) = expr.internalizeExpr {
        z3Ctx.mkSelect(array.internalize() as ArrayExpr, index.internalize())
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayConst<D, R>) = expr.internalizeExpr {
        z3Ctx.mkConstArray(expr.sort.internalize(), expr.value.internalize())
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>) = expr.internalizeExpr {
        val internalizedIndex = indexVarDecl.internalize()
        z3Ctx.mkLambda(arrayOf(internalizedIndex.range), arrayOf(internalizedIndex.name), body.internalize())
    }

    override fun <T : KArithSort<T>> transform(expr: KAddArithExpr<T>) = expr.internalizeExpr {
        z3Ctx.mkAdd(*args.map { it.internalize() as ArithExpr }.toTypedArray())
    }

    override fun <T : KArithSort<T>> transform(expr: KSubArithExpr<T>) = expr.internalizeExpr {
        z3Ctx.mkSub(*args.map { it.internalize() as ArithExpr }.toTypedArray())
    }

    override fun <T : KArithSort<T>> transform(expr: KMulArithExpr<T>) = expr.internalizeExpr {
        z3Ctx.mkMul(*args.map { it.internalize() as ArithExpr }.toTypedArray())
    }

    override fun <T : KArithSort<T>> transform(expr: KUnaryMinusArithExpr<T>) = expr.internalizeExpr {
        z3Ctx.mkUnaryMinus(arg.internalize() as ArithExpr)
    }

    override fun <T : KArithSort<T>> transform(expr: KDivArithExpr<T>) = expr.internalizeExpr {
        z3Ctx.mkDiv(lhs.internalize() as ArithExpr, rhs.internalize() as ArithExpr)
    }

    override fun <T : KArithSort<T>> transform(expr: KPowerArithExpr<T>) = expr.internalizeExpr {
        z3Ctx.mkPower(lhs.internalize() as ArithExpr, rhs.internalize() as ArithExpr)
    }

    override fun <T : KArithSort<T>> transform(expr: KLtArithExpr<T>) = expr.internalizeExpr {
        z3Ctx.mkLt(lhs.internalize() as ArithExpr, rhs.internalize() as ArithExpr)
    }

    override fun <T : KArithSort<T>> transform(expr: KLeArithExpr<T>) = expr.internalizeExpr {
        z3Ctx.mkLe(lhs.internalize() as ArithExpr, rhs.internalize() as ArithExpr)
    }

    override fun <T : KArithSort<T>> transform(expr: KGtArithExpr<T>) = expr.internalizeExpr {
        z3Ctx.mkGt(lhs.internalize() as ArithExpr, rhs.internalize() as ArithExpr)
    }

    override fun <T : KArithSort<T>> transform(expr: KGeArithExpr<T>) = expr.internalizeExpr {
        z3Ctx.mkGe(lhs.internalize() as ArithExpr, rhs.internalize() as ArithExpr)
    }

    override fun transform(expr: KModIntExpr) = expr.internalizeExpr {
        z3Ctx.mkMod(lhs.internalize() as IntExpr, rhs.internalize() as IntExpr)
    }

    override fun transform(expr: KRemIntExpr) = expr.internalizeExpr {
        z3Ctx.mkRem(lhs.internalize() as IntExpr, rhs.internalize() as IntExpr)
    }

    override fun transform(expr: KToRealIntExpr) = expr.internalizeExpr {
        z3Ctx.mkInt2Real(arg.internalize() as IntExpr)
    }

    override fun transform(expr: KInt32NumExpr) = expr.internalizeExpr {
        z3Ctx.mkInt(expr.value)
    }

    override fun transform(expr: KInt64NumExpr) = expr.internalizeExpr {
        z3Ctx.mkInt(expr.value)
    }

    override fun transform(expr: KIntBigNumExpr) = expr.internalizeExpr {
        z3Ctx.mkInt(expr.value.toString())
    }

    override fun transform(expr: KToIntRealExpr) = expr.internalizeExpr {
        z3Ctx.mkReal2Int(arg.internalize() as RealExpr)
    }

    override fun transform(expr: KIsIntRealExpr) = expr.internalizeExpr {
        z3Ctx.mkIsInteger(arg.internalize() as RealExpr)
    }

    override fun transform(expr: KRealNumExpr) = expr.internalizeExpr {
        val numerator = numerator.internalize()
        val denominator = denominator.internalize()
        z3Ctx.mkDiv(
            z3Ctx.mkInt2Real(numerator as IntExpr),
            z3Ctx.mkInt2Real(denominator as IntExpr)
        )
    }

    override fun transform(expr: KExistentialQuantifier) = expr.internalizeExpr {
        z3Ctx.mkExistsQuantifier(
            boundConstants = bounds.map { z3Ctx.mkConst(it.internalize()) }.toTypedArray(),
            body = body.internalize(),
            weight = 0,
            patterns = arrayOf(),
            noPatterns = arrayOf(),
            quantifierId = null,
            skolemId = null
        )
    }

    override fun transform(expr: KUniversalQuantifier) = expr.internalizeExpr {
        z3Ctx.mkForallQuantifier(
            boundConstants = bounds.map { z3Ctx.mkConst(it.internalize()) }.toTypedArray(),
            body = body.internalize(),
            weight = 0,
            patterns = arrayOf(),
            noPatterns = arrayOf(),
            quantifierId = null,
            skolemId = null
        )
    }

    @Suppress("MemberVisibilityCanBePrivate")
    inline fun <T : KExpr<*>> T.internalizeExpr(crossinline internalizer: T.() -> Expr): T {
        z3InternCtx.internalizeExpr(this) {
            internalizer()
        }
        return this
    }
}
