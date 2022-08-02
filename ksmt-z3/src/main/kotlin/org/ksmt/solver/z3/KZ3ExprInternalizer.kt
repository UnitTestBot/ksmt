package org.ksmt.solver.z3

import com.microsoft.z3.ArithExpr
import com.microsoft.z3.ArrayExpr
import com.microsoft.z3.BitVecExpr
import com.microsoft.z3.BoolExpr
import com.microsoft.z3.Context
import com.microsoft.z3.Expr
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
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KSort

open class KZ3ExprInternalizer(
    override val ctx: KContext,
    private val z3Ctx: Context,
    private val z3InternCtx: KZ3InternalizationContext,
    private val sortInternalizer: KZ3SortInternalizer,
    private val declInternalizer: KZ3DeclInternalizer
) : KTransformer {

    val exprStack = arrayListOf<KExpr<*>>()
    var internalizedExpr: Expr? = null

    fun <T : KSort> KExpr<T>.internalize(): Expr {
        exprStack.add(this)
        while (exprStack.isNotEmpty()) {
            internalizedExpr = null
            val expr = exprStack.removeLast()

            val internalized = z3InternCtx.findInternalizedExpr(expr)
            if (internalized != null) continue

            expr.accept(this@KZ3ExprInternalizer)

            if (internalizedExpr != null) {
                z3InternCtx.internalizeExpr(expr) { internalizedExpr!! }
            }
        }
        return z3InternCtx.findInternalizedExpr(this)
            ?: error("expr is not properly internalized: $this")
    }

    fun <T : KDecl<*>> T.internalizeDecl(): FuncDecl = accept(declInternalizer)

    fun <T : KSort> T.internalizeSort(): Sort = accept(sortInternalizer)

    override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> =
        error("Unexpected expr $expr")

    override fun <T : KSort> transform(expr: KFunctionApp<T>) = expr.transformList(expr.args) { args: Array<Expr> ->
        z3Ctx.mkApp(expr.decl.internalizeDecl(), *args)
    }

    override fun <T : KSort> transform(expr: KConst<T>) = expr.transform {
        z3Ctx.mkConst(expr.decl.internalizeDecl())
    }

    override fun transform(expr: KAndExpr) = expr.transformList(expr.args) { args: Array<BoolExpr> ->
        z3Ctx.mkAnd(*args)
    }

    override fun transform(expr: KOrExpr) = expr.transformList(expr.args) { args: Array<BoolExpr> ->
        z3Ctx.mkOr(*args)
    }

    override fun transform(expr: KNotExpr) = expr.transform(expr.arg, z3Ctx::mkNot)

    override fun transform(expr: KImpliesExpr) = expr.transform(expr.p, expr.q, z3Ctx::mkImplies)

    override fun transform(expr: KXorExpr) = expr.transform(expr.a, expr.b, z3Ctx::mkXor)

    override fun transform(expr: KTrue) = expr.transform { z3Ctx.mkTrue() }

    override fun transform(expr: KFalse) = expr.transform { z3Ctx.mkFalse() }

    override fun <T : KSort> transform(expr: KEqExpr<T>) = expr.transform(expr.lhs, expr.rhs, z3Ctx::mkEq)

    override fun <T : KSort> transform(expr: KDistinctExpr<T>) = expr.transformList(expr.args) { args: Array<Expr> ->
        z3Ctx.mkDistinct(*args)
    }

    override fun <T : KSort> transform(expr: KIteExpr<T>) =
        expr.transform(expr.condition, expr.trueBranch, expr.falseBranch, z3Ctx::mkITE)

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
            else -> error("Unknown bv expression class ${expr::class} in transformation method: ${expr.print()}")
        }
    }

    override fun transform(expr: KBitVec1Value) = transformBitVecValue(expr)

    override fun transform(expr: KBitVec8Value) = transformBitVecValue(expr)

    override fun transform(expr: KBitVec16Value) = transformBitVecValue(expr)

    override fun transform(expr: KBitVec32Value) = transformBitVecValue(expr)

    override fun transform(expr: KBitVec64Value) = transformBitVecValue(expr)

    override fun transform(expr: KBitVecCustomValue) = transformBitVecValue(expr)

    override fun <T : KBvSort> transform(expr: KBvNotExpr<T>) =
        expr.transform(expr.value, z3Ctx::mkBVNot)

    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>) =
        expr.transform(expr.value, z3Ctx::mkBVRedAND)

    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>) =
        expr.transform(expr.value, z3Ctx::mkBVRedOR)

    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVAND)

    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVOR)

    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVXOR)

    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVNAND)

    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVNOR)

    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVXNOR)

    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>) =
        expr.transform(expr.value, z3Ctx::mkBVNeg)

    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVAdd)

    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVSub)

    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVMul)

    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVUDiv)

    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVSDiv)

    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVURem)

    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVSRem)

    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVSMod)

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVULT)

    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVSLT)

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVULE)

    override fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVSLE)

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVUGE)

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVSGE)

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVUGT)

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVSGT)

    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkConcat)

    override fun transform(expr: KBvExtractExpr) = expr.transform(expr.value) { value: BitVecExpr ->
        z3Ctx.mkExtract(expr.high, expr.low, value)
    }

    override fun transform(expr: KBvSignExtensionExpr) = expr.transform(expr.value) { value: BitVecExpr ->
        z3Ctx.mkSignExt(expr.i, value)
    }

    override fun transform(expr: KBvZeroExtensionExpr) = expr.transform(expr.value) { value: BitVecExpr ->
        z3Ctx.mkZeroExt(expr.i, value)
    }

    override fun transform(expr: KBvRepeatExpr) = expr.transform(expr.value) { value: BitVecExpr ->
        z3Ctx.mkRepeat(expr.i, value)
    }

    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVSHL)

    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVLSHR)

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVASHR)

    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVRotateLeft)

    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>) =
        expr.transform(expr.value) { value: BitVecExpr ->
            z3Ctx.mkBVRotateLeft(expr.i, value)
        }

    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVRotateRight)

    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>) =
        expr.transform(expr.value) { value: BitVecExpr ->
            z3Ctx.mkBVRotateRight(expr.i, value)
        }

    override fun transform(expr: KBv2IntExpr) = expr.transform(expr.value) { value: BitVecExpr ->
        z3Ctx.mkBV2Int(value, expr.isSigned)
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>) =
        expr.transform(expr.arg0, expr.arg1) { a0: BitVecExpr, a1: BitVecExpr ->
            z3Ctx.mkBVAddNoOverflow(a0, a1, expr.isSigned)
        }

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVAddNoUnderflow)

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVSubNoOverflow)

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort> =
        expr.transform(expr.arg0, expr.arg1) { a0: BitVecExpr, a1: BitVecExpr ->
            z3Ctx.mkBVSubNoUnderflow(a0, a1, expr.isSigned)
        }

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVSDivNoOverflow)

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>) =
        expr.transform(expr.value, z3Ctx::mkBVNegNoOverflow)

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>) =
        expr.transform(expr.arg0, expr.arg1) { a0: BitVecExpr, a1: BitVecExpr ->
            z3Ctx.mkBVMulNoOverflow(a0, a1, expr.isSigned)
        }

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>) =
        expr.transform(expr.arg0, expr.arg1, z3Ctx::mkBVMulNoUnderflow)

    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>) =
        expr.transform<ArrayExpr, Expr, Expr, KArrayStore<D, R>>(expr.array, expr.index, expr.value, z3Ctx::mkStore)

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>) =
        expr.transform<ArrayExpr, Expr, KArraySelect<D, R>>(expr.array, expr.index, z3Ctx::mkSelect)

    override fun <D : KSort, R : KSort> transform(expr: KArrayConst<D, R>) = expr.transform(expr.value) { value: Expr ->
        z3Ctx.mkConstArray(expr.sort.internalizeSort(), value)
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>) = expr.transform(expr.body) { body: Expr ->
        val internalizedIndex = expr.indexVarDecl.internalizeDecl()
        z3Ctx.mkLambda(arrayOf(internalizedIndex.range), arrayOf(internalizedIndex.name), body)
    }

    override fun <T : KArithSort<T>> transform(expr: KAddArithExpr<T>) =
        expr.transformList(expr.args) { args: Array<ArithExpr> -> z3Ctx.mkAdd(*args) }

    override fun <T : KArithSort<T>> transform(expr: KSubArithExpr<T>) =
        expr.transformList(expr.args) { args: Array<ArithExpr> -> z3Ctx.mkSub(*args) }

    override fun <T : KArithSort<T>> transform(expr: KMulArithExpr<T>) =
        expr.transformList(expr.args) { args: Array<ArithExpr> -> z3Ctx.mkMul(*args) }

    override fun <T : KArithSort<T>> transform(expr: KUnaryMinusArithExpr<T>) =
        expr.transform(expr.arg, z3Ctx::mkUnaryMinus)

    override fun <T : KArithSort<T>> transform(expr: KDivArithExpr<T>) =
        expr.transform(expr.lhs, expr.rhs, z3Ctx::mkDiv)

    override fun <T : KArithSort<T>> transform(expr: KPowerArithExpr<T>) =
        expr.transform(expr.lhs, expr.rhs, z3Ctx::mkPower)

    override fun <T : KArithSort<T>> transform(expr: KLtArithExpr<T>) = expr.transform(expr.lhs, expr.rhs, z3Ctx::mkLt)

    override fun <T : KArithSort<T>> transform(expr: KLeArithExpr<T>) = expr.transform(expr.lhs, expr.rhs, z3Ctx::mkLe)

    override fun <T : KArithSort<T>> transform(expr: KGtArithExpr<T>) = expr.transform(expr.lhs, expr.rhs, z3Ctx::mkGt)

    override fun <T : KArithSort<T>> transform(expr: KGeArithExpr<T>) = expr.transform(expr.lhs, expr.rhs, z3Ctx::mkGe)

    override fun transform(expr: KModIntExpr) = expr.transform(expr.lhs, expr.rhs, z3Ctx::mkMod)

    override fun transform(expr: KRemIntExpr) = expr.transform(expr.lhs, expr.rhs, z3Ctx::mkRem)

    override fun transform(expr: KToRealIntExpr) = expr.transform(expr.arg, z3Ctx::mkInt2Real)

    override fun transform(expr: KInt32NumExpr) = expr.transform { z3Ctx.mkInt(expr.value) }

    override fun transform(expr: KInt64NumExpr) = expr.transform { z3Ctx.mkInt(expr.value) }

    override fun transform(expr: KIntBigNumExpr) = expr.transform { z3Ctx.mkInt(expr.value.toString()) }

    override fun transform(expr: KToIntRealExpr) = expr.transform(expr.arg, z3Ctx::mkReal2Int)

    override fun transform(expr: KIsIntRealExpr) = expr.transform(expr.arg, z3Ctx::mkIsInteger)

    override fun transform(expr: KRealNumExpr) =
        expr.transform(expr.numerator, expr.denominator) { numerator: IntExpr, denominator: IntExpr ->
            z3Ctx.mkDiv(z3Ctx.mkInt2Real(numerator), z3Ctx.mkInt2Real(denominator))
        }

    override fun transform(expr: KExistentialQuantifier) = expr.transform(expr.body) { body: BoolExpr ->
        z3Ctx.mkExistsQuantifier(
            boundConstants = expr.bounds.map { z3Ctx.mkConst(it.internalizeDecl()) }.toTypedArray(),
            body = body,
            weight = 0,
            patterns = arrayOf(),
            noPatterns = arrayOf(),
            quantifierId = null,
            skolemId = null
        )
    }

    override fun transform(expr: KUniversalQuantifier) = expr.transform(expr.body) { body: BoolExpr ->
        z3Ctx.mkForallQuantifier(
            boundConstants = expr.bounds.map { z3Ctx.mkConst(it.internalizeDecl()) }.toTypedArray(),
            body = body,
            weight = 0,
            patterns = arrayOf(),
            noPatterns = arrayOf(),
            quantifierId = null,
            skolemId = null
        )
    }

    fun internalizedExpr(expr: KExpr<*>): Expr? = z3InternCtx.findInternalizedExpr(expr)

    inline fun <S : KExpr<*>> S.transform(operation: () -> Expr): S = also {
        internalizedExpr = operation()
    }

    inline fun <reified A0 : Expr, S : KExpr<*>> S.transform(
        arg: KExpr<*>,
        operation: (A0) -> Expr
    ): S = also {
        val internalizedArg = internalizedExpr(arg)
        if (internalizedArg == null) {
            exprStack.add(this)
            exprStack.add(arg)
        } else {
            internalizedExpr = operation(internalizedArg as A0)
        }
    }

    inline fun <reified A0 : Expr, reified A1 : Expr, S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        operation: (A0, A1) -> Expr
    ): S = also {
        val internalizedArg0 = internalizedExpr(arg0)
        val internalizedArg1 = internalizedExpr(arg1)
        if (internalizedArg0 == null || internalizedArg1 == null) {
            exprStack.add(this)
            internalizedArg0 ?: exprStack.add(arg0)
            internalizedArg1 ?: exprStack.add(arg1)
        } else {
            internalizedExpr = operation(internalizedArg0 as A0, internalizedArg1 as A1)
        }
    }

    inline fun <reified A0 : Expr, reified A1 : Expr, reified A2 : Expr, S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        arg2: KExpr<*>,
        operation: (A0, A1, A2) -> Expr
    ): S = also {
        val internalizedArg0 = internalizedExpr(arg0)
        val internalizedArg1 = internalizedExpr(arg1)
        val internalizedArg2 = internalizedExpr(arg2)
        if (internalizedArg0 == null || internalizedArg1 == null || internalizedArg2 == null) {
            exprStack.add(this)
            internalizedArg0 ?: exprStack.add(arg0)
            internalizedArg1 ?: exprStack.add(arg1)
            internalizedArg2 ?: exprStack.add(arg2)
        } else {
            internalizedExpr = operation(internalizedArg0 as A0, internalizedArg1 as A1, internalizedArg2 as A2)
        }
    }

    inline fun <reified A : Expr, S : KExpr<*>> S.transformList(
        args: List<KExpr<*>>,
        operation: (Array<A>) -> Expr
    ): S = also {
        val internalizedArgs = mutableListOf<A>()
        var exprAdded = false
        var argsReady = true
        for (arg in args) {
            val internalized = internalizedExpr(arg)
            if (internalized != null) {
                internalizedArgs.add(internalized as A)
                continue
            }
            argsReady = false
            if (!exprAdded) {
                exprStack.add(this)
                exprAdded = true
            }
            exprStack.add(arg)
        }
        if (argsReady) {
            internalizedExpr = operation(internalizedArgs.toTypedArray())
        }
    }
}
