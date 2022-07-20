package org.ksmt.solver.z3

import com.microsoft.z3.ArithExpr
import com.microsoft.z3.ArrayExpr
import com.microsoft.z3.BoolExpr
import com.microsoft.z3.Context
import com.microsoft.z3.Expr
import com.microsoft.z3.FuncDecl
import com.microsoft.z3.IntExpr
import com.microsoft.z3.RealExpr
import com.microsoft.z3.Sort
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
import org.ksmt.expr.KBitVec16Expr
import org.ksmt.expr.KBitVec32Expr
import org.ksmt.expr.KBitVec64Expr
import org.ksmt.expr.KBitVec8Expr
import org.ksmt.expr.KBitVecCustomExpr
import org.ksmt.expr.KBitVecExpr
import org.ksmt.expr.KBitVecNumberExpr
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
import org.ksmt.expr.KTransformer
import org.ksmt.expr.KTrue
import org.ksmt.expr.KUnaryMinusArithExpr
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.KXorExpr
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KBVSort
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

    override fun <T : KBVSort> transformBitVecExpr(expr: KBitVecExpr<T>): KExpr<T> = expr.internalizeExpr {
        val sizeBits = expr.sort().sizeBits.toInt()
        when (expr) {
            is KBitVec8Expr, is KBitVec16Expr, is KBitVec32Expr -> {
                z3Ctx.mkBV((expr as KBitVecNumberExpr<*, *>).numberValue.toInt(), sizeBits)
            }
            is KBitVec64Expr -> z3Ctx.mkBV(expr.numberValue, sizeBits)
            is KBitVecCustomExpr -> z3Ctx.mkBV(expr.value, sizeBits)
            else -> error("Unknown bv expression class ${expr::class} in transformation method: ${expr.print()}")
        }
    }

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
