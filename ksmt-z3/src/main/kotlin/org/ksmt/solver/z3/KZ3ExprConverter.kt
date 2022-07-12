package org.ksmt.solver.z3

import com.microsoft.z3.ArraySort
import com.microsoft.z3.Expr
import com.microsoft.z3.FuncDecl
import com.microsoft.z3.IntNum
import com.microsoft.z3.RatNum
import com.microsoft.z3.Sort
import com.microsoft.z3.enumerations.Z3_ast_kind
import com.microsoft.z3.enumerations.Z3_decl_kind
import com.microsoft.z3.enumerations.Z3_sort_kind
import com.microsoft.z3.intOrNull
import com.microsoft.z3.longOrNull
import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.expr.KIntNumExpr
import org.ksmt.expr.KRealNumExpr
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

open class KZ3ExprConverter(
    private val ctx: KContext,
    private val z3InternCtx: KZ3InternalizationContext
) {
    @Suppress("UNCHECKED_CAST")
    fun <T : KSort> Expr.convert(): KExpr<T> = z3InternCtx.convertExpr(this) {
        convertExpr(this)
    } as? KExpr<T> ?: error("expr is not properly converted")

    @Suppress("UNCHECKED_CAST")
    fun <T : KSort> Sort.convert(): T = z3InternCtx.convertSort(this) {
        convertSort(this)
    } as? T ?: error("sort is not properly converted")

    @Suppress("UNCHECKED_CAST")
    fun <T : KSort> FuncDecl.convert(): KDecl<T> = z3InternCtx.convertDecl(this) {
        convertDecl(this)
    } as? KDecl<T> ?: error("decl is not properly converted")

    open fun convertDecl(decl: FuncDecl): KDecl<*> = with(ctx) {
        val sort = convertSort(decl.range)
        val args = decl.domain.map { convertSort(it) }
        return mkFuncDecl("${decl.name}", sort, args)
    }

    open fun convertSort(sort: Sort): KSort = with(ctx) {
        when (sort.sortKind) {
            Z3_sort_kind.Z3_BOOL_SORT -> boolSort
            Z3_sort_kind.Z3_INT_SORT -> intSort
            Z3_sort_kind.Z3_REAL_SORT -> realSort
            Z3_sort_kind.Z3_ARRAY_SORT -> (sort as ArraySort).let {
                mkArraySort(convertSort(it.domain), convertSort(it.range))
            }
            Z3_sort_kind.Z3_BV_SORT -> TODO("bit-vec are not supported yet")
            Z3_sort_kind.Z3_UNINTERPRETED_SORT,
            Z3_sort_kind.Z3_DATATYPE_SORT,
            Z3_sort_kind.Z3_RELATION_SORT,
            Z3_sort_kind.Z3_FINITE_DOMAIN_SORT,
            Z3_sort_kind.Z3_FLOATING_POINT_SORT,
            Z3_sort_kind.Z3_ROUNDING_MODE_SORT,
            Z3_sort_kind.Z3_SEQ_SORT,
            Z3_sort_kind.Z3_RE_SORT,
            Z3_sort_kind.Z3_UNKNOWN_SORT -> TODO("$sort is not supported yet")
            null -> error("z3 sort kind cannot be null")
        }
    }

    open fun convertExpr(expr: Expr): KExpr<*> = when (expr.astKind) {
        Z3_ast_kind.Z3_NUMERAL_AST -> convertNumeral(expr)
        Z3_ast_kind.Z3_APP_AST -> convertApp(expr)
        Z3_ast_kind.Z3_QUANTIFIER_AST -> TODO("quantifier conversion is not implemented")
        Z3_ast_kind.Z3_VAR_AST -> TODO("var conversion is not implemented")
        Z3_ast_kind.Z3_SORT_AST,
        Z3_ast_kind.Z3_FUNC_DECL_AST,
        Z3_ast_kind.Z3_UNKNOWN_AST -> error("impossible ast kind for expressions")
        null -> error("z3 ast kind cannot be null")
    }

    @Suppress(
        "RemoveExplicitTypeArguments",
        "USELESS_CAST",
        "UPPER_BOUND_VIOLATED_WARNING",
        "LongMethod",
        "ComplexMethod"
    )
    open fun convertApp(expr: Expr): KExpr<*> = with(ctx) {
        when (expr.funcDecl.declKind) {
            Z3_decl_kind.Z3_OP_TRUE -> trueExpr
            Z3_decl_kind.Z3_OP_FALSE -> falseExpr
            Z3_decl_kind.Z3_OP_UNINTERPRETED -> mkApp(convertDecl(expr.funcDecl), expr.args.map { it.convert<KSort>() })
            Z3_decl_kind.Z3_OP_AND -> mkAnd(expr.args.map { it.convert() })
            Z3_decl_kind.Z3_OP_OR -> mkOr(expr.args.map { it.convert() })
            Z3_decl_kind.Z3_OP_NOT -> mkNot(expr.args[0].convert())
            Z3_decl_kind.Z3_OP_EQ -> mkEq(expr.args[0].convert(), expr.args[1].convert())
            Z3_decl_kind.Z3_OP_ITE -> mkIte(
                expr.args[0].convert(),
                expr.args[1].convert(),
                expr.args[2].convert()
            )
            Z3_decl_kind.Z3_OP_LE -> mkArithLe<KArithSort<*>>(
                expr.args[0].convert<KArithSort<*>>(),
                expr.args[1].convert<KArithSort<*>>()
            )
            Z3_decl_kind.Z3_OP_GE -> mkArithGe<KArithSort<*>>(
                expr.args[0].convert<KArithSort<*>>(),
                expr.args[1].convert<KArithSort<*>>()
            )
            Z3_decl_kind.Z3_OP_LT -> mkArithLt<KArithSort<*>>(
                expr.args[0].convert<KArithSort<*>>(),
                expr.args[1].convert<KArithSort<*>>()
            )
            Z3_decl_kind.Z3_OP_GT -> mkArithGt<KArithSort<*>>(
                expr.args[0].convert<KArithSort<*>>(),
                expr.args[1].convert<KArithSort<*>>()
            )
            Z3_decl_kind.Z3_OP_ADD -> mkArithAdd<KArithSort<*>>(
                expr.args.map { it.convert<KArithSort<*>>() } as List<KExpr<KArithSort<*>>>
            )
            Z3_decl_kind.Z3_OP_SUB -> mkArithSub<KArithSort<*>>(
                expr.args.map { it.convert<KArithSort<*>>() } as List<KExpr<KArithSort<*>>>
            )
            Z3_decl_kind.Z3_OP_MUL -> mkArithMul<KArithSort<*>>(
                expr.args.map { it.convert<KArithSort<*>>() } as List<KExpr<KArithSort<*>>>
            )
            Z3_decl_kind.Z3_OP_UMINUS -> mkArithUnaryMinus<KArithSort<*>>(expr.args[0].convert<KArithSort<*>>())
            Z3_decl_kind.Z3_OP_DIV -> mkArithDiv<KArithSort<*>>(
                expr.args[0].convert<KArithSort<*>>(),
                expr.args[1].convert<KArithSort<*>>()
            )
            Z3_decl_kind.Z3_OP_POWER -> mkArithPower<KArithSort<*>>(
                expr.args[0].convert<KArithSort<*>>(),
                expr.args[1].convert<KArithSort<*>>()
            )
            Z3_decl_kind.Z3_OP_REM -> mkIntRem(expr.args[0].convert(), expr.args[1].convert())
            Z3_decl_kind.Z3_OP_MOD -> mkIntMod(expr.args[0].convert(), expr.args[1].convert())
            Z3_decl_kind.Z3_OP_TO_REAL -> mkIntToReal(expr.args[0].convert())
            Z3_decl_kind.Z3_OP_TO_INT -> mkRealToInt(expr.args[0].convert())
            Z3_decl_kind.Z3_OP_IS_INT -> mkRealIsInt(expr.args[0].convert())
            Z3_decl_kind.Z3_OP_STORE -> mkArrayStore(
                expr.args[0].convert(),
                expr.args[1].convert(),
                expr.args[2].convert()
            )
            Z3_decl_kind.Z3_OP_SELECT -> mkArraySelect(
                expr.args[0].convert(),
                expr.args[1].convert()
            )
            Z3_decl_kind.Z3_OP_CONST_ARRAY -> mkArrayConst(
                convertSort(expr.funcDecl.range) as KArraySort<*, *>,
                expr.args[0].convert()
            )
            else -> TODO("${expr.funcDecl} is not supported")
        }
    }

    open fun convertNumeral(expr: Expr): KExpr<*> = when (expr.sort.sortKind) {
        Z3_sort_kind.Z3_INT_SORT -> convertNumeral(expr as IntNum)
        Z3_sort_kind.Z3_REAL_SORT -> convertNumeral(expr as RatNum)
        Z3_sort_kind.Z3_BV_SORT -> TODO("bit-vec are not supported yet")
        else -> TODO("numerals with ${expr.sort} are not supported")
    }

    @Suppress("MemberVisibilityCanBePrivate")
    fun convertNumeral(expr: IntNum): KIntNumExpr = with(ctx) {
        expr.intOrNull()?.let { mkIntNum(it) }
            ?: expr.longOrNull()?.let { mkIntNum(it) }
            ?: mkIntNum(expr.bigInteger)
    }

    @Suppress("MemberVisibilityCanBePrivate")
    fun convertNumeral(expr: RatNum): KRealNumExpr = with(ctx) {
        mkRealNum(convertNumeral(expr.numerator), convertNumeral(expr.denominator))
    }
}
