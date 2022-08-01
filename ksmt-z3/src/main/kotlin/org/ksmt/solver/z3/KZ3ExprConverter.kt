package org.ksmt.solver.z3

import com.microsoft.z3.ArraySort
import com.microsoft.z3.BitVecNum
import com.microsoft.z3.BitVecSort
import com.microsoft.z3.Expr
import com.microsoft.z3.FuncDecl
import com.microsoft.z3.IntNum
import com.microsoft.z3.Quantifier
import com.microsoft.z3.RatNum
import com.microsoft.z3.Sort
import com.microsoft.z3.ctx
import com.microsoft.z3.enumerations.Z3_ast_kind
import com.microsoft.z3.enumerations.Z3_decl_kind
import com.microsoft.z3.enumerations.Z3_sort_kind
import com.microsoft.z3.intOrNull
import com.microsoft.z3.longOrNull
import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KExpr
import org.ksmt.expr.KIntNumExpr
import org.ksmt.expr.KRealNumExpr
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
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
            Z3_sort_kind.Z3_BV_SORT -> mkBvSort((sort as BitVecSort).size.toUInt())
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
        Z3_ast_kind.Z3_QUANTIFIER_AST -> convertQuantifier(expr)
        Z3_ast_kind.Z3_VAR_AST -> error("var conversion is not supported")
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
            Z3_decl_kind.Z3_OP_AND -> convertList(expr.args, ::mkAnd)
            Z3_decl_kind.Z3_OP_OR -> convertList(expr.args, ::mkOr)
            Z3_decl_kind.Z3_OP_XOR -> convert(expr.args, ::mkXor)
            Z3_decl_kind.Z3_OP_NOT -> convert(expr.args, ::mkNot)
            Z3_decl_kind.Z3_OP_IMPLIES -> convert(expr.args, ::mkImplies)
            Z3_decl_kind.Z3_OP_EQ -> convert(expr.args, ::mkEq)
            Z3_decl_kind.Z3_OP_DISTINCT -> convertList(expr.args, ::mkDistinct)
            Z3_decl_kind.Z3_OP_ITE -> convert(expr.args, ::mkIte)
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
            Z3_decl_kind.Z3_OP_REM -> convert(expr.args, ::mkIntRem)
            Z3_decl_kind.Z3_OP_MOD -> convert(expr.args, ::mkIntMod)
            Z3_decl_kind.Z3_OP_TO_REAL -> convert(expr.args, ::mkIntToReal)
            Z3_decl_kind.Z3_OP_TO_INT -> convert(expr.args, ::mkRealToInt)
            Z3_decl_kind.Z3_OP_IS_INT -> convert(expr.args, ::mkRealIsInt)
            Z3_decl_kind.Z3_OP_STORE -> convert(expr.args, ::mkArrayStore)
            Z3_decl_kind.Z3_OP_SELECT -> convert(expr.args, ::mkArraySelect)
            Z3_decl_kind.Z3_OP_CONST_ARRAY -> mkArrayConst(
                convertSort(expr.funcDecl.range) as KArraySort<*, *>,
                expr.args[0].convert()
            )
            Z3_decl_kind.Z3_OP_BNUM,
            Z3_decl_kind.Z3_OP_BIT1,
            Z3_decl_kind.Z3_OP_BIT0 -> error("unexpected Bv numeral in app converter: $expr")
            Z3_decl_kind.Z3_OP_BNEG -> convert(expr.args, ::mkBvNegationExpr)
            Z3_decl_kind.Z3_OP_BADD -> convertReduced(expr.args, ::mkBvAddExpr)
            Z3_decl_kind.Z3_OP_BSUB -> convertReduced(expr.args, ::mkBvSubExpr)
            Z3_decl_kind.Z3_OP_BMUL -> convertReduced(expr.args, ::mkBvMulExpr)
            Z3_decl_kind.Z3_OP_BSDIV, Z3_decl_kind.Z3_OP_BSDIV_I -> convert(expr.args, ::mkBvSignedDivExpr)
            Z3_decl_kind.Z3_OP_BUDIV, Z3_decl_kind.Z3_OP_BUDIV_I -> convert(expr.args, ::mkBvUnsignedDivExpr)
            Z3_decl_kind.Z3_OP_BSREM, Z3_decl_kind.Z3_OP_BSREM_I -> convert(expr.args, ::mkBvSignedRemExpr)
            Z3_decl_kind.Z3_OP_BUREM, Z3_decl_kind.Z3_OP_BUREM_I -> convert(expr.args, ::mkBvUnsignedRemExpr)
            Z3_decl_kind.Z3_OP_BSMOD, Z3_decl_kind.Z3_OP_BSMOD_I -> convert(expr.args, ::mkBvSignedModExpr)
            Z3_decl_kind.Z3_OP_BSDIV0,
            Z3_decl_kind.Z3_OP_BUDIV0,
            Z3_decl_kind.Z3_OP_BSREM0,
            Z3_decl_kind.Z3_OP_BUREM0,
            Z3_decl_kind.Z3_OP_BSMOD0 -> error("unexpected Bv internal function app: $expr")
            Z3_decl_kind.Z3_OP_ULEQ -> convert(expr.args, ::mkBvUnsignedLessOrEqualExpr)
            Z3_decl_kind.Z3_OP_SLEQ -> convert(expr.args, ::mkBvSignedLessOrEqualExpr)
            Z3_decl_kind.Z3_OP_UGEQ -> convert(expr.args, ::mkBvUnsignedGreaterOrEqualExpr)
            Z3_decl_kind.Z3_OP_SGEQ -> convert(expr.args, ::mkBvSignedGreaterOrEqualExpr)
            Z3_decl_kind.Z3_OP_ULT -> convert(expr.args, ::mkBvUnsignedLessExpr)
            Z3_decl_kind.Z3_OP_SLT -> convert(expr.args, ::mkBvSignedLessExpr)
            Z3_decl_kind.Z3_OP_UGT -> convert(expr.args, ::mkBvUnsignedGreaterExpr)
            Z3_decl_kind.Z3_OP_SGT -> convert(expr.args, ::mkBvSignedGreaterExpr)
            Z3_decl_kind.Z3_OP_BAND -> convert(expr.args, ::mkBvAndExpr)
            Z3_decl_kind.Z3_OP_BOR -> convert(expr.args, ::mkBvOrExpr)
            Z3_decl_kind.Z3_OP_BNOT -> convert(expr.args, ::mkBvNotExpr)
            Z3_decl_kind.Z3_OP_BXOR -> convert(expr.args, ::mkBvXorExpr)
            Z3_decl_kind.Z3_OP_BNAND -> convert(expr.args, ::mkBvNAndExpr)
            Z3_decl_kind.Z3_OP_BNOR -> convert(expr.args, ::mkBvNorExpr)
            Z3_decl_kind.Z3_OP_BXNOR -> convert(expr.args, ::mkBvXNorExpr)
            Z3_decl_kind.Z3_OP_CONCAT -> convertReduced(expr.args, ::mkBvConcatExpr)
            Z3_decl_kind.Z3_OP_SIGN_EXT -> {
                val size = expr.funcDecl.parameters[0].int
                mkBvSignExtensionExpr(size, expr.args[0].convert())
            }
            Z3_decl_kind.Z3_OP_ZERO_EXT -> {
                val size = expr.funcDecl.parameters[0].int
                mkBvZeroExtensionExpr(size, expr.args[0].convert())
            }
            Z3_decl_kind.Z3_OP_EXTRACT -> {
                val high = expr.funcDecl.parameters[0].int
                val low = expr.funcDecl.parameters[1].int
                mkBvExtractExpr(high, low, expr.args[0].convert())
            }
            Z3_decl_kind.Z3_OP_REPEAT -> {
                val repeatCount = expr.funcDecl.parameters[0].int
                mkBvRepeatExpr(repeatCount, expr.args[0].convert())
            }
            Z3_decl_kind.Z3_OP_BREDOR -> convert(expr.args, ::mkBvReductionOrExpr)
            Z3_decl_kind.Z3_OP_BREDAND -> convert(expr.args, ::mkBvReductionAndExpr)
            Z3_decl_kind.Z3_OP_BCOMP -> TODO("bcomp conversion is not supported")
            Z3_decl_kind.Z3_OP_BSHL -> convert(expr.args, ::mkBvShiftLeftExpr)
            Z3_decl_kind.Z3_OP_BLSHR -> convert(expr.args, ::mkBvLogicalShiftRightExpr)
            Z3_decl_kind.Z3_OP_BASHR -> convert(expr.args, ::mkBvArithShiftRightExpr)
            Z3_decl_kind.Z3_OP_ROTATE_LEFT -> {
                val rotation = expr.funcDecl.parameters[0].int
                mkBvRotateLeftExpr(rotation, expr.args[0].convert())
            }
            Z3_decl_kind.Z3_OP_ROTATE_RIGHT -> {
                val rotation = expr.funcDecl.parameters[0].int
                mkBvRotateRightExpr(rotation, expr.args[0].convert())
            }
            Z3_decl_kind.Z3_OP_EXT_ROTATE_LEFT -> convert(expr.args, ::mkBvRotateLeftExpr)
            Z3_decl_kind.Z3_OP_EXT_ROTATE_RIGHT -> convert(expr.args, ::mkBvRotateRightExpr)
            Z3_decl_kind.Z3_OP_BIT2BOOL -> TODO("bit2bool conversion is not supported")
            Z3_decl_kind.Z3_OP_INT2BV -> TODO("int2bv conversion is not supported")
            Z3_decl_kind.Z3_OP_BV2INT -> TODO("bv2int conversion is not supported")
            Z3_decl_kind.Z3_OP_CARRY -> {
                mkBvOrExpr(
                    mkBvAndExpr(expr.args[0].convert(), expr.args[1].convert()),
                    mkBvOrExpr(
                        mkBvAndExpr(expr.args[0].convert(), expr.args[2].convert()),
                        mkBvAndExpr(expr.args[1].convert(), expr.args[2].convert())
                    )
                )
            }
            Z3_decl_kind.Z3_OP_XOR3 -> convertReduced(expr.args, ::mkBvXorExpr)
            Z3_decl_kind.Z3_OP_BSMUL_NO_OVFL -> mkBvMulNoOverflowExpr(
                expr.args[0].convert(),
                expr.args[1].convert(),
                isSigned = true
            )
            Z3_decl_kind.Z3_OP_BUMUL_NO_OVFL -> mkBvMulNoOverflowExpr(
                expr.args[0].convert(),
                expr.args[1].convert(),
                isSigned = false
            )
            Z3_decl_kind.Z3_OP_BSMUL_NO_UDFL -> convert(expr.args, ::mkBvMulNoUnderflowExpr)
            else -> TODO("${expr.funcDecl} is not supported")
        }
    }

    inline fun <T : KSort, A0 : KSort> convert(
        args: Array<out Expr>,
        op: (KExpr<A0>) -> KExpr<T>
    ): KExpr<T> {
        check(args.size == 1) { "arguments size mismatch: expected 1, actual ${args.size}" }
        return op(args[0].convert())
    }

    inline fun <T : KSort, A0 : KSort, A1 : KSort> convert(
        args: Array<out Expr>,
        op: (KExpr<A0>, KExpr<A1>) -> KExpr<T>
    ): KExpr<T> {
        check(args.size == 2) { "arguments size mismatch: expected 2, actual ${args.size}" }
        return op(args[0].convert(), args[1].convert())
    }

    inline fun <T : KSort, A0 : KSort, A1 : KSort, A2 : KSort> convert(
        args: Array<out Expr>,
        op: (KExpr<A0>, KExpr<A1>, KExpr<A2>) -> KExpr<T>
    ): KExpr<T> {
        check(args.size == 3) { "arguments size mismatch: expected 3, actual ${args.size}" }
        return op(args[0].convert(), args[1].convert(), args[2].convert())
    }

    inline fun <T : KSort, A : KSort> convertList(
        args: Array<out Expr>,
        op: (List<KExpr<A>>) -> KExpr<T>
    ): KExpr<T> = op(args.map { it.convert() })

    inline fun <T : KSort> convertReduced(
        args: Array<out Expr>,
        op: (KExpr<T>, KExpr<T>) -> KExpr<T>
    ): KExpr<T> = args.map { it.convert<T>() }.reduce(op)

    open fun convertNumeral(expr: Expr): KExpr<*> = when (expr.sort.sortKind) {
        Z3_sort_kind.Z3_INT_SORT -> convertNumeral(expr as IntNum)
        Z3_sort_kind.Z3_REAL_SORT -> convertNumeral(expr as RatNum)
        Z3_sort_kind.Z3_BV_SORT -> convertNumeral(expr as BitVecNum)
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

    @Suppress("MemberVisibilityCanBePrivate")
    fun convertNumeral(expr: BitVecNum): KBitVecValue<*> = with(ctx) {
        val sizeBits = expr.sortSize.toUInt()
        mkBv(value = expr.toBinaryString().padStart(sizeBits.toInt(), '0'), sizeBits)
    }

    open fun convertQuantifier(expr: Expr): KExpr<KBoolSort> = with(ctx) {
        expr as Quantifier

        val z3Bounds = expr.boundVariableSorts.zip(expr.boundVariableNames).map { (sort, name) ->
            expr.ctx.mkConst(name, sort)
        }.reversed()

        val body = expr.body.substituteVars(z3Bounds.toTypedArray()).convert<KBoolSort>()

        val bounds = z3Bounds.map { it.funcDecl.convert<KSort>() }

        if (expr.isUniversal) {
            mkUniversalQuantifier(body, bounds)
        } else {
            mkExistentialQuantifier(body, bounds)
        }
    }
}
