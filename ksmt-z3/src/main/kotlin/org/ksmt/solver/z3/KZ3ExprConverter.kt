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
import com.microsoft.z3.isLambda
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
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KSort

open class KZ3ExprConverter(
    private val ctx: KContext,
    private val z3InternCtx: KZ3InternalizationContext
) {

    val exprStack = arrayListOf<Expr<*>>()

    @Suppress("UNCHECKED_CAST")
    fun <T : KSort> Expr<*>.convert(): KExpr<T> {
        exprStack.add(this)
        while (exprStack.isNotEmpty()) {
            val expr = exprStack.removeLast()

            if (z3InternCtx.findConvertedExpr(expr) != null) continue

            val converted = convertExpr(expr)

            if (converted != null) {
                z3InternCtx.convertExpr(expr) { converted }
            }
        }
        return z3InternCtx.findConvertedExpr(this) as? KExpr<T>
            ?: error("expr is not properly converted")
    }

    @Suppress("UNCHECKED_CAST")
    fun <T : KSort> Sort.convert(): T = z3InternCtx.convertSort(this) {
        convertSort(this)
    } as? T ?: error("sort is not properly converted")

    @Suppress("UNCHECKED_CAST")
    fun <T : KSort> FuncDecl<*>.convert(): KDecl<T> = z3InternCtx.convertDecl(this) {
        convertDecl(this)
    } as? KDecl<T> ?: error("decl is not properly converted")

    open fun convertDecl(decl: FuncDecl<*>): KDecl<*> = with(ctx) {
        val sort = convertSort(decl.range)
        val args = decl.domain.map { convertSort(it) }
        return mkFuncDecl("${decl.name}", sort, args)
    }

    open fun convertSort(sort: Sort): KSort = with(ctx) {
        when (sort.sortKind) {
            Z3_sort_kind.Z3_BOOL_SORT -> boolSort
            Z3_sort_kind.Z3_INT_SORT -> intSort
            Z3_sort_kind.Z3_REAL_SORT -> realSort
            Z3_sort_kind.Z3_ARRAY_SORT -> (sort as ArraySort<*, *>).let {
                mkArraySort(convertSort(it.domain), convertSort(it.range))
            }
            Z3_sort_kind.Z3_BV_SORT -> mkBvSort((sort as BitVecSort).size.toUInt())
            Z3_sort_kind.Z3_UNINTERPRETED_SORT -> mkUninterpretedSort(sort.name.toString())
            Z3_sort_kind.Z3_DATATYPE_SORT,
            Z3_sort_kind.Z3_RELATION_SORT,
            Z3_sort_kind.Z3_FINITE_DOMAIN_SORT,
            Z3_sort_kind.Z3_FLOATING_POINT_SORT,
            Z3_sort_kind.Z3_ROUNDING_MODE_SORT,
            Z3_sort_kind.Z3_SEQ_SORT,
            Z3_sort_kind.Z3_RE_SORT,
            Z3_sort_kind.Z3_CHAR_SORT,
            Z3_sort_kind.Z3_UNKNOWN_SORT -> TODO("$sort is not supported yet")
            null -> error("z3 sort kind cannot be null")
        }
    }

    open fun convertExpr(expr: Expr<*>): KExpr<*>? = when (expr.astKind) {
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
    open fun convertApp(expr: Expr<*>): KExpr<*>? = with(ctx) {
        when (expr.funcDecl.declKind) {
            Z3_decl_kind.Z3_OP_TRUE -> trueExpr
            Z3_decl_kind.Z3_OP_FALSE -> falseExpr
            Z3_decl_kind.Z3_OP_UNINTERPRETED -> expr.convertList { args: List<KExpr<KSort>> ->
                mkApp(convertDecl(expr.funcDecl), args)
            }
            Z3_decl_kind.Z3_OP_AND -> expr.convertList(::mkAnd)
            Z3_decl_kind.Z3_OP_OR -> expr.convertList(::mkOr)
            Z3_decl_kind.Z3_OP_XOR -> expr.convert(::mkXor)
            Z3_decl_kind.Z3_OP_NOT -> expr.convert(::mkNot)
            Z3_decl_kind.Z3_OP_IMPLIES -> expr.convert(::mkImplies)
            Z3_decl_kind.Z3_OP_EQ -> expr.convert(::mkEq)
            Z3_decl_kind.Z3_OP_DISTINCT -> expr.convertList(::mkDistinct)
            Z3_decl_kind.Z3_OP_ITE -> expr.convert(::mkIte)
            Z3_decl_kind.Z3_OP_LE -> expr.convert<KBoolSort, KArithSort<*>, KArithSort<*>>(::mkArithLe)
            Z3_decl_kind.Z3_OP_GE -> expr.convert<KBoolSort, KArithSort<*>, KArithSort<*>>(::mkArithGe)
            Z3_decl_kind.Z3_OP_LT -> expr.convert<KBoolSort, KArithSort<*>, KArithSort<*>>(::mkArithLt)
            Z3_decl_kind.Z3_OP_GT -> expr.convert<KBoolSort, KArithSort<*>, KArithSort<*>>(::mkArithGt)
            Z3_decl_kind.Z3_OP_ADD -> expr.convertList<KArithSort<*>, KArithSort<*>>(::mkArithAdd)
            Z3_decl_kind.Z3_OP_SUB -> expr.convertList<KArithSort<*>, KArithSort<*>>(::mkArithSub)
            Z3_decl_kind.Z3_OP_MUL -> expr.convertList<KArithSort<*>, KArithSort<*>>(::mkArithMul)
            Z3_decl_kind.Z3_OP_UMINUS -> expr.convert<KArithSort<*>, KArithSort<*>>(::mkArithUnaryMinus)
            Z3_decl_kind.Z3_OP_DIV -> expr.convert<KArithSort<*>, KArithSort<*>, KArithSort<*>>(::mkArithDiv)
            Z3_decl_kind.Z3_OP_POWER -> expr.convert<KArithSort<*>, KArithSort<*>, KArithSort<*>>(::mkArithPower)
            Z3_decl_kind.Z3_OP_REM -> expr.convert(::mkIntRem)
            Z3_decl_kind.Z3_OP_MOD -> expr.convert(::mkIntMod)
            Z3_decl_kind.Z3_OP_TO_REAL -> expr.convert(::mkIntToReal)
            Z3_decl_kind.Z3_OP_TO_INT -> expr.convert(::mkRealToInt)
            Z3_decl_kind.Z3_OP_IS_INT -> expr.convert(::mkRealIsInt)
            Z3_decl_kind.Z3_OP_STORE -> expr.convert(::mkArrayStore)
            Z3_decl_kind.Z3_OP_SELECT -> expr.convert(::mkArraySelect)
            Z3_decl_kind.Z3_OP_CONST_ARRAY -> expr.convert { arg: KExpr<KSort> ->
                mkArrayConst(convertSort(expr.funcDecl.range) as KArraySort<*, *>, arg)
            }
            Z3_decl_kind.Z3_OP_BNUM,
            Z3_decl_kind.Z3_OP_BIT1,
            Z3_decl_kind.Z3_OP_BIT0 -> error("unexpected Bv numeral in app converter: $expr")
            Z3_decl_kind.Z3_OP_BNEG -> expr.convert(::mkBvNegationExpr)
            Z3_decl_kind.Z3_OP_BADD -> expr.convertReduced(::mkBvAddExpr)
            Z3_decl_kind.Z3_OP_BSUB -> expr.convertReduced(::mkBvSubExpr)
            Z3_decl_kind.Z3_OP_BMUL -> expr.convertReduced(::mkBvMulExpr)
            Z3_decl_kind.Z3_OP_BSDIV, Z3_decl_kind.Z3_OP_BSDIV_I -> expr.convert(::mkBvSignedDivExpr)
            Z3_decl_kind.Z3_OP_BUDIV, Z3_decl_kind.Z3_OP_BUDIV_I -> expr.convert(::mkBvUnsignedDivExpr)
            Z3_decl_kind.Z3_OP_BSREM, Z3_decl_kind.Z3_OP_BSREM_I -> expr.convert(::mkBvSignedRemExpr)
            Z3_decl_kind.Z3_OP_BUREM, Z3_decl_kind.Z3_OP_BUREM_I -> expr.convert(::mkBvUnsignedRemExpr)
            Z3_decl_kind.Z3_OP_BSMOD, Z3_decl_kind.Z3_OP_BSMOD_I -> expr.convert(::mkBvSignedModExpr)
            Z3_decl_kind.Z3_OP_BSDIV0,
            Z3_decl_kind.Z3_OP_BUDIV0,
            Z3_decl_kind.Z3_OP_BSREM0,
            Z3_decl_kind.Z3_OP_BUREM0,
            Z3_decl_kind.Z3_OP_BSMOD0 -> error("unexpected Bv internal function app: $expr")
            Z3_decl_kind.Z3_OP_ULEQ -> expr.convert(::mkBvUnsignedLessOrEqualExpr)
            Z3_decl_kind.Z3_OP_SLEQ -> expr.convert(::mkBvSignedLessOrEqualExpr)
            Z3_decl_kind.Z3_OP_UGEQ -> expr.convert(::mkBvUnsignedGreaterOrEqualExpr)
            Z3_decl_kind.Z3_OP_SGEQ -> expr.convert(::mkBvSignedGreaterOrEqualExpr)
            Z3_decl_kind.Z3_OP_ULT -> expr.convert(::mkBvUnsignedLessExpr)
            Z3_decl_kind.Z3_OP_SLT -> expr.convert(::mkBvSignedLessExpr)
            Z3_decl_kind.Z3_OP_UGT -> expr.convert(::mkBvUnsignedGreaterExpr)
            Z3_decl_kind.Z3_OP_SGT -> expr.convert(::mkBvSignedGreaterExpr)
            Z3_decl_kind.Z3_OP_BAND -> expr.convert(::mkBvAndExpr)
            Z3_decl_kind.Z3_OP_BOR -> expr.convert(::mkBvOrExpr)
            Z3_decl_kind.Z3_OP_BNOT -> expr.convert(::mkBvNotExpr)
            Z3_decl_kind.Z3_OP_BXOR -> expr.convert(::mkBvXorExpr)
            Z3_decl_kind.Z3_OP_BNAND -> expr.convert(::mkBvNAndExpr)
            Z3_decl_kind.Z3_OP_BNOR -> expr.convert(::mkBvNorExpr)
            Z3_decl_kind.Z3_OP_BXNOR -> expr.convert(::mkBvXNorExpr)
            Z3_decl_kind.Z3_OP_CONCAT -> expr.convertReduced(::mkBvConcatExpr)
            Z3_decl_kind.Z3_OP_SIGN_EXT -> expr.convert { arg: KExpr<KBvSort> ->
                val size = expr.funcDecl.parameters[0].int
                mkBvSignExtensionExpr(size, arg)
            }
            Z3_decl_kind.Z3_OP_ZERO_EXT -> expr.convert { arg: KExpr<KBvSort> ->
                val size = expr.funcDecl.parameters[0].int
                mkBvZeroExtensionExpr(size, arg)
            }
            Z3_decl_kind.Z3_OP_EXTRACT -> expr.convert { arg: KExpr<KBvSort> ->
                val high = expr.funcDecl.parameters[0].int
                val low = expr.funcDecl.parameters[1].int
                mkBvExtractExpr(high, low, arg)
            }
            Z3_decl_kind.Z3_OP_REPEAT -> expr.convert { arg: KExpr<KBvSort> ->
                val repeatCount = expr.funcDecl.parameters[0].int
                mkBvRepeatExpr(repeatCount, arg)
            }
            Z3_decl_kind.Z3_OP_BREDOR -> expr.convert(::mkBvReductionOrExpr)
            Z3_decl_kind.Z3_OP_BREDAND -> expr.convert(::mkBvReductionAndExpr)
            Z3_decl_kind.Z3_OP_BCOMP -> TODO("bcomp conversion is not supported")
            Z3_decl_kind.Z3_OP_BSHL -> expr.convert(::mkBvShiftLeftExpr)
            Z3_decl_kind.Z3_OP_BLSHR -> expr.convert(::mkBvLogicalShiftRightExpr)
            Z3_decl_kind.Z3_OP_BASHR -> expr.convert(::mkBvArithShiftRightExpr)
            Z3_decl_kind.Z3_OP_ROTATE_LEFT -> expr.convert { arg: KExpr<KBvSort> ->
                val rotation = expr.funcDecl.parameters[0].int
                mkBvRotateLeftExpr(rotation, arg)
            }
            Z3_decl_kind.Z3_OP_ROTATE_RIGHT -> expr.convert { arg: KExpr<KBvSort> ->
                val rotation = expr.funcDecl.parameters[0].int
                mkBvRotateRightExpr(rotation, arg)
            }
            Z3_decl_kind.Z3_OP_EXT_ROTATE_LEFT -> expr.convert(::mkBvRotateLeftExpr)
            Z3_decl_kind.Z3_OP_EXT_ROTATE_RIGHT -> expr.convert(::mkBvRotateRightExpr)
            Z3_decl_kind.Z3_OP_BIT2BOOL -> TODO("bit2bool conversion is not supported")
            Z3_decl_kind.Z3_OP_INT2BV -> TODO("int2bv conversion is not supported")
            Z3_decl_kind.Z3_OP_BV2INT -> TODO("bv2int conversion is not supported")
            Z3_decl_kind.Z3_OP_CARRY -> expr.convert { a0: KExpr<KBvSort>, a1: KExpr<KBvSort>, a2: KExpr<KBvSort> ->
                mkBvOrExpr(
                    mkBvAndExpr(a0, a1),
                    mkBvOrExpr(mkBvAndExpr(a0, a2), mkBvAndExpr(a1, a2))
                )
            }
            Z3_decl_kind.Z3_OP_XOR3 -> expr.convertReduced(::mkBvXorExpr)
            Z3_decl_kind.Z3_OP_BSMUL_NO_OVFL -> expr.convert { a0: KExpr<KBvSort>, a1: KExpr<KBvSort> ->
                mkBvMulNoOverflowExpr(a0, a1, isSigned = true)
            }
            Z3_decl_kind.Z3_OP_BUMUL_NO_OVFL -> expr.convert { a0: KExpr<KBvSort>, a1: KExpr<KBvSort> ->
                mkBvMulNoOverflowExpr(a0, a1, isSigned = false)
            }
            Z3_decl_kind.Z3_OP_BSMUL_NO_UDFL -> expr.convert(::mkBvMulNoUnderflowExpr)
            else -> TODO("${expr.funcDecl} is not supported")
        }
    }

    fun Expr<*>.findConvertedExpr(): KExpr<*>? = z3InternCtx.findConvertedExpr(this)

    inline fun ensureArgsAndConvert(
        expr: Expr<*>,
        args: Array<out Expr<*>>,
        expectedSize: Int,
        converter: (List<KExpr<*>>) -> KExpr<*>
    ): KExpr<*>? {
        check(args.size == expectedSize) { "arguments size mismatch: expected $expectedSize, actual ${args.size}" }
        val convertedArgs = mutableListOf<KExpr<*>>()
        var exprAdded = false
        var argsReady = true
        for (arg in args) {
            val converted = arg.findConvertedExpr()
            if (converted != null) {
                convertedArgs.add(converted)
                continue
            }
            argsReady = false
            if (!exprAdded) {
                exprStack.add(expr)
                exprAdded = true
            }
            exprStack.add(arg)
        }

        if (!argsReady) return null

        return converter(convertedArgs)
    }

    @Suppress("UNCHECKED_CAST")
    inline fun <T : KSort, A0 : KSort> Expr<*>.convert(
        op: (KExpr<A0>) -> KExpr<T>
    ) = ensureArgsAndConvert(this, args, 1) { args -> op(args[0] as KExpr<A0>) }

    @Suppress("UNCHECKED_CAST")
    inline fun <T : KSort, A0 : KSort, A1 : KSort> Expr<*>.convert(
        op: (KExpr<A0>, KExpr<A1>) -> KExpr<T>
    ) = ensureArgsAndConvert(this, args, 2) { args -> op(args[0] as KExpr<A0>, args[1] as KExpr<A1>) }

    @Suppress("UNCHECKED_CAST")
    inline fun <T : KSort, A0 : KSort, A1 : KSort, A2 : KSort> Expr<*>.convert(
        op: (KExpr<A0>, KExpr<A1>, KExpr<A2>) -> KExpr<T>
    ) = ensureArgsAndConvert(this, args, 3) { args ->
        op(args[0] as KExpr<A0>, args[1] as KExpr<A1>, args[2] as KExpr<A2>)
    }

    @Suppress("UNCHECKED_CAST")
    inline fun <T : KSort, A : KSort> Expr<*>.convertList(
        op: (List<KExpr<A>>) -> KExpr<T>
    ) = ensureArgsAndConvert(this, args, numArgs) { args -> op(args as List<KExpr<A>>) }

    @Suppress("UNCHECKED_CAST")
    inline fun <T : KSort> Expr<*>.convertReduced(
        op: (KExpr<T>, KExpr<T>) -> KExpr<T>
    ) = ensureArgsAndConvert(this, args, numArgs) { args -> (args as List<KExpr<T>>).reduce(op) }

    open fun convertNumeral(expr: Expr<*>): KExpr<*> = when (expr.sort.sortKind) {
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

    open fun convertQuantifier(expr: Expr<*>): KExpr<KBoolSort>? = with(ctx) {
        expr as Quantifier

        val z3Bounds = expr.boundVariableSorts.zip(expr.boundVariableNames).map { (sort, name) ->
            expr.ctx.mkConst(name, sort)
        }.reversed()

        val preparedBody = expr.body.substituteVars(z3Bounds.toTypedArray())

        val body = preparedBody.findConvertedExpr()
        if (body == null) {
            exprStack.add(expr)
            exprStack.add(preparedBody)
            return null
        }

        @Suppress("UNCHECKED_CAST")
        body as? KExpr<KBoolSort> ?: error("Body is not properly converted")

        val bounds = z3Bounds.map { it.funcDecl.convert<KSort>() }

        when {
            expr.isUniversal -> mkUniversalQuantifier(body, bounds)
            expr.isExistential -> mkExistentialQuantifier(body, bounds)
            expr.isLambda -> TODO("array lambda converter")
            else -> TODO("unexpected quantifier: $expr")
        }
    }
}
