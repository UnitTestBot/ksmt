package org.ksmt.solver.z3

import com.microsoft.z3.ArithSort
import com.microsoft.z3.ArraySort
import com.microsoft.z3.BitVecNum
import com.microsoft.z3.BitVecSort
import com.microsoft.z3.Expr
import com.microsoft.z3.FPExpr
import com.microsoft.z3.FPNum
import com.microsoft.z3.FPRMNum
import com.microsoft.z3.FPRMSort
import com.microsoft.z3.FPSort
import com.microsoft.z3.FuncDecl
import com.microsoft.z3.IntNum
import com.microsoft.z3.Quantifier
import com.microsoft.z3.RatNum
import com.microsoft.z3.RealSort
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
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.expr.KFpRoundingModeExpr
import org.ksmt.expr.KIntNumExpr
import org.ksmt.expr.KRealNumExpr
import org.ksmt.solver.util.KExprConverterBase
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort

open class KZ3ExprConverter(
    private val ctx: KContext,
    private val z3InternCtx: KZ3InternalizationContext
) : KExprConverterBase<Expr<*>>() {

    override fun findConvertedNative(expr: Expr<*>): KExpr<*>? {
        return z3InternCtx.findConvertedExpr(expr)
    }

    override fun saveConvertedNative(native: Expr<*>, converted: KExpr<*>) {
        z3InternCtx.convertExpr(native) { converted }
    }

    fun <T : KSort> Expr<*>.convert(): KExpr<T> = convertFromNative()

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
            Z3_sort_kind.Z3_FLOATING_POINT_SORT -> {
                val fpSort = sort as FPSort
                mkFpSort(fpSort.eBits.toUInt(), fpSort.sBits.toUInt())
            }

            Z3_sort_kind.Z3_UNINTERPRETED_SORT -> mkUninterpretedSort(sort.name.toString())
            Z3_sort_kind.Z3_ROUNDING_MODE_SORT -> mkFpRoundingModeSort()
            Z3_sort_kind.Z3_DATATYPE_SORT,
            Z3_sort_kind.Z3_RELATION_SORT,
            Z3_sort_kind.Z3_FINITE_DOMAIN_SORT,
            Z3_sort_kind.Z3_SEQ_SORT,
            Z3_sort_kind.Z3_RE_SORT,
            Z3_sort_kind.Z3_CHAR_SORT,
            Z3_sort_kind.Z3_UNKNOWN_SORT -> TODO("$sort is not supported yet")
            null -> error("z3 sort kind cannot be null")
        }
    }

    /**
     * Convert expression non-recursively.
     * 1. Ensure all expression arguments are already converted and available in [z3InternCtx].
     * If any argument is not converted [argumentsConversionRequired] is returned.
     * 2. If all arguments are available converted expression is returned.
     * */
    override fun convertNativeExpr(expr: Expr<*>): ExprConversionResult = when (expr.astKind) {
        Z3_ast_kind.Z3_NUMERAL_AST -> convertNumeral(expr)
        Z3_ast_kind.Z3_APP_AST -> convertApp(expr)
        Z3_ast_kind.Z3_QUANTIFIER_AST -> convertQuantifier(expr as Quantifier)

        /**
         * Vars are only possible in Quantifier bodies and function interpretations.
         * Currently we remove vars in all of these cases and therefore
         * if a var occurs then we are missing something
         * */
        Z3_ast_kind.Z3_VAR_AST -> error("unexpected var")

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
    open fun convertApp(expr: Expr<*>): ExprConversionResult = with(ctx) {
        when (expr.funcDecl.declKind) {
            Z3_decl_kind.Z3_OP_TRUE -> convert { trueExpr }
            Z3_decl_kind.Z3_OP_FALSE -> convert { falseExpr }
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
            Z3_decl_kind.Z3_OP_BAND -> expr.convertReduced(::mkBvAndExpr)
            Z3_decl_kind.Z3_OP_BOR -> expr.convertReduced(::mkBvOrExpr)
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
            Z3_decl_kind.Z3_OP_BV2INT -> expr.convert { arg: KExpr<KBvSort> ->
                // bv2int is always unsigned in Z3
                ctx.mkBv2IntExpr(arg, isSigned = false)
            }
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
            Z3_decl_kind.Z3_OP_AS_ARRAY -> convert {
                val z3Decl = expr.funcDecl.parameters[0].funcDecl

                @Suppress("UNCHECKED_CAST")
                val decl = convertDecl(z3Decl) as? KFuncDecl<KSort>
                    ?: error("unexpected as-array decl $z3Decl")
                mkFunctionAsArray<KSort, KSort>(decl)
            }

            Z3_decl_kind.Z3_OP_FPA_NEG -> expr.convert(::mkFpNegationExpr)
            Z3_decl_kind.Z3_OP_FPA_ADD -> expr.convert(::mkFpAddExpr)
            Z3_decl_kind.Z3_OP_FPA_SUB -> expr.convert(::mkFpSubExpr)
            Z3_decl_kind.Z3_OP_FPA_MUL -> expr.convert(::mkFpMulExpr)
            Z3_decl_kind.Z3_OP_FPA_FMA -> expr.convert(::mkFpFusedMulAddExpr)
            Z3_decl_kind.Z3_OP_FPA_DIV -> expr.convert(::mkFpDivExpr)
            Z3_decl_kind.Z3_OP_FPA_REM -> expr.convert(::mkFpRemExpr)
            Z3_decl_kind.Z3_OP_FPA_ABS -> expr.convert(::mkFpAbsExpr)
            Z3_decl_kind.Z3_OP_FPA_MIN -> expr.convert(::mkFpMinExpr)
            Z3_decl_kind.Z3_OP_FPA_MAX -> expr.convert(::mkFpMaxExpr)
            Z3_decl_kind.Z3_OP_FPA_SQRT -> expr.convert(::mkFpSqrtExpr)
            Z3_decl_kind.Z3_OP_FPA_ROUND_TO_INTEGRAL -> expr.convert(::mkFpRoundToIntegralExpr)
            Z3_decl_kind.Z3_OP_FPA_EQ -> expr.convert(::mkFpEqualExpr)
            Z3_decl_kind.Z3_OP_FPA_LT -> expr.convert(::mkFpLessExpr)
            Z3_decl_kind.Z3_OP_FPA_GT -> expr.convert(::mkFpGreaterExpr)
            Z3_decl_kind.Z3_OP_FPA_LE -> expr.convert(::mkFpLessOrEqualExpr)
            Z3_decl_kind.Z3_OP_FPA_GE -> expr.convert(::mkFpGreaterOrEqualExpr)
            Z3_decl_kind.Z3_OP_FPA_IS_NAN -> expr.convert(::mkFpIsNaNExpr)
            Z3_decl_kind.Z3_OP_FPA_IS_INF -> expr.convert(::mkFpIsInfiniteExpr)
            Z3_decl_kind.Z3_OP_FPA_IS_ZERO -> expr.convert(::mkFpIsZeroExpr)
            Z3_decl_kind.Z3_OP_FPA_IS_NORMAL -> expr.convert(::mkFpIsNormalExpr)
            Z3_decl_kind.Z3_OP_FPA_IS_SUBNORMAL -> expr.convert(::mkFpIsSubnormalExpr)
            Z3_decl_kind.Z3_OP_FPA_IS_NEGATIVE -> expr.convert(::mkFpIsNegativeExpr)
            Z3_decl_kind.Z3_OP_FPA_IS_POSITIVE -> expr.convert(::mkFpIsPositiveExpr)
            Z3_decl_kind.Z3_OP_FPA_TO_UBV -> expr.convert { rm: KExpr<KFpRoundingModeSort>, value: KExpr<KFpSort> ->
                val size = expr.funcDecl.parameters[0].int
                mkFpToBvExpr(rm, value, bvSize = size, isSigned = false)
            }

            Z3_decl_kind.Z3_OP_FPA_TO_SBV -> expr.convert { rm: KExpr<KFpRoundingModeSort>, value: KExpr<KFpSort> ->
                val size = expr.funcDecl.parameters[0].int
                mkFpToBvExpr(rm, value, bvSize = size, isSigned = true)
            }

            Z3_decl_kind.Z3_OP_FPA_FP -> expr.convert(::mkFpFromBvExpr)
            Z3_decl_kind.Z3_OP_FPA_TO_REAL -> expr.convert(::mkFpToRealExpr)
            Z3_decl_kind.Z3_OP_FPA_TO_IEEE_BV -> expr.convert(::mkFpToIEEEBvExpr)
            Z3_decl_kind.Z3_OP_FPA_TO_FP -> convertFpaToFp(expr)
            Z3_decl_kind.Z3_OP_FPA_PLUS_INF -> convert {
                val sort = convertSort(expr.sort) as KFpSort
                mkFpInf(signBit = false, sort)
            }

            Z3_decl_kind.Z3_OP_FPA_MINUS_INF -> convert {
                val sort = convertSort(expr.sort) as KFpSort
                mkFpInf(signBit = true, sort)
            }

            Z3_decl_kind.Z3_OP_FPA_NAN -> convert {
                val sort = convertSort(expr.sort) as KFpSort
                mkFpNan(sort)
            }

            Z3_decl_kind.Z3_OP_FPA_PLUS_ZERO -> convert {
                val sort = convertSort(expr.sort) as KFpSort
                mkFpZero(signBit = false, sort)
            }

            Z3_decl_kind.Z3_OP_FPA_MINUS_ZERO -> convert {
                val sort = convertSort(expr.sort) as KFpSort
                mkFpZero(signBit = true, sort)
            }

            Z3_decl_kind.Z3_OP_FPA_NUM -> convertNumeral(expr as FPExpr)
            Z3_decl_kind.Z3_OP_FPA_TO_FP_UNSIGNED ->
                expr.convert { rm: KExpr<KFpRoundingModeSort>, value: KExpr<KBvSort> ->
                    val fpSort = convertSort(expr.sort) as KFpSort
                    mkBvToFpExpr(fpSort, rm, value, signed = false)
                }
            Z3_decl_kind.Z3_OP_FPA_BVWRAP,
            Z3_decl_kind.Z3_OP_FPA_BV2RM -> {
                TODO("Fp ${expr.funcDecl} (${expr.funcDecl.declKind}) is not supported")
            }
            else -> TODO("${expr.funcDecl} (${expr.funcDecl.declKind}) is not supported")
        }
    }

    @Suppress("ComplexMethod", "MagicNumber")
    private fun KContext.convertFpaToFp(expr: Expr<*>): ExprConversionResult {
        val fpSort = convertSort(expr.sort) as KFpSort
        val args = expr.args
        val sorts = args.map { it.sort }
        return when {
            args.size == 1 && sorts[0] is BitVecSort -> expr.convert { bv: KExpr<KBvSort> ->
                val exponentBits = fpSort.exponentBits.toInt()
                val size = bv.sort.sizeBits.toInt()

                @Suppress("UNCHECKED_CAST")
                val sign = mkBvExtractExpr(size - 1, size - 1, bv) as KExpr<KBv1Sort>
                val exponent = mkBvExtractExpr(size - 2, size - exponentBits - 1, bv)
                val significand = mkBvExtractExpr(size - exponentBits - 2, 0, bv)

                mkFpFromBvExpr(sign, exponent, significand)
            }
            args.size == 2 && sorts[0] is FPRMSort -> when (sorts[1]) {
                is FPSort -> expr.convert { rm: KExpr<KFpRoundingModeSort>, value: KExpr<KFpSort> ->
                    mkFpToFpExpr(fpSort, rm, value)
                }
                is RealSort -> expr.convert { rm: KExpr<KFpRoundingModeSort>, value: KExpr<KRealSort> ->
                    mkRealToFpExpr(fpSort, rm, value)
                }
                is BitVecSort -> expr.convert { rm: KExpr<KFpRoundingModeSort>, value: KExpr<KBvSort> ->
                    mkBvToFpExpr(fpSort, rm, value, signed = true)
                }
                else -> TODO("unsupported fpaTofp: $expr")
            }
            args.size == 3 && sorts[0] is BitVecSort && sorts[1] is BitVecSort && sorts[2] is BitVecSort ->
                expr.convert { sign: KExpr<KBv1Sort>, exp: KExpr<KBvSort>, significand: KExpr<KBvSort> ->
                    mkFpFromBvExpr(sign, exp, significand)
                }

            args.size == 3 && sorts[0] is FPRMSort && sorts[1] is ArithSort && sorts[2] is ArithSort ->
                expr.convert<KFpSort, KFpRoundingModeSort, KArithSort<*>, KArithSort<*>> {
                        rm: KExpr<KFpRoundingModeSort>, arg1: KExpr<KArithSort<*>>, arg2: KExpr<KArithSort<*>> ->
                    TODO("${rm.sort} + real (${arg1.sort}) + int (${arg2.sort}) -> float")
                }
            else -> error("unexpected fpaTofp: $expr")
        }
    }

    open fun convertNumeral(expr: Expr<*>): ExprConversionResult = when (expr.sort.sortKind) {
        Z3_sort_kind.Z3_INT_SORT -> convert { convertNumeral(expr as IntNum) }
        Z3_sort_kind.Z3_REAL_SORT -> convert { convertNumeral(expr as RatNum) }
        Z3_sort_kind.Z3_BV_SORT -> convert { convertNumeral(expr as BitVecNum) }
        Z3_sort_kind.Z3_ROUNDING_MODE_SORT -> convert { convertNumeral(expr as FPRMNum) }
        Z3_sort_kind.Z3_FLOATING_POINT_SORT -> convertNumeral(expr as FPExpr)
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

    @Suppress("MemberVisibilityCanBePrivate")
    fun convertNumeral(expr: FPExpr): ExprConversionResult = when {
        expr is FPNum -> convert {
            with(ctx) {
                val sort = convertSort(expr.sort) as KFpSort
                val sBits = sort.significandBits.toInt()
                val fp64SizeBits = KFp64Sort.exponentBits.toInt() + KFp64Sort.significandBits.toInt()

                // if we have sBits greater than long size bits, take it all, otherwise take last (sBits - 1) bits
                val significandMask = if (sBits < fp64SizeBits) (1L shl (sBits - 1)) - 1 else -1
                // TODO it is not right if we have significand with number of bits greater than 64
                val significand = expr.significandUInt64 and significandMask

                val exponentMask = (1L shl sort.exponentBits.toInt()) - 1
                val exponent = expr.getExponentInt64(false) and exponentMask

                mkFp(significand, exponent, expr.sign, sort)
            }
        }
        expr.funcDecl.declKind == Z3_decl_kind.Z3_OP_FPA_NUM -> {
            TODO("unexpected fpa num")
        }
        else -> convertApp(expr)
    }

    @Suppress("MemberVisibilityCanBePrivate")
    fun convertNumeral(expr: FPRMNum): KFpRoundingModeExpr = with(ctx) {
        val roundingMode = when (expr.funcDecl.declKind) {
            Z3_decl_kind.Z3_OP_FPA_RM_NEAREST_TIES_TO_EVEN -> KFpRoundingMode.RoundNearestTiesToEven
            Z3_decl_kind.Z3_OP_FPA_RM_NEAREST_TIES_TO_AWAY -> KFpRoundingMode.RoundNearestTiesToAway
            Z3_decl_kind.Z3_OP_FPA_RM_TOWARD_POSITIVE -> KFpRoundingMode.RoundTowardPositive
            Z3_decl_kind.Z3_OP_FPA_RM_TOWARD_NEGATIVE -> KFpRoundingMode.RoundTowardNegative
            Z3_decl_kind.Z3_OP_FPA_RM_TOWARD_ZERO -> KFpRoundingMode.RoundTowardZero
            else -> error("unexpected rounding mode: $expr")
        }
        mkFpRoundingModeExpr(roundingMode)
    }

    open fun convertQuantifier(expr: Quantifier): ExprConversionResult = with(ctx) {
        val z3Bounds = expr.boundVariableSorts
            .zip(expr.boundVariableNames)
            .map { (sort, name) -> expr.ctx.mkConst(name, sort) }
            .asReversed()

        val preparedBody = expr.body.substituteVars(z3Bounds.toTypedArray())

        val body = findConvertedNative(preparedBody)
        if (body == null) {
            exprStack.add(expr)
            exprStack.add(preparedBody)
            return argumentsConversionRequired
        }

        @Suppress("UNCHECKED_CAST")
        body as? KExpr<KBoolSort> ?: error("Body is not properly converted")

        val bounds = z3Bounds.map { it.funcDecl.convert<KSort>() }

        val convertedExpr = when {
            expr.isUniversal -> mkUniversalQuantifier(body, bounds)
            expr.isExistential -> mkExistentialQuantifier(body, bounds)
            expr.isLambda -> TODO("array lambda converter")
            else -> TODO("unexpected quantifier: $expr")
        }
        ExprConversionResult(convertedExpr)
    }

    inline fun <T : KSort, A0 : KSort> Expr<*>.convert(op: (KExpr<A0>) -> KExpr<T>) = convert(args, op)

    inline fun <T : KSort, A0 : KSort, A1 : KSort> Expr<*>.convert(op: (KExpr<A0>, KExpr<A1>) -> KExpr<T>) =
        convert(args, op)

    inline fun <T : KSort, A0 : KSort, A1 : KSort, A2 : KSort> Expr<*>.convert(
        op: (KExpr<A0>, KExpr<A1>, KExpr<A2>) -> KExpr<T>
    ) = convert(args, op)

    inline fun <T : KSort, A0 : KSort, A1 : KSort, A2 : KSort, A3 : KSort> Expr<*>.convert(
        op: (KExpr<A0>, KExpr<A1>, KExpr<A2>, KExpr<A3>) -> KExpr<T>
    ) = convert(args, op)

    inline fun <T : KSort, A : KSort> Expr<*>.convertList(op: (List<KExpr<A>>) -> KExpr<T>) = convertList(args, op)

    inline fun <T : KSort> Expr<*>.convertReduced(op: (KExpr<T>, KExpr<T>) -> KExpr<T>) = convertReduced(args, op)

}
