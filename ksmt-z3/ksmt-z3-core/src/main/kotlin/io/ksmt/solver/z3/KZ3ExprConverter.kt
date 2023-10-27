package io.ksmt.solver.z3

import com.microsoft.z3.Native
import com.microsoft.z3.enumerations.Z3_ast_kind
import com.microsoft.z3.enumerations.Z3_decl_kind
import com.microsoft.z3.enumerations.Z3_parameter_kind
import com.microsoft.z3.enumerations.Z3_sort_kind
import com.microsoft.z3.enumerations.Z3_symbol_kind
import com.microsoft.z3.fpSignOrNull
import com.microsoft.z3.getAppArgs
import com.microsoft.z3.getArraySortDomain
import com.microsoft.z3.intOrNull
import com.microsoft.z3.longOrNull
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap
import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.decl.KFuncDecl
import io.ksmt.expr.KBitVecValue
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFpRoundingMode
import io.ksmt.expr.KFpRoundingModeExpr
import io.ksmt.expr.KIntNumExpr
import io.ksmt.expr.KInterpretedValue
import io.ksmt.expr.KRealNumExpr
import io.ksmt.expr.rewrite.KExprUninterpretedDeclCollector
import io.ksmt.solver.KSolverUnsupportedFeatureException
import io.ksmt.solver.util.ExprConversionResult
import io.ksmt.solver.util.KExprConverterUtils.argumentsConversionRequired
import io.ksmt.solver.util.KExprLongConverterBase
import io.ksmt.sort.KArithSort
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import io.ksmt.utils.asExpr
import io.ksmt.utils.uncheckedCast

open class KZ3ExprConverter(
    private val ctx: KContext,
    private val z3Ctx: KZ3Context,
    private val model: KZ3Model? = null
) : KExprLongConverterBase() {

    private val internalizer = KZ3ExprInternalizer(ctx, z3Ctx)

    @JvmField
    val nCtx: Long = z3Ctx.nCtx

    private val converterLocalCache = Long2ObjectOpenHashMap<KExpr<*>>()

    override fun findConvertedNative(expr: Long): KExpr<*>? {
        val localCached = converterLocalCache.get(expr)
        if (localCached != null) return localCached

        val globalCached = z3Ctx.findConvertedExpr(expr)
        if (globalCached != null) {
            converterLocalCache.put(expr, globalCached)
        }

        return globalCached
    }

    override fun saveConvertedNative(native: Long, converted: KExpr<*>) {
        if (converterLocalCache.putIfAbsent(native, converted) == null) {
            /**
             * It is not safe to save complex converted expressions
             * because they may contain model bound variables.
             * */
            if (converted is KInterpretedValue<*>) {
                z3Ctx.saveConvertedExpr(native, converted)
            } else {
                z3Ctx.saveConverterNativeObject(native)
            }
        }
    }

    fun <T : KSort> Long.convertExpr(): KExpr<T> = convertFromNative(this)

    @Suppress("UNCHECKED_CAST")
    fun <T : KSort> Long.convertSort(): T = z3Ctx.convertSort(this) {
        convertNativeSort(this)
    } as? T ?: error("sort is not properly converted")

    @Suppress("UNCHECKED_CAST")
    fun <T : KSort> Long.convertDecl(): KDecl<T> = z3Ctx.convertDecl(this) {
        convertNativeDecl(this)
    } as? KDecl<T> ?: error("decl is not properly converted")

    open fun convertNativeSymbol(symbol: Long): String =
        when (Z3_symbol_kind.fromInt(Native.getSymbolKind(nCtx, symbol))!!) {
            Z3_symbol_kind.Z3_INT_SYMBOL -> Native.getSymbolInt(nCtx, symbol).toString()
            Z3_symbol_kind.Z3_STRING_SYMBOL -> Native.getSymbolString(nCtx, symbol)
        }

    open fun convertNativeDecl(decl: Long): KDecl<*> = with(ctx) {
        val range = Native.getRange(nCtx, decl).convertSort<KSort>()
        val domainSize = Native.getDomainSize(nCtx, decl)
        val domain = List(domainSize) { idx ->
            Native.getDomain(nCtx, decl, idx).convertSort<KSort>()
        }
        val name = convertNativeSymbol(Native.getDeclName(nCtx, decl))
        return mkFuncDecl(name, range, domain)
    }

    open fun convertNativeSort(sort: Long): KSort = with(ctx) {
        when (Z3_sort_kind.fromInt(Native.getSortKind(nCtx, sort))) {
            Z3_sort_kind.Z3_BOOL_SORT -> boolSort
            Z3_sort_kind.Z3_INT_SORT -> intSort
            Z3_sort_kind.Z3_REAL_SORT -> realSort
            Z3_sort_kind.Z3_ARRAY_SORT -> convertNativeArraySort(sort)
            Z3_sort_kind.Z3_BV_SORT -> mkBvSort(Native.getBvSortSize(nCtx, sort).toUInt())
            Z3_sort_kind.Z3_FLOATING_POINT_SORT ->
                mkFpSort(Native.fpaGetEbits(nCtx, sort).toUInt(), Native.fpaGetSbits(nCtx, sort).toUInt())
            Z3_sort_kind.Z3_UNINTERPRETED_SORT -> {
                val name = convertNativeSymbol(Native.getSortName(nCtx, sort))
                mkUninterpretedSort(name)
            }
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

    open fun KContext.convertNativeArraySort(sort: Long): KSort {
        val domain = getArraySortDomain(nCtx, sort).map { it.convertSort<KSort>() }
        val range = Native.getArraySortRange(nCtx, sort).convertSort<KSort>()

        return when (domain.size) {
            KArraySort.DOMAIN_SIZE -> mkArraySort(domain.single(), range)
            KArray2Sort.DOMAIN_SIZE -> mkArraySort(domain.first(), domain.last(), range)
            KArray3Sort.DOMAIN_SIZE -> {
                val (d0, d1, d2) = domain
                mkArraySort(d0, d1, d2, range)
            }

            else -> mkArrayNSort(domain, range)
        }
    }

    /**
     * Convert expression non-recursively.
     * 1. Ensure all expression arguments are already converted and available in [z3Ctx].
     * If any argument is not converted [argumentsConversionRequired] is returned.
     * 2. If all arguments are available converted expression is returned.
     * */
    override fun convertNativeExpr(expr: Long): ExprConversionResult =
        when (Z3_ast_kind.fromInt(Native.getAstKind(nCtx, expr))) {
            Z3_ast_kind.Z3_NUMERAL_AST -> convertNumeral(expr)
            Z3_ast_kind.Z3_APP_AST -> convertApp(expr)
            Z3_ast_kind.Z3_QUANTIFIER_AST -> convertQuantifier(expr)

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
        "LongMethod",
        "ComplexMethod"
    )
    open fun convertApp(expr: Long): ExprConversionResult = with(ctx) {
        val decl = Native.getAppDecl(nCtx, expr)
        when (val declKind = Z3_decl_kind.fromInt(Native.getDeclKind(nCtx, decl))) {
            Z3_decl_kind.Z3_OP_TRUE -> convert { trueExpr }
            Z3_decl_kind.Z3_OP_FALSE -> convert { falseExpr }
            Z3_decl_kind.Z3_OP_UNINTERPRETED -> expr.convertList { args: List<KExpr<KSort>> ->
                val numParameters = Native.getDeclNumParameters(nCtx, decl)
                val convertedParametrizedDeclApp = if (numParameters != 0) {
                    tryConvertParametrizedDeclApp(expr, decl, args, numParameters)
                } else {
                    null
                }
                convertedParametrizedDeclApp ?: mkApp(decl.convertDecl(), args)
            }
            Z3_decl_kind.Z3_OP_AND -> expr.convertList(::mkAnd)
            Z3_decl_kind.Z3_OP_OR -> expr.convertList(::mkOr)
            Z3_decl_kind.Z3_OP_XOR -> expr.convert(::mkXor)
            Z3_decl_kind.Z3_OP_NOT -> expr.convert(::mkNot)
            Z3_decl_kind.Z3_OP_IMPLIES -> expr.convert(::mkImplies)
            Z3_decl_kind.Z3_OP_IFF,
            Z3_decl_kind.Z3_OP_EQ -> expr.convert(::mkEq)
            Z3_decl_kind.Z3_OP_DISTINCT -> expr.convertList(::mkDistinct)
            Z3_decl_kind.Z3_OP_ITE -> expr.convert(::mkIte)
            Z3_decl_kind.Z3_OP_LE -> expr.convert(::mkArithLe)
            Z3_decl_kind.Z3_OP_GE -> expr.convert(::mkArithGe)
            Z3_decl_kind.Z3_OP_LT -> expr.convert(::mkArithLt)
            Z3_decl_kind.Z3_OP_GT -> expr.convert(::mkArithGt)
            Z3_decl_kind.Z3_OP_ADD -> expr.convertList(::mkArithAdd)
            Z3_decl_kind.Z3_OP_SUB -> expr.convertList(::mkArithSub)
            Z3_decl_kind.Z3_OP_MUL -> expr.convertList(::mkArithMul)
            Z3_decl_kind.Z3_OP_UMINUS -> expr.convert(::mkArithUnaryMinus)
            Z3_decl_kind.Z3_OP_DIV -> expr.convert(::mkArithDiv)
            // Special case when arguments are integers
            Z3_decl_kind.Z3_OP_IDIV -> expr.convert(::mkArithDiv)
            Z3_decl_kind.Z3_OP_POWER -> expr.convert(::mkArithPower)
            Z3_decl_kind.Z3_OP_REM -> expr.convert(::mkIntRem)
            Z3_decl_kind.Z3_OP_MOD -> expr.convert(::mkIntMod)
            Z3_decl_kind.Z3_OP_TO_REAL -> expr.convert(::mkIntToReal)
            Z3_decl_kind.Z3_OP_TO_INT -> expr.convert(::mkRealToInt)
            Z3_decl_kind.Z3_OP_IS_INT -> expr.convert(::mkRealIsInt)
            Z3_decl_kind.Z3_OP_ANUM -> {
                error("Unexpected algebraic numeral in app converter")
            }
            Z3_decl_kind.Z3_OP_AGNUM -> {
                throw KSolverUnsupportedFeatureException("Irrational algebraic numbers are not supported")
            }
            Z3_decl_kind.Z3_OP_STORE -> convertArrayStore(expr)
            Z3_decl_kind.Z3_OP_SELECT -> convertArraySelect(expr)
            Z3_decl_kind.Z3_OP_CONST_ARRAY -> expr.convert { arg: KExpr<KSort> ->
                val sort = Native.getRange(nCtx, decl).convertSort<KArraySortBase<KSort>>()
                mkArrayConst(sort, arg)
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
                val size = Native.getDeclIntParameter(nCtx, decl, 0)

                mkBvSignExtensionExpr(size, arg)
            }
            Z3_decl_kind.Z3_OP_ZERO_EXT -> expr.convert { arg: KExpr<KBvSort> ->
                val size = Native.getDeclIntParameter(nCtx, decl, 0)

                mkBvZeroExtensionExpr(size, arg)
            }
            Z3_decl_kind.Z3_OP_EXTRACT -> expr.convert { arg: KExpr<KBvSort> ->
                val high = Native.getDeclIntParameter(nCtx, decl, 0)
                val low = Native.getDeclIntParameter(nCtx, decl, 1)

                mkBvExtractExpr(high, low, arg)
            }
            Z3_decl_kind.Z3_OP_REPEAT -> expr.convert { arg: KExpr<KBvSort> ->
                val repeatCount = Native.getDeclIntParameter(nCtx, decl, 0)

                mkBvRepeatExpr(repeatCount, arg)
            }
            Z3_decl_kind.Z3_OP_BREDOR -> expr.convert(::mkBvReductionOrExpr)
            Z3_decl_kind.Z3_OP_BREDAND -> expr.convert(::mkBvReductionAndExpr)
            Z3_decl_kind.Z3_OP_BCOMP -> expr.convert { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> ->
                mkIte(arg0 eq arg1, mkBv(true), mkBv(false))
            }
            Z3_decl_kind.Z3_OP_BSHL -> expr.convert(::mkBvShiftLeftExpr)
            Z3_decl_kind.Z3_OP_BLSHR -> expr.convert(::mkBvLogicalShiftRightExpr)
            Z3_decl_kind.Z3_OP_BASHR -> expr.convert(::mkBvArithShiftRightExpr)
            Z3_decl_kind.Z3_OP_ROTATE_LEFT -> expr.convert { arg: KExpr<KBvSort> ->
                val rotation = Native.getDeclIntParameter(nCtx, decl, 0)

                mkBvRotateLeftIndexedExpr(rotation, arg)
            }
            Z3_decl_kind.Z3_OP_ROTATE_RIGHT -> expr.convert { arg: KExpr<KBvSort> ->
                val rotation = Native.getDeclIntParameter(nCtx, decl, 0)

                mkBvRotateRightIndexedExpr(rotation, arg)
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
                val sort = Native.getRange(nCtx, decl).convertSort<KArraySortBase<KSort>>()
                val z3Decl = Native.getAsArrayFuncDecl(nCtx, expr).convertDecl<KSort>()
                val funDecl = z3Decl as? KFuncDecl<KSort> ?: error("unexpected as-array decl $z3Decl")
                mkFunctionAsArray(sort, funDecl)
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
                val size = Native.getDeclIntParameter(nCtx, decl, 0)
                mkFpToBvExpr(rm, value, bvSize = size, isSigned = false)
            }

            Z3_decl_kind.Z3_OP_FPA_TO_SBV -> expr.convert { rm: KExpr<KFpRoundingModeSort>, value: KExpr<KFpSort> ->
                val size = Native.getDeclIntParameter(nCtx, decl, 0)
                mkFpToBvExpr(rm, value, bvSize = size, isSigned = true)
            }

            Z3_decl_kind.Z3_OP_FPA_FP -> expr.convert(::mkFpFromBvExpr)
            Z3_decl_kind.Z3_OP_FPA_TO_REAL -> expr.convert(::mkFpToRealExpr)
            Z3_decl_kind.Z3_OP_FPA_TO_IEEE_BV -> expr.convert(::mkFpToIEEEBvExpr)
            Z3_decl_kind.Z3_OP_FPA_TO_FP -> convertFpaToFp(expr)
            Z3_decl_kind.Z3_OP_FPA_PLUS_INF -> convert {
                val sort = Native.getSort(nCtx, expr).convertSort<KFpSort>()
                mkFpInf(signBit = false, sort)
            }

            Z3_decl_kind.Z3_OP_FPA_MINUS_INF -> convert {
                val sort = Native.getSort(nCtx, expr).convertSort<KFpSort>()
                mkFpInf(signBit = true, sort)
            }

            Z3_decl_kind.Z3_OP_FPA_NAN -> convert {
                val sort = Native.getSort(nCtx, expr).convertSort<KFpSort>()
                mkFpNaN(sort)
            }

            Z3_decl_kind.Z3_OP_FPA_PLUS_ZERO -> convert {
                val sort = Native.getSort(nCtx, expr).convertSort<KFpSort>()
                mkFpZero(signBit = false, sort)
            }

            Z3_decl_kind.Z3_OP_FPA_MINUS_ZERO -> convert {
                val sort = Native.getSort(nCtx, expr).convertSort<KFpSort>()
                mkFpZero(signBit = true, sort)
            }

            Z3_decl_kind.Z3_OP_FPA_NUM -> convertFpNumeral(expr, Native.getSort(nCtx, expr))
            Z3_decl_kind.Z3_OP_FPA_TO_FP_UNSIGNED ->
                expr.convert { rm: KExpr<KFpRoundingModeSort>, value: KExpr<KBvSort> ->
                    val fpSort = Native.getSort(nCtx, expr).convertSort<KFpSort>()
                    mkBvToFpExpr(fpSort, rm, value, signed = false)
                }
            Z3_decl_kind.Z3_OP_FPA_BVWRAP,
            Z3_decl_kind.Z3_OP_FPA_BV2RM -> {
                throw KSolverUnsupportedFeatureException("Fp $declKind is not supported")
            }

           Z3_decl_kind.Z3_OP_INTERNAL -> {
                throw KSolverUnsupportedFeatureException("Z3 internal decl $declKind is not supported")
            }

            Z3_decl_kind.Z3_OP_RECURSIVE -> {
                throw KSolverUnsupportedFeatureException("Recursive decl $declKind is not supported")
            }

            else -> throw KSolverUnsupportedFeatureException("$declKind is not supported")
        }
    }

    /**
     * Parameterized function declarations are usually internal
     * Z3 functions with some additional information.
     * We trying recognize such parameterized functions
     * and use this information for conversion purposes.
     * If the recognition fails, then we treat these declarations
     * as ordinary uninterpreted functions.
     *
     * Currently, we can recognize declarations of an uninterpreted sort values.
     * */
    private fun tryConvertParametrizedDeclApp(
        expr: Long,
        nativeDecl: Long,
        args: List<KExpr<KSort>>,
        numParameters: Int
    ): KExpr<*>? {
        if (numParameters != 2 || args.isNotEmpty()) return null

        val param0Kind = Native.getDeclParameterKind(nCtx, nativeDecl, 0).let { Z3_parameter_kind.fromInt(it) }
        val param1Kind = Native.getDeclParameterKind(nCtx, nativeDecl, 1).let { Z3_parameter_kind.fromInt(it) }

        if (param0Kind != Z3_parameter_kind.Z3_PARAMETER_INT && param1Kind != Z3_parameter_kind.Z3_PARAMETER_SORT) {
            return null
        }

        val nativeSort = Native.getDeclSortParameter(nCtx, nativeDecl, 1)
        val sort = nativeSort.convertSort<KSort>()

        if (sort !is KUninterpretedSort) return null

        /**
         * Decl is an interpreted value of an uninterpreted sort.
         * Such a declaration may appear only in models.
         * */
        model ?: error("Uninterpreted value without model")

        return model.resolveUninterpretedSortValue(sort, nativeDecl).also {
            // Save here to skip save to the global cache
            converterLocalCache.putIfAbsent(expr, it)
        }
    }

    @Suppress("ComplexMethod", "MagicNumber")
    private fun KContext.convertFpaToFp(expr: Long): ExprConversionResult {
        val fpSort = Native.getSort(nCtx, expr).convertSort<KFpSort>()
        val args = getAppArgs(nCtx, expr)
        val sorts = args.map { Native.getSort(nCtx, it) }
        val sortKinds = sorts.map { Z3_sort_kind.fromInt(Native.getSortKind(nCtx, it)) }

        return when {
            args.size == 1 && sortKinds.single() == Z3_sort_kind.Z3_BV_SORT -> expr.convert { bv: KExpr<KBvSort> ->
                val exponentBits = fpSort.exponentBits.toInt()
                val size = bv.sort.sizeBits.toInt()

                @Suppress("UNCHECKED_CAST")
                val sign = mkBvExtractExpr(size - 1, size - 1, bv) as KExpr<KBv1Sort>
                val exponent = mkBvExtractExpr(size - 2, size - exponentBits - 1, bv)
                val significand = mkBvExtractExpr(size - exponentBits - 2, 0, bv)

                mkFpFromBvExpr(sign, exponent, significand)
            }
            args.size == 2 && sortKinds[0] == Z3_sort_kind.Z3_ROUNDING_MODE_SORT -> when (sortKinds[1]) {
                Z3_sort_kind.Z3_FLOATING_POINT_SORT ->
                    expr.convert { rm: KExpr<KFpRoundingModeSort>, value: KExpr<KFpSort> ->
                        mkFpToFpExpr(fpSort, rm, value)
                    }
                Z3_sort_kind.Z3_REAL_SORT ->
                    expr.convert { rm: KExpr<KFpRoundingModeSort>, value: KExpr<KRealSort> ->
                        mkRealToFpExpr(fpSort, rm, value)
                    }
                Z3_sort_kind.Z3_BV_SORT ->
                    expr.convert { rm: KExpr<KFpRoundingModeSort>, value: KExpr<KBvSort> ->
                        mkBvToFpExpr(fpSort, rm, value, signed = true)
                    }
                else -> TODO("unsupported fpaTofp: $expr")
            }
            args.size == 3 && sortKinds.all { it == Z3_sort_kind.Z3_BV_SORT } ->
                expr.convert { sign: KExpr<KBv1Sort>, exp: KExpr<KBvSort>, significand: KExpr<KBvSort> ->
                    mkFpFromBvExpr(sign, exp, significand)
                }
            args.size == 3 && sortKinds[0] == Z3_sort_kind.Z3_ROUNDING_MODE_SORT
                    && sortKinds.drop(1).all { it == Z3_sort_kind.Z3_INT_SORT || it == Z3_sort_kind.Z3_REAL_SORT } ->
                expr.convert(::convertRealToFpExpr)
            else -> error("unexpected fpaTofp: ${Native.astToString(nCtx, expr)}")
        }
    }

    private fun convertRealToFpExpr(
        rm: KExpr<KFpRoundingModeSort>,
        arg1: KExpr<KArithSort>,
        arg2: KExpr<KArithSort>
    ): KExpr<KFpSort> = with(ctx) {
        TODO("unsupported fpaTofp: ${rm.sort} + real (${arg1.sort}) + int (${arg2.sort}) -> float")
    }

    open fun convertNumeral(expr: Long): ExprConversionResult {
        val sort = Native.getSort(nCtx, expr)
        return when (Z3_sort_kind.fromInt(Native.getSortKind(nCtx, sort))) {
            Z3_sort_kind.Z3_INT_SORT -> convert { convertIntNumeral(expr) }
            Z3_sort_kind.Z3_REAL_SORT -> convert { convertRealNumeral(expr) }
            Z3_sort_kind.Z3_BV_SORT -> convert { convertBvNumeral(expr, sort) }
            Z3_sort_kind.Z3_ROUNDING_MODE_SORT -> convert { convertFpRmNumeral(expr) }
            Z3_sort_kind.Z3_FLOATING_POINT_SORT -> convertFpNumeral(expr, sort)
            else -> TODO("numerals with ${Native.sortToString(nCtx, sort)} are not supported")
        }
    }

    @Suppress("MemberVisibilityCanBePrivate")
    fun convertIntNumeral(expr: Long): KIntNumExpr = with(ctx) {
        intOrNull(nCtx, expr)
            ?.let { mkIntNum(it) }
            ?: longOrNull(nCtx, expr)?.let { mkIntNum(it) }
            ?: mkIntNum(Native.getNumeralString(nCtx, expr))
    }

    @Suppress("MemberVisibilityCanBePrivate")
    fun convertRealNumeral(expr: Long): KRealNumExpr = with(ctx) {
        val numerator = z3Ctx.temporaryAst(Native.getNumerator(nCtx, expr))
        val denominator = z3Ctx.temporaryAst(Native.getDenominator(nCtx, expr))
        mkRealNum(convertIntNumeral(numerator), convertIntNumeral(denominator)).also {
            z3Ctx.releaseTemporaryAst(numerator)
            z3Ctx.releaseTemporaryAst(denominator)
        }
    }

    @Suppress("MemberVisibilityCanBePrivate")
    fun convertBvNumeral(expr: Long, sort: Long): KBitVecValue<*> = with(ctx) {
        val sizeBits = Native.getBvSortSize(nCtx, sort).toUInt()
        val bits = Native.getNumeralBinaryString(nCtx, expr)
        mkBv(value = bits.padStart(sizeBits.toInt(), '0'), sizeBits)
    }

    @Suppress("MemberVisibilityCanBePrivate")
    fun convertFpNumeral(expr: Long, sortx: Long): ExprConversionResult = when {
        Native.isNumeralAst(nCtx, expr) -> {
            with(ctx) {
                val biasedExponentBv = z3Ctx.temporaryAst(Native.fpaGetNumeralExponentBv(nCtx, expr, true))
                val significandBv = z3Ctx.temporaryAst(Native.fpaGetNumeralSignificandBv(nCtx, expr))

                expr.convert(
                    longArrayOf(biasedExponentBv, significandBv)
                ) { exponent: KExpr<KBvSort>, significand: KExpr<KBvSort> ->
                    val sort = sortx.convertSort<KFpSort>()

                    val sign = fpSignOrNull(nCtx, expr) ?: error("unexpected fp value")

                    z3Ctx.releaseTemporaryAst(biasedExponentBv)
                    z3Ctx.releaseTemporaryAst(significandBv)

                    ctx.mkFpBiased(
                        significand = significand as KBitVecValue<*>,
                        biasedExponent = exponent as KBitVecValue<*>,
                        signBit = sign,
                        sort = sort
                    )
                }
            }
        }
        Z3_decl_kind.Z3_OP_FPA_NUM.toInt() == Native.getDeclKind(nCtx, Native.getAppDecl(nCtx, expr)) -> {
            TODO("unexpected fpa num")
        }
        else -> convertApp(expr)
    }

    @Suppress("MemberVisibilityCanBePrivate")
    fun convertFpRmNumeral(expr: Long): KFpRoundingModeExpr = with(ctx) {
        val decl = Native.getAppDecl(nCtx, expr)
        val roundingMode = when (Z3_decl_kind.fromInt(Native.getDeclKind(nCtx, decl))) {
            Z3_decl_kind.Z3_OP_FPA_RM_NEAREST_TIES_TO_EVEN -> KFpRoundingMode.RoundNearestTiesToEven
            Z3_decl_kind.Z3_OP_FPA_RM_NEAREST_TIES_TO_AWAY -> KFpRoundingMode.RoundNearestTiesToAway
            Z3_decl_kind.Z3_OP_FPA_RM_TOWARD_POSITIVE -> KFpRoundingMode.RoundTowardPositive
            Z3_decl_kind.Z3_OP_FPA_RM_TOWARD_NEGATIVE -> KFpRoundingMode.RoundTowardNegative
            Z3_decl_kind.Z3_OP_FPA_RM_TOWARD_ZERO -> KFpRoundingMode.RoundTowardZero
            else -> error("unexpected rounding mode: ${Native.astToString(nCtx, expr)}")
        }
        mkFpRoundingModeExpr(roundingMode)
    }

    private fun convertArrayStore(expr: Long): ExprConversionResult =
        when (Native.getAppNumArgs(nCtx, expr)) {
            KArraySort.DOMAIN_SIZE + 2 -> expr.convert(::mkArray1Store)
            KArray2Sort.DOMAIN_SIZE + 2 -> expr.convert(::mkArray2Store)
            KArray3Sort.DOMAIN_SIZE + 2 -> expr.convertList { args: List<KExpr<KSort>> ->
                val (i0, i1, i2) = args.subList(1, args.lastIndex)
                mkArray3Store(args.first().uncheckedCast(), i0, i1, i2, args.last())
            }
            else -> expr.convertList { args: List<KExpr<KSort>> ->
                mkArrayNStore(args.first().uncheckedCast(), args.subList(1, args.lastIndex), args.last())
            }
        }

    private fun convertArraySelect(expr: Long): ExprConversionResult =
        when (Native.getAppNumArgs(nCtx, expr)) {
            KArraySort.DOMAIN_SIZE + 1 -> expr.convert(::mkArray1Select)
            KArray2Sort.DOMAIN_SIZE + 1 -> expr.convert(::mkArray2Select)
            KArray3Sort.DOMAIN_SIZE + 1 -> expr.convert(::mkArray3Select)
            else -> expr.convertList { args: List<KExpr<KSort>> ->
                mkArrayNSelect(args.first().uncheckedCast(), args.drop(1))
            }
        }

    private fun mkArray1Select(
        array: KExpr<KArraySort<KSort, KSort>>,
        index: KExpr<KSort>
    ) = ctx.mkArraySelect(array, index)

    private fun mkArray2Select(
        array: KExpr<KArray2Sort<KSort, KSort, KSort>>,
        index0: KExpr<KSort>, index1: KExpr<KSort>
    ) = ctx.mkArraySelect(array, index0, index1)

    private fun mkArray3Select(
        array: KExpr<KArray3Sort<KSort, KSort, KSort, KSort>>,
        index0: KExpr<KSort>, index1: KExpr<KSort>, index2: KExpr<KSort>
    ) = ctx.mkArraySelect(array, index0, index1, index2)

    private fun mkArrayNSelect(
        array: KExpr<KArrayNSort<KSort>>,
        indices: List<KExpr<KSort>>
    ) = ctx.mkArrayNSelect(array, indices)

    private fun mkArray1Store(
        array: KExpr<KArraySort<KSort, KSort>>,
        index: KExpr<KSort>,
        value: KExpr<KSort>
    ) = ctx.mkArrayStore(array, index, value)

    private fun mkArray2Store(
        array: KExpr<KArray2Sort<KSort, KSort, KSort>>,
        index0: KExpr<KSort>, index1: KExpr<KSort>,
        value: KExpr<KSort>
    ) = ctx.mkArrayStore(array, index0, index1, value)

    private fun mkArray3Store(
        array: KExpr<KArray3Sort<KSort, KSort, KSort, KSort>>,
        index0: KExpr<KSort>, index1: KExpr<KSort>, index2: KExpr<KSort>,
        value: KExpr<KSort>
    ) = ctx.mkArrayStore(array, index0, index1, index2, value)

    private fun mkArrayNStore(
        array: KExpr<KArrayNSort<KSort>>,
        indices: List<KExpr<KSort>>,
        value: KExpr<KSort>
    ) = ctx.mkArrayNStore(array, indices, value)

    open fun convertQuantifier(expr: Long): ExprConversionResult = with(ctx) {
        val numBound = Native.getQuantifierNumBound(nCtx, expr)
        val boundSorts = List(numBound) { idx -> Native.getQuantifierBoundSort(nCtx, expr, idx) }
        val boundNames = List(numBound) { idx -> Native.getQuantifierBoundName(nCtx, expr, idx) }
        val bodyWithVars = Native.getQuantifierBody(nCtx, expr)

        val bounds = boundSorts.zip(boundNames)
            .map { (sort, name) -> mkConstDecl(convertNativeSymbol(name), sort.convertSort()) }

        val z3Bounds = bounds.map { mkConstApp(it) }
            .map { with(internalizer) { it.internalizeExpr() } }
            .asReversed()

        val preparedBody = z3Ctx.temporaryAst(
            Native.substituteVars(nCtx, bodyWithVars, z3Bounds.size, z3Bounds.toLongArray())
        )

        val body = findConvertedNative(preparedBody)

        if (body == null) {
            exprStack.add(expr)
            exprStack.add(preparedBody)

            return argumentsConversionRequired
        }

        z3Ctx.releaseTemporaryAst(preparedBody)

        val convertedExpr = when {
            Native.isQuantifierForall(nCtx, expr) -> mkUniversalQuantifier(body.asExpr(boolSort), bounds)
            Native.isQuantifierExists(nCtx, expr) -> mkExistentialQuantifier(body.asExpr(boolSort), bounds)
            Native.isLambda(nCtx, expr) -> {
                val usedDeclarations = KExprUninterpretedDeclCollector.collectUninterpretedDeclarations(body)
                if (bounds.all { it !in usedDeclarations }) {
                    val sort = mkArrayAnySort(bounds.map { it.sort }, body.sort)
                    mkArrayConst(sort, body.uncheckedCast())
                } else {
                    mkArrayAnyLambda(bounds, body)
                }
            }
            else -> TODO("unexpected quantifier: ${Native.astToString(nCtx, expr)}")
        }

        ExprConversionResult(convertedExpr)
    }

    private fun KContext.mkArrayAnyLambda(bounds: List<KDecl<*>>, body: KExpr<*>) =
        when (bounds.size) {
            KArraySort.DOMAIN_SIZE -> mkArrayLambda(bounds.single(), body)
            KArray2Sort.DOMAIN_SIZE -> mkArrayLambda(bounds.first(), bounds.last(), body)
            KArray3Sort.DOMAIN_SIZE -> {
                val (b0, b1, b2) = bounds
                mkArrayLambda(b0, b1, b2, body)
            }

            else -> mkArrayNLambda(bounds, body)
        }

    private fun KContext.mkArrayAnySort(domain: List<KSort>, range: KSort) =
        when (domain.size) {
            KArraySort.DOMAIN_SIZE -> mkArraySort(domain.single(), range)
            KArray2Sort.DOMAIN_SIZE -> mkArraySort(domain.first(), domain.last(), range)
            KArray3Sort.DOMAIN_SIZE -> {
                val (b0, b1, b2) = domain
                mkArraySort(b0, b1, b2, range)
            }

            else -> mkArrayNSort(domain, range)
        }

    inline fun <T : KSort, A0 : KSort> Long.convert(op: (KExpr<A0>) -> KExpr<T>) =
        convert(getAppArgs(nCtx, this), op)

    inline fun <T : KSort, A0 : KSort, A1 : KSort> Long.convert(op: (KExpr<A0>, KExpr<A1>) -> KExpr<T>) =
        convert(getAppArgs(nCtx, this), op)

    inline fun <T : KSort, A0 : KSort, A1 : KSort, A2 : KSort> Long.convert(
        op: (KExpr<A0>, KExpr<A1>, KExpr<A2>) -> KExpr<T>
    ) = convert(getAppArgs(nCtx, this), op)

    inline fun <T : KSort, A0 : KSort, A1 : KSort, A2 : KSort, A3 : KSort> Long.convert(
        op: (KExpr<A0>, KExpr<A1>, KExpr<A2>, KExpr<A3>) -> KExpr<T>
    ) = convert(getAppArgs(nCtx, this), op)

    inline fun <T : KSort, A : KSort> Long.convertList(op: (List<KExpr<A>>) -> KExpr<T>) =
        convertList(getAppArgs(nCtx, this), op)

    inline fun <T : KSort> Long.convertReduced(op: (KExpr<T>, KExpr<T>) -> KExpr<T>) =
        convertReduced(getAppArgs(nCtx, this), op)

}
