package io.ksmt.solver.bitwuzla

import io.ksmt.KContext
import io.ksmt.cache.hash
import io.ksmt.cache.structurallyEqual
import io.ksmt.decl.KConstDecl
import io.ksmt.decl.KDecl
import io.ksmt.decl.KFuncDecl
import io.ksmt.expr.KAndExpr
import io.ksmt.expr.KArrayConst
import io.ksmt.expr.KArraySelectBase
import io.ksmt.expr.KBitVec32Value
import io.ksmt.expr.KBitVecValue
import io.ksmt.expr.KConst
import io.ksmt.expr.KEqExpr
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFpRoundingMode
import io.ksmt.expr.KInterpretedValue
import io.ksmt.expr.KIteExpr
import io.ksmt.expr.printer.ExpressionPrinter
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.solver.KSolverUnsupportedFeatureException
import org.ksmt.solver.bitwuzla.bindings.Bitwuzla
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaKind
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaSort
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native
import io.ksmt.solver.util.ExprConversionResult
import io.ksmt.solver.util.KExprLongConverterBase
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import io.ksmt.utils.powerOfTwo
import io.ksmt.utils.uncheckedCast
import java.math.BigInteger

@Suppress("LargeClass")
open class KBitwuzlaExprConverter(
    private val ctx: KContext,
    private val bitwuzlaCtx: KBitwuzlaContext,
    private val scopedVars: Map<BitwuzlaTerm, KDecl<*>>? = null
) : KExprLongConverterBase() {
    private val bitwuzla: Bitwuzla = bitwuzlaCtx.bitwuzla

    private val adapterTermRewriter = AdapterTermRewriter(ctx)

    /**
     * Create KSMT expression from Bitwuzla term.
     *
     * @param expectedSort expected sort of resulting expression
     *
     * @see convertToExpectedIfNeeded
     * @see convertToBoolIfNeeded
     * */
    fun <T : KSort> BitwuzlaTerm.convertExpr(expectedSort: T): KExpr<T> =
        convertFromNative<KSort>(this)
            .convertToExpectedIfNeeded(expectedSort)
            .let { adapterTermRewriter.apply(it) }

    private var uninterpretedSortValueContext: KBitwuzlaUninterpretedSortValueContext? = null

    fun useUninterpretedSortValueContext(ctx: KBitwuzlaUninterpretedSortValueContext) {
        uninterpretedSortValueContext = ctx
    }

    fun resetUninterpretedSortValueContext() {
        uninterpretedSortValueContext = null
    }

    inline fun <T> withUninterpretedSortValueContext(
        ctx: KBitwuzlaUninterpretedSortValueContext,
        body: () -> T
    ): T = try {
        useUninterpretedSortValueContext(ctx)
        body()
    } finally {
        resetUninterpretedSortValueContext()
    }

    override fun findConvertedNative(expr: BitwuzlaTerm): KExpr<*>? =
        bitwuzlaCtx.findConvertedExpr(expr)

    override fun saveConvertedNative(native: BitwuzlaTerm, converted: KExpr<*>) {
        bitwuzlaCtx.saveConvertedExpr(native, converted)
    }

    private fun BitwuzlaSort.convertSort(): KSort = bitwuzlaCtx.convertSort(this) {
        convertSortHelper(this)
    }

    open fun convertSortHelper(sort: BitwuzlaSort): KSort = with(ctx) {
        when {
            Native.bitwuzlaSortIsEqual(sort, bitwuzlaCtx.boolSort) -> {
                boolSort
            }
            Native.bitwuzlaSortIsArray(sort) -> {
                val domain = Native.bitwuzlaSortArrayGetIndex(sort).convertSort()
                val range = Native.bitwuzlaSortArrayGetElement(sort).convertSort()
                mkArraySort(domain, range)
            }
            Native.bitwuzlaSortIsFun(sort) -> {
                error("Fun sorts are not allowed for conversion")
            }
            Native.bitwuzlaSortIsBv(sort) -> {
                val size = Native.bitwuzlaSortBvGetSize(sort)
                mkBvSort(size.toUInt())
            }
            Native.bitwuzlaSortIsFp(sort) -> {
                val exponent = Native.bitwuzlaSortFpGetExpSize(sort)
                val significand = Native.bitwuzlaSortFpGetSigSize(sort)
                mkFpSort(exponent.toUInt(), significand.toUInt())
            }
            Native.bitwuzlaSortIsRm(sort) -> {
                mkFpRoundingModeSort()
            }
            else -> TODO("Given sort $sort is not supported yet")
        }
    }

    @Suppress("LongMethod", "ComplexMethod")
    override fun convertNativeExpr(expr: BitwuzlaTerm): ExprConversionResult = with(ctx) {
        when (val kind = Native.bitwuzlaTermGetBitwuzlaKind(expr)) {
            // constants, functions, values
            BitwuzlaKind.BITWUZLA_KIND_CONST -> convertConst(expr)
            BitwuzlaKind.BITWUZLA_KIND_APPLY -> convertFunctionApp(expr)
            BitwuzlaKind.BITWUZLA_KIND_VAL -> convertValue(expr)
            BitwuzlaKind.BITWUZLA_KIND_VAR -> convertVar(expr)

            // bool
            BitwuzlaKind.BITWUZLA_KIND_IFF,
            BitwuzlaKind.BITWUZLA_KIND_EQUAL -> expr.convert(::convertEqExpr)

            BitwuzlaKind.BITWUZLA_KIND_DISTINCT -> expr.convertList(::convertDistinctExpr)
            BitwuzlaKind.BITWUZLA_KIND_ITE -> expr.convert(::convertIteExpr)

            BitwuzlaKind.BITWUZLA_KIND_IMPLIES -> expr.convert { p: KExpr<KBoolSort>, q: KExpr<KBoolSort> ->
                mkImplies(p.ensureBoolExpr(), q.ensureBoolExpr())
            }

            // array
            BitwuzlaKind.BITWUZLA_KIND_CONST_ARRAY -> expr.convert { value: KExpr<KSort> ->
                val sort: KArraySort<KSort, KSort> = Native.bitwuzlaTermGetSort(expr).convertSort().uncheckedCast()
                convertConstArrayExpr(sort, value)
            }

            BitwuzlaKind.BITWUZLA_KIND_ARRAY_SELECT -> expr.convert(::convertArraySelectExpr)
            BitwuzlaKind.BITWUZLA_KIND_ARRAY_STORE -> expr.convert(::convertArrayStoreExpr)

            // quantifiers
            BitwuzlaKind.BITWUZLA_KIND_LAMBDA,
            BitwuzlaKind.BITWUZLA_KIND_EXISTS,
            BitwuzlaKind.BITWUZLA_KIND_FORALL -> convertQuantifier(expr, kind)

            // bit-vec or bool
            BitwuzlaKind.BITWUZLA_KIND_AND,
            BitwuzlaKind.BITWUZLA_KIND_OR,
            BitwuzlaKind.BITWUZLA_KIND_NOT,
            BitwuzlaKind.BITWUZLA_KIND_XOR -> convertBoolExpr(expr, kind)
            BitwuzlaKind.BITWUZLA_KIND_BV_NOR,
            BitwuzlaKind.BITWUZLA_KIND_BV_NOT,
            BitwuzlaKind.BITWUZLA_KIND_BV_OR,
            BitwuzlaKind.BITWUZLA_KIND_BV_AND,
            BitwuzlaKind.BITWUZLA_KIND_BV_NAND,
            BitwuzlaKind.BITWUZLA_KIND_BV_XOR,
            BitwuzlaKind.BITWUZLA_KIND_BV_XNOR ->
                if (Native.bitwuzlaTermGetSort(expr) == bitwuzlaCtx.boolSort) {
                    convertBoolExpr(expr, kind)
                } else {
                    convertBVExpr(expr, kind)
                }

            // bit-vec
            BitwuzlaKind.BITWUZLA_KIND_BV_ADD,
            BitwuzlaKind.BITWUZLA_KIND_BV_ASHR,
            BitwuzlaKind.BITWUZLA_KIND_BV_COMP,
            BitwuzlaKind.BITWUZLA_KIND_BV_CONCAT,
            BitwuzlaKind.BITWUZLA_KIND_BV_DEC,
            BitwuzlaKind.BITWUZLA_KIND_BV_INC,
            BitwuzlaKind.BITWUZLA_KIND_BV_MUL,
            BitwuzlaKind.BITWUZLA_KIND_BV_NEG,
            BitwuzlaKind.BITWUZLA_KIND_BV_REDAND,
            BitwuzlaKind.BITWUZLA_KIND_BV_REDOR,
            BitwuzlaKind.BITWUZLA_KIND_BV_REDXOR,
            BitwuzlaKind.BITWUZLA_KIND_BV_ROL,
            BitwuzlaKind.BITWUZLA_KIND_BV_ROR,
            BitwuzlaKind.BITWUZLA_KIND_BV_SADD_OVERFLOW,
            BitwuzlaKind.BITWUZLA_KIND_BV_SDIV_OVERFLOW,
            BitwuzlaKind.BITWUZLA_KIND_BV_SDIV,
            BitwuzlaKind.BITWUZLA_KIND_BV_SGE,
            BitwuzlaKind.BITWUZLA_KIND_BV_SGT,
            BitwuzlaKind.BITWUZLA_KIND_BV_SHL,
            BitwuzlaKind.BITWUZLA_KIND_BV_SHR,
            BitwuzlaKind.BITWUZLA_KIND_BV_SLE,
            BitwuzlaKind.BITWUZLA_KIND_BV_SLT,
            BitwuzlaKind.BITWUZLA_KIND_BV_SMOD,
            BitwuzlaKind.BITWUZLA_KIND_BV_SMUL_OVERFLOW,
            BitwuzlaKind.BITWUZLA_KIND_BV_SREM,
            BitwuzlaKind.BITWUZLA_KIND_BV_SSUB_OVERFLOW,
            BitwuzlaKind.BITWUZLA_KIND_BV_SUB,
            BitwuzlaKind.BITWUZLA_KIND_BV_UADD_OVERFLOW,
            BitwuzlaKind.BITWUZLA_KIND_BV_UDIV,
            BitwuzlaKind.BITWUZLA_KIND_BV_UGE,
            BitwuzlaKind.BITWUZLA_KIND_BV_UGT,
            BitwuzlaKind.BITWUZLA_KIND_BV_ULE,
            BitwuzlaKind.BITWUZLA_KIND_BV_ULT,
            BitwuzlaKind.BITWUZLA_KIND_BV_UMUL_OVERFLOW,
            BitwuzlaKind.BITWUZLA_KIND_BV_UREM,
            BitwuzlaKind.BITWUZLA_KIND_BV_USUB_OVERFLOW,
            BitwuzlaKind.BITWUZLA_KIND_BV_EXTRACT,
            BitwuzlaKind.BITWUZLA_KIND_BV_REPEAT,
            BitwuzlaKind.BITWUZLA_KIND_BV_ROLI,
            BitwuzlaKind.BITWUZLA_KIND_BV_RORI,
            BitwuzlaKind.BITWUZLA_KIND_BV_SIGN_EXTEND,
            BitwuzlaKind.BITWUZLA_KIND_BV_ZERO_EXTEND -> convertBVExpr(expr, kind)

            // fp
            BitwuzlaKind.BITWUZLA_KIND_FP_ABS,
            BitwuzlaKind.BITWUZLA_KIND_FP_ADD,
            BitwuzlaKind.BITWUZLA_KIND_FP_DIV,
            BitwuzlaKind.BITWUZLA_KIND_FP_EQ,
            BitwuzlaKind.BITWUZLA_KIND_FP_FMA,
            BitwuzlaKind.BITWUZLA_KIND_FP_FP,
            BitwuzlaKind.BITWUZLA_KIND_FP_GEQ,
            BitwuzlaKind.BITWUZLA_KIND_FP_GT,
            BitwuzlaKind.BITWUZLA_KIND_FP_IS_INF,
            BitwuzlaKind.BITWUZLA_KIND_FP_IS_NAN,
            BitwuzlaKind.BITWUZLA_KIND_FP_IS_NEG,
            BitwuzlaKind.BITWUZLA_KIND_FP_IS_NORMAL,
            BitwuzlaKind.BITWUZLA_KIND_FP_IS_POS,
            BitwuzlaKind.BITWUZLA_KIND_FP_IS_SUBNORMAL,
            BitwuzlaKind.BITWUZLA_KIND_FP_IS_ZERO,
            BitwuzlaKind.BITWUZLA_KIND_FP_LEQ,
            BitwuzlaKind.BITWUZLA_KIND_FP_LT,
            BitwuzlaKind.BITWUZLA_KIND_FP_MAX,
            BitwuzlaKind.BITWUZLA_KIND_FP_MIN,
            BitwuzlaKind.BITWUZLA_KIND_FP_MUL,
            BitwuzlaKind.BITWUZLA_KIND_FP_NEG,
            BitwuzlaKind.BITWUZLA_KIND_FP_REM,
            BitwuzlaKind.BITWUZLA_KIND_FP_RTI,
            BitwuzlaKind.BITWUZLA_KIND_FP_SQRT,
            BitwuzlaKind.BITWUZLA_KIND_FP_SUB,
            BitwuzlaKind.BITWUZLA_KIND_FP_TO_FP_FROM_BV,
            BitwuzlaKind.BITWUZLA_KIND_FP_TO_FP_FROM_FP,
            BitwuzlaKind.BITWUZLA_KIND_FP_TO_FP_FROM_SBV,
            BitwuzlaKind.BITWUZLA_KIND_FP_TO_FP_FROM_UBV,
            BitwuzlaKind.BITWUZLA_KIND_FP_TO_SBV,
            BitwuzlaKind.BITWUZLA_KIND_FP_TO_UBV -> convertFpExpr(expr, kind)

            // unsupported
            BitwuzlaKind.BITWUZLA_NUM_KINDS -> TODO("unsupported kind $kind")
        }
    }

    private fun KContext.convertFunctionApp(expr: BitwuzlaTerm): ExprConversionResult {
        val children = Native.bitwuzlaTermGetChildren(expr)

        check(children.isNotEmpty()) { "Apply has no function term" }

        val function = children[0]
        val isFunctionDecl = Native.bitwuzlaTermGetBitwuzlaKind(function) == BitwuzlaKind.BITWUZLA_KIND_CONST

        val appArgs = if (isFunctionDecl) {
            // convert function decl separately
            children.copyOfRange(fromIndex = 1, toIndex = children.size)
        } else {
            // convert array expression as part of arguments
            children
        }

        return expr.convertList(appArgs) { convertedArgs: List<KExpr<KSort>> ->
            if (isFunctionDecl) {
                val funcDecl = convertFuncDecl(function)
                if (convertedArgs.isNotEmpty() && funcDecl is KConstDecl<*> && funcDecl.sort is KArraySortBase<*>) {
                    val array: KExpr<KArraySortBase<*>> = mkConstApp(funcDecl).uncheckedCast()
                    mkAnyArraySelect(array, convertedArgs).convertToBoolIfNeeded()
                } else {
                    applyFunction(funcDecl, convertedArgs)
                }
            } else {
                val array: KExpr<KArraySortBase<*>> = convertedArgs.first().uncheckedCast()
                val args = convertedArgs.drop(1)

                mkAnyArraySelect(array, args).convertToBoolIfNeeded()
            }
        }
    }

    private fun applyFunction(funcDecl: KFuncDecl<*>, args: List<KExpr<KSort>>): KExpr<*> {
        check(args.size == funcDecl.argSorts.size) { "Function arguments size mismatch" }

        val wellSortedArgs = args.zip(funcDecl.argSorts) { arg, expectedSort ->
            arg.convertToExpectedIfNeeded(expectedSort)
        }

        return funcDecl.apply(wellSortedArgs).convertToBoolIfNeeded()
    }

    private fun KContext.convertFuncDecl(function: BitwuzlaTerm): KFuncDecl<*> {
        val knownFuncDecl = bitwuzlaCtx.convertConstantIfKnown(function)

        if (knownFuncDecl != null) {
            return knownFuncDecl as? KFuncDecl<*> ?: error("Expected a function, actual: $knownFuncDecl")
        }

        // new function
        val domain = Native.bitwuzlaTermFunGetDomainSorts(function).map { it.convertSort() }
        val range = Native.bitwuzlaTermFunGetCodomainSort(function).convertSort()

        return generateDecl(function) { mkFuncDecl(it, range, domain) }
    }

    private fun KContext.convertConst(expr: BitwuzlaTerm): ExprConversionResult = convert {
        val knownConstDecl = bitwuzlaCtx.convertConstantIfKnown(expr)

        val convertedExpr: KExpr<*> = if (knownConstDecl != null) {
            when (knownConstDecl) {
                is KConstDecl<*> -> mkConstApp(knownConstDecl)
                is KFuncDecl<*> -> rewriteFunctionAsArray(knownConstDecl)
                else -> error("Unexpected declaration: $knownConstDecl")
            }
        } else {
            // newly generated constant
            val sort = Native.bitwuzlaTermGetSort(expr)

            if (!Native.bitwuzlaSortIsFun(sort) || Native.bitwuzlaSortIsArray(sort)) {
                val decl = generateDecl(expr) { mkConstDecl(it, sort.convertSort()) }
                mkConstApp(decl)
            } else {
                // newly generated functional constant
                val decl = convertFuncDecl(expr)
                rewriteFunctionAsArray(decl)
            }
        }

        @Suppress("UNCHECKED_CAST")
        return@convert convertedExpr.convertToBoolIfNeeded() as KExpr<KSort>
    }

    private fun KContext.rewriteFunctionAsArray(decl: KFuncDecl<*>): KExpr<*> {
        val sort = mkAnyArraySort(decl.argSorts, decl.sort)
        return mkFunctionAsArray(sort, decl.uncheckedCast())
    }

    private fun KContext.convertValue(expr: BitwuzlaTerm): ExprConversionResult = convert {
        when {
            bitwuzlaCtx.trueTerm == expr -> trueExpr
            bitwuzlaCtx.falseTerm == expr -> falseExpr
            /**
             * Search for cached value first because [Native.bitwuzlaGetBvValue]
             * is only available after check-sat call
             * */
            Native.bitwuzlaTermIsBv(expr) -> bitwuzlaCtx.convertValue(expr) ?: run {
                convertBvValue(expr)
            }
            Native.bitwuzlaTermIsFp(expr) -> bitwuzlaCtx.convertValue(expr) ?: run {
                convertFpValue(expr)
            }
            Native.bitwuzlaTermIsRm(expr) -> bitwuzlaCtx.convertValue(expr) ?: run {
                convertRmValue(expr)
            }
            else -> TODO("unsupported value $expr")
        }
    }

    private fun KContext.convertVar(expr: BitwuzlaTerm): ExprConversionResult = convert {
        val varsScope = scopedVars ?: error("Unexpected var without scope")
        val decl = varsScope[expr] ?: error("Unregistered var")
        mkConstApp(decl)
    }

    private fun KContext.convertBvValue(expr: BitwuzlaTerm): KBitVecValue<KBvSort> {
        val size = Native.bitwuzlaTermBvGetSize(expr)

        val convertedValue = if (Native.bitwuzlaTermIsBvValue(expr)) {
            // convert Bv value from native representation
            when {
                size <= Int.SIZE_BITS -> {
                    val bits = Native.bitwuzlaBvConstNodeGetBitsUInt32(bitwuzla, expr)
                    mkBv(bits, size.toUInt())
                }
                else -> {
                    val intBits = Native.bitwuzlaBvConstNodeGetBitsUIntArray(bitwuzla, expr)
                    val bits = bvBitsToBigInteger(intBits)
                    mkBv(bits, size.toUInt())
                }
            }
        } else {
            val value = Native.bitwuzlaGetBvValue(bitwuzla, expr)
            mkBv(value, size.toUInt())
        }

        bitwuzlaCtx.saveInternalizedValue(convertedValue, expr)

        return convertedValue
    }

    private fun KContext.convertFpValue(expr: BitwuzlaTerm): KExpr<KFpSort> {
        val sort = Native.bitwuzlaTermGetSort(expr).convertSort() as KFpSort

        val convertedValue = if (Native.bitwuzlaTermIsFpValue(expr)) {
            when (sort) {
                fp32Sort -> {
                    val fpBits = Native.bitwuzlaFpConstNodeGetBitsUInt32(bitwuzla, expr)
                    mkFp(Float.fromBits(fpBits), sort)
                }
                fp64Sort -> {
                    val fpBitsArray = Native.bitwuzlaFpConstNodeGetBitsUIntArray(bitwuzla, expr)
                    val higherBits = fpBitsArray[1].toLong() shl Int.SIZE_BITS
                    val lowerBits = fpBitsArray[0].toUInt().toLong()
                    val fpBits = higherBits or lowerBits
                    mkFp(Double.fromBits(fpBits), sort)
                }
                else -> {
                    val fpBitsArray = Native.bitwuzlaFpConstNodeGetBitsUIntArray(bitwuzla, expr)
                    val fpBits = bvBitsToBigInteger(fpBitsArray)

                    val significandMask = powerOfTwo(sort.significandBits - 1u) - BigInteger.ONE
                    val exponentMask = powerOfTwo(sort.exponentBits) - BigInteger.ONE

                    val significandBits = fpBits.and(significandMask)
                    val exponentBits = fpBits.shiftRight(sort.significandBits.toInt() - 1).and(exponentMask)
                    val signBit = fpBits.testBit(sort.significandBits.toInt() + sort.exponentBits.toInt() - 1)

                    mkFpBiased(
                        signBit = signBit,
                        biasedExponent = mkBv(exponentBits, sort.exponentBits),
                        significand = mkBv(significandBits, sort.significandBits - 1u),
                        sort = sort
                    )
                }
            }
        } else {
            val value = Native.bitwuzlaGetFpValue(bitwuzla, expr)

            mkFpFromBvExpr(
                sign = mkBv(value.sign, sizeBits = 1u).uncheckedCast(),
                biasedExponent = mkBv(value.exponent, value.exponent.length.toUInt()),
                significand = mkBv(value.significand, value.significand.length.toUInt())
            )
        }

        bitwuzlaCtx.saveInternalizedValue(convertedValue, expr)

        return convertedValue
    }

    private fun KContext.convertRmValue(expr: BitwuzlaTerm): KExpr<KFpRoundingModeSort> {
        val kind = when {
            Native.bitwuzlaTermIsRmValueRne(expr) -> KFpRoundingMode.RoundNearestTiesToEven
            Native.bitwuzlaTermIsRmValueRna(expr) -> KFpRoundingMode.RoundNearestTiesToAway
            Native.bitwuzlaTermIsRmValueRtp(expr) -> KFpRoundingMode.RoundTowardPositive
            Native.bitwuzlaTermIsRmValueRtn(expr) -> KFpRoundingMode.RoundTowardNegative
            Native.bitwuzlaTermIsRmValueRtz(expr) -> KFpRoundingMode.RoundTowardZero
            else -> error("Unexpected rounding mode")
        }
        return mkFpRoundingModeExpr(kind)
    }

    open fun KContext.convertBoolExpr(expr: BitwuzlaTerm, kind: BitwuzlaKind): ExprConversionResult = when (kind) {
        BitwuzlaKind.BITWUZLA_KIND_BV_AND, BitwuzlaKind.BITWUZLA_KIND_AND -> expr.convertList(::mkAnd)
        BitwuzlaKind.BITWUZLA_KIND_BV_OR, BitwuzlaKind.BITWUZLA_KIND_OR -> expr.convertList(::mkOr)
        BitwuzlaKind.BITWUZLA_KIND_BV_NOT, BitwuzlaKind.BITWUZLA_KIND_NOT -> expr.convert(::mkNot)
        BitwuzlaKind.BITWUZLA_KIND_BV_XOR, BitwuzlaKind.BITWUZLA_KIND_XOR -> expr.convert(::mkXor)
        BitwuzlaKind.BITWUZLA_KIND_BV_NAND -> expr.convertList { args: List<KExpr<KBoolSort>> ->
            mkAnd(args).not()
        }
        BitwuzlaKind.BITWUZLA_KIND_BV_NOR -> expr.convertList { args: List<KExpr<KBoolSort>> ->
            mkOr(args).not()
        }
        BitwuzlaKind.BITWUZLA_KIND_BV_XNOR -> expr.convert { arg0: KExpr<KBoolSort>, arg1: KExpr<KBoolSort> ->
            mkXor(arg0, arg1).not()
        }
        else -> error("unexpected bool kind $kind")
    }

    @Suppress("LongMethod", "ComplexMethod")
    open fun KContext.convertBVExpr(expr: BitwuzlaTerm, kind: BitwuzlaKind): ExprConversionResult = when (kind) {
        BitwuzlaKind.BITWUZLA_KIND_BV_AND -> expr.convertBv(::mkBvAndExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_NAND -> expr.convertBv(::mkBvNAndExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_NEG -> expr.convertBv(::mkBvNegationExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_NOR -> expr.convertBv(::mkBvNorExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_NOT -> expr.convertBv(::mkBvNotExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_OR -> expr.convertBv(::mkBvOrExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_XNOR -> expr.convertBv(::mkBvXNorExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_XOR -> expr.convertBv(::mkBvXorExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_REDAND -> expr.convertBv(::mkBvReductionAndExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_REDOR -> expr.convertBv(::mkBvReductionOrExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_REDXOR -> TODO("$kind conversion is unsupported yet")
        BitwuzlaKind.BITWUZLA_KIND_BV_SGE -> expr.convertBv(::mkBvSignedGreaterOrEqualExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_SGT -> expr.convertBv(::mkBvSignedGreaterExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_SLE -> expr.convertBv(::mkBvSignedLessOrEqualExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_SLT -> expr.convertBv(::mkBvSignedLessExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_UGE -> expr.convertBv(::mkBvUnsignedGreaterOrEqualExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_UGT -> expr.convertBv(::mkBvUnsignedGreaterExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_ULE -> expr.convertBv(::mkBvUnsignedLessOrEqualExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_ULT -> expr.convertBv(::mkBvUnsignedLessExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_ADD -> expr.convertBv(::mkBvAddExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_SUB -> expr.convertBv(::mkBvSubExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_DEC,
        BitwuzlaKind.BITWUZLA_KIND_BV_INC -> TODO("$kind conversion is unsupported yet")
        BitwuzlaKind.BITWUZLA_KIND_BV_MUL -> expr.convertBv(::mkBvMulExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_SDIV -> expr.convertBv(::mkBvSignedDivExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_SMOD -> expr.convertBv(::mkBvSignedModExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_SREM -> expr.convertBv(::mkBvSignedRemExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_UDIV -> expr.convertBv(::mkBvUnsignedDivExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_UREM -> expr.convertBv(::mkBvUnsignedRemExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_SADD_OVERFLOW -> expr.convertBv { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> ->
            mkBvAddNoOverflowExpr(arg0, arg1, isSigned = true)
        }
        BitwuzlaKind.BITWUZLA_KIND_BV_UADD_OVERFLOW -> expr.convertBv { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> ->
            mkBvAddNoOverflowExpr(arg0, arg1, isSigned = false)
        }
        BitwuzlaKind.BITWUZLA_KIND_BV_SSUB_OVERFLOW -> expr.convertBv { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> ->
            mkBvSubNoOverflowExpr(arg0, arg1)
        }
        BitwuzlaKind.BITWUZLA_KIND_BV_USUB_OVERFLOW -> TODO("$kind")
        BitwuzlaKind.BITWUZLA_KIND_BV_SMUL_OVERFLOW -> expr.convertBv { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> ->
            mkBvMulNoOverflowExpr(arg0, arg1, isSigned = true)
        }
        BitwuzlaKind.BITWUZLA_KIND_BV_UMUL_OVERFLOW -> expr.convertBv { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> ->
            mkBvMulNoOverflowExpr(arg0, arg1, isSigned = false)
        }
        BitwuzlaKind.BITWUZLA_KIND_BV_SDIV_OVERFLOW -> expr.convertBv { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> ->
            mkBvDivNoOverflowExpr(arg0, arg1)
        }
        BitwuzlaKind.BITWUZLA_KIND_BV_ROL -> expr.convertBv(::mkBvRotateLeftExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_ROR -> expr.convertBv(::mkBvRotateRightExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_ASHR -> expr.convertBv(::mkBvArithShiftRightExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_SHR -> expr.convertBv(::mkBvLogicalShiftRightExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_SHL -> expr.convertBv(::mkBvShiftLeftExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_ROLI -> expr.convertBv { value: KExpr<KBvSort> ->
            val indices = Native.bitwuzlaTermGetIndices(expr)
            val i = indices.single()
            mkBvRotateLeftIndexedExpr(i, value)
        }
        BitwuzlaKind.BITWUZLA_KIND_BV_RORI -> expr.convertBv { value: KExpr<KBvSort> ->
            val indices = Native.bitwuzlaTermGetIndices(expr)
            val i = indices.single()
            mkBvRotateRightIndexedExpr(i, value)
        }
        BitwuzlaKind.BITWUZLA_KIND_BV_SIGN_EXTEND -> expr.convertBv { value: KExpr<KBvSort> ->
            val indices = Native.bitwuzlaTermGetIndices(expr)
            val i = indices.single()
            mkBvSignExtensionExpr(i, value)
        }
        BitwuzlaKind.BITWUZLA_KIND_BV_ZERO_EXTEND -> expr.convertBv { value: KExpr<KBvSort> ->
            val indices = Native.bitwuzlaTermGetIndices(expr)
            val i = indices.single()
            mkBvZeroExtensionExpr(i, value)
        }
        BitwuzlaKind.BITWUZLA_KIND_BV_REPEAT -> expr.convertBv { value: KExpr<KBvSort> ->
            val indices = Native.bitwuzlaTermGetIndices(expr)
            val i = indices.single()
            mkBvRepeatExpr(i, value)
        }
        BitwuzlaKind.BITWUZLA_KIND_BV_EXTRACT -> expr.convertBv { value: KExpr<KBvSort> ->
            val indices = Native.bitwuzlaTermGetIndices(expr)
            check(indices.size == 2) { "unexpected extract indices: $indices" }
            val (high, low) = indices
            mkBvExtractExpr(high, low, value)
        }
        BitwuzlaKind.BITWUZLA_KIND_BV_CONCAT -> expr.convertBv(::mkBvConcatExpr)
        BitwuzlaKind.BITWUZLA_KIND_BV_COMP -> TODO("$kind")
        else -> error("unexpected BV kind $kind")
    }

    @Suppress("LongMethod", "ComplexMethod")
    open fun convertFpExpr(expr: BitwuzlaTerm, kind: BitwuzlaKind): ExprConversionResult = when (kind) {
        BitwuzlaKind.BITWUZLA_KIND_FP_ABS -> expr.convert(ctx::mkFpAbsExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_ADD -> expr.convert(ctx::mkFpAddExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_SUB -> expr.convert(ctx::mkFpSubExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_MUL -> expr.convert(ctx::mkFpMulExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_FMA -> expr.convert(ctx::mkFpFusedMulAddExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_DIV -> expr.convert(ctx::mkFpDivExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_REM -> expr.convert(ctx::mkFpRemExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_MAX -> expr.convert(ctx::mkFpMaxExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_MIN -> expr.convert(ctx::mkFpMinExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_NEG -> expr.convert(ctx::mkFpNegationExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_RTI -> expr.convert(ctx::mkFpRoundToIntegralExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_SQRT -> expr.convert(ctx::mkFpSqrtExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_IS_INF -> expr.convert(ctx::mkFpIsInfiniteExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_IS_NAN -> expr.convert(ctx::mkFpIsNaNExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_IS_NORMAL -> expr.convert(ctx::mkFpIsNormalExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_IS_SUBNORMAL -> expr.convert(ctx::mkFpIsSubnormalExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_IS_NEG -> expr.convert(ctx::mkFpIsNegativeExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_IS_POS -> expr.convert(ctx::mkFpIsPositiveExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_IS_ZERO -> expr.convert(ctx::mkFpIsZeroExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_EQ -> expr.convert(ctx::mkFpEqualExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_LEQ -> expr.convert(ctx::mkFpLessOrEqualExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_LT -> expr.convert(ctx::mkFpLessExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_GEQ -> expr.convert(ctx::mkFpGreaterOrEqualExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_GT -> expr.convert(ctx::mkFpGreaterExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_TO_SBV ->
            expr.convert { rm: KExpr<KFpRoundingModeSort>, value: KExpr<KFpSort> ->
                val bvSize = Native.bitwuzlaTermGetIndices(expr).single()
                ctx.mkFpToBvExpr(rm, value, bvSize, isSigned = true)
            }
        BitwuzlaKind.BITWUZLA_KIND_FP_TO_UBV ->
            expr.convert { rm: KExpr<KFpRoundingModeSort>, value: KExpr<KFpSort> ->
                val bvSize = Native.bitwuzlaTermGetIndices(expr).single()
                ctx.mkFpToBvExpr(rm, value, bvSize, isSigned = false)
            }
        BitwuzlaKind.BITWUZLA_KIND_FP_TO_FP_FROM_SBV ->
            expr.convert { rm: KExpr<KFpRoundingModeSort>, value: KExpr<KBvSort> ->
                val sort = Native.bitwuzlaTermGetSort(expr).convertSort() as KFpSort
                ctx.mkBvToFpExpr(sort, rm, value, signed = true)
            }
        BitwuzlaKind.BITWUZLA_KIND_FP_TO_FP_FROM_UBV ->
            expr.convert { rm: KExpr<KFpRoundingModeSort>, value: KExpr<KBvSort> ->
                val sort = Native.bitwuzlaTermGetSort(expr).convertSort() as KFpSort
                ctx.mkBvToFpExpr(sort, rm, value, signed = false)
            }
        BitwuzlaKind.BITWUZLA_KIND_FP_FP -> expr.convert(ctx::mkFpFromBvExpr)
        BitwuzlaKind.BITWUZLA_KIND_FP_TO_FP_FROM_BV ->
            expr.convert { bv: KExpr<KBvSort> ->
                with(ctx) {
                    val indices = Native.bitwuzlaTermGetIndices(expr)
                    check(indices.size == 2) { "unexpected fp-from-bv indices: $indices" }
                    val exponentSize = indices.first()
                    val size = bv.sort.sizeBits.toInt()

                    val sign = mkBvExtractExpr(size - 1, size - 1, bv)
                    val exponent = mkBvExtractExpr(size - 2, size - exponentSize - 1, bv)
                    val significand = mkBvExtractExpr(size - exponentSize - 2, 0, bv)

                    mkFpFromBvExpr(sign.uncheckedCast(), exponent, significand)
                }
            }
        BitwuzlaKind.BITWUZLA_KIND_FP_TO_FP_FROM_FP ->
            expr.convert { rm: KExpr<KFpRoundingModeSort>, value: KExpr<KFpSort> ->
                val sort = Native.bitwuzlaTermGetSort(expr).convertSort() as KFpSort
                ctx.mkFpToFpExpr(sort, rm, value)
            }
        else -> error("unexpected Fp kind $kind")
    }

    fun convertQuantifier(expr: BitwuzlaTerm, kind: BitwuzlaKind): ExprConversionResult = convert {
        val children = Native.bitwuzlaTermGetChildren(expr)
        val boundVars = children.dropLast(1)
        val body = children.last()

        val nestedScope = scopedVars?.toMutableMap() ?: hashMapOf()

        val convertedBounds = boundVars.map { boundVar ->
            val decl = bitwuzlaCtx.findConvertedVar(boundVar) ?: run {
                val sort = Native.bitwuzlaTermGetSort(boundVar)
                val name = Native.bitwuzlaTermGetSymbol(boundVar)
                ctx.mkFreshConstDecl(name ?: "var", sort.convertSort())
            }
            nestedScope[boundVar] = decl
            decl
        }

        val bodyConverter = KBitwuzlaExprConverter(ctx, bitwuzlaCtx, nestedScope)

        when (kind) {
            BitwuzlaKind.BITWUZLA_KIND_FORALL,
            BitwuzlaKind.BITWUZLA_KIND_EXISTS -> {
                val convertedBody = with(bodyConverter) { body.convertExpr(ctx.boolSort) }
                when (kind) {
                    BitwuzlaKind.BITWUZLA_KIND_FORALL -> ctx.mkUniversalQuantifier(convertedBody, convertedBounds)
                    BitwuzlaKind.BITWUZLA_KIND_EXISTS -> ctx.mkExistentialQuantifier(convertedBody, convertedBounds)
                    else -> error("impossible: when is exhaustive")
                }
            }

            BitwuzlaKind.BITWUZLA_KIND_LAMBDA -> {
                val convertedBody = bodyConverter.convertFromNative<KSort>(body)
                ctx.convertArrayLambdaSimplified(convertedBounds, convertedBody)
            }

            else -> error("Unexpected quantifier: $kind")
        }
    }

    private fun KContext.convertArrayLambdaSimplified(
        bounds: List<KDecl<*>>,
        body: KExpr<*>
    ): KExpr<*> {
        if (body is KInterpretedValue<*>) {
            val sort = mkAnyArraySort(bounds.map { it.sort }, body.sort)
            return mkArrayConst(sort, body.uncheckedCast())
        }

        if (body is KIteExpr<*>) {
            tryRecognizeArrayStore(body, bounds)?.let { return it }
        }

        return mkAnyArrayLambda(bounds, body)
    }

    /**
     * In a case of multidimensional array store expressions in model are represented as:
     * (lambda (i0 ... in) (ite (and (= i0 c0) .. (= in cn)) value (select array i0...in)))
     * We try to recognize this pattern and rewrite such lambda expressions as normal
     * array stores.
     * */
    private fun KContext.tryRecognizeArrayStore(
        body: KIteExpr<*>,
        bounds: List<KDecl<*>>
    ): KExpr<*>? {
        val storedValue = body.trueBranch as? KInterpretedValue<*> ?: return null
        val conditionArgs = (body.condition as? KAndExpr)?.args ?: return null

        val boundConsts = bounds.map { it.apply(emptyList()) }

        val indexBindings = conditionArgs.associate {
            val binding = it as? KEqExpr<*> ?: return null
            val lhs = binding.lhs
            val rhs = binding.rhs

            when {
                lhs is KInterpretedValue<*> && rhs is KConst<*> -> rhs to lhs
                lhs is KConst<*> && rhs is KInterpretedValue<*> -> lhs to rhs
                else -> return null
            }
        }

        val indices = boundConsts.map { indexBindings[it] ?: return null }

        val nestedValue = body.falseBranch
        if (nestedValue is KArraySelectBase<*, *>) {
            if (boundConsts != nestedValue.indices) return null

            val nestedArray = nestedValue.array
            return mkAnyArrayStore(nestedArray, indices.uncheckedCast(), storedValue.uncheckedCast())
        }

        return null
    }

    fun convertEqExpr(lhs: KExpr<KSort>, rhs: KExpr<KSort>): KExpr<KBoolSort> = with(ctx) {
        val expectedSort = selectExpectedSort(lhs.sort, rhs.sort)
        return mkEq(
            lhs.convertToExpectedIfNeeded(expectedSort),
            rhs.convertToExpectedIfNeeded(expectedSort)
        )
    }

    fun convertDistinctExpr(args: List<KExpr<KSort>>): KExpr<KBoolSort> = with(ctx) {
        val expectedSort = selectExpectedSort(args.map { it.sort })
        val normalizedArgs = args.map { it.convertToExpectedIfNeeded(expectedSort) }
        return mkDistinct(normalizedArgs)
    }

    fun convertIteExpr(
        condition: KExpr<KBoolSort>,
        trueBranch: KExpr<KSort>,
        falseBranch: KExpr<KSort>
    ): KExpr<KSort> = with(ctx) {
        val expectedSort = selectExpectedSort(trueBranch.sort, falseBranch.sort)
        return mkIte(
            condition,
            trueBranch.convertToExpectedIfNeeded(expectedSort),
            falseBranch.convertToExpectedIfNeeded(expectedSort)
        )
    }

    fun convertConstArrayExpr(
        currentSort: KArraySort<KSort, KSort>,
        value: KExpr<KSort>
    ): KArrayConst<KArraySort<KSort, KSort>, KSort> = with(ctx) {
        val expectedValueSort = selectExpectedSort(currentSort.range, value.sort)
        val expectedArraySort = mkArraySort(currentSort.domain, expectedValueSort)
        return mkArrayConst(expectedArraySort, value.convertToExpectedIfNeeded(expectedValueSort))
    }

    fun convertArraySelectExpr(
        array: KExpr<KArraySort<KSort, KSort>>,
        index: KExpr<KSort>
    ): KExpr<KSort> = ctx.mkAnyArraySelect(array, listOf(index))

    fun convertArrayStoreExpr(
        array: KExpr<KArraySort<KSort, KSort>>,
        index: KExpr<KSort>,
        value: KExpr<KSort>
    ): KExpr<KArraySort<KSort, KSort>> = ctx.mkAnyArrayStore(array, listOf(index), value)

    private fun <A : KArraySortBase<*>> KContext.mkAnyArrayStore(
        array: KExpr<A>,
        indices: List<KExpr<KSort>>,
        value: KExpr<KSort>
    ): KExpr<A> {
        val expectedValueSort = selectExpectedSort(array.sort.range, value.sort)
        val wellSortedValue = value.convertToExpectedIfNeeded(expectedValueSort)
        return mkAnyArrayOperation(
            array, expectedValueSort, indices,
            { a, d0 -> mkArrayStore(a, d0, wellSortedValue) },
            { a, d0, d1 -> mkArrayStore(a, d0, d1, wellSortedValue) },
            { a, d0, d1, d2 -> mkArrayStore(a, d0, d1, d2, wellSortedValue) },
            { a, domain -> mkArrayNStore(a, domain, wellSortedValue) }
        ).uncheckedCast()
    }

    fun <A : KArraySortBase<*>> KContext.mkAnyArraySelect(
        array: KExpr<A>,
        indices: List<KExpr<KSort>>
    ): KExpr<KSort> = mkAnyArrayOperation(
        array, array.sort.range, indices,
        { a, d0 -> mkArraySelect(a, d0) },
        { a, d0, d1 -> mkArraySelect(a, d0, d1) },
        { a, d0, d1, d2 -> mkArraySelect(a, d0, d1, d2) },
        { a, domain -> mkArrayNSelect(a, domain) }
    )

    @Suppress("LongParameterList")
    private inline fun <A : KArraySortBase<*>, R> KContext.mkAnyArrayOperation(
        array: KExpr<A>,
        expectedArrayRange: KSort,
        indices: List<KExpr<KSort>>,
        array1: (KExpr<KArraySort<KSort, KSort>>, KExpr<KSort>) -> R,
        array2: (KExpr<KArray2Sort<KSort, KSort, KSort>>, KExpr<KSort>, KExpr<KSort>) -> R,
        array3: (KExpr<KArray3Sort<KSort, KSort, KSort, KSort>>, KExpr<KSort>, KExpr<KSort>, KExpr<KSort>) -> R,
        arrayN: (KExpr<KArrayNSort<KSort>>, List<KExpr<KSort>>) -> R
    ): R {
        val expectedIndicesSorts = array.sort.domainSorts.zip(indices) { domainSort, index ->
            selectExpectedSort(domainSort, index.sort)
        }
        val expectedArraySort = mkAnyArraySort(expectedIndicesSorts, expectedArrayRange)

        val wellSortedArray = array.convertToExpectedIfNeeded(expectedArraySort)
        val wellSortedIndices = indices.zip(expectedIndicesSorts) { index, expectedSort ->
            index.convertToExpectedIfNeeded(expectedSort)
        }

        return mkAnyArrayOperation(
            wellSortedIndices,
            { d0 -> array1(wellSortedArray.uncheckedCast(), d0) },
            { d0, d1 -> array2(wellSortedArray.uncheckedCast(), d0, d1) },
            { d0, d1, d2 -> array3(wellSortedArray.uncheckedCast(), d0, d1, d2) },
            { arrayN(wellSortedArray.uncheckedCast(), it) }
        )
    }

    private fun KContext.mkAnyArrayLambda(domain: List<KDecl<*>>, body: KExpr<*>) =
        mkAnyArrayOperation(
            domain,
            { d0 -> mkArrayLambda(d0, body) },
            { d0, d1 -> mkArrayLambda(d0, d1, body) },
            { d0, d1, d2 -> mkArrayLambda(d0, d1, d2, body) },
            { mkArrayNLambda(it, body) }
        )

    fun KContext.mkAnyArraySort(domain: List<KSort>, range: KSort): KArraySortBase<KSort> =
        mkAnyArrayOperation(
            domain,
            { d0 -> mkArraySort(d0, range) },
            { d0, d1 -> mkArraySort(d0, d1, range) },
            { d0, d1, d2 -> mkArraySort(d0, d1, d2, range) },
            { mkArrayNSort(it, range) }
        )

    private inline fun <T, R> mkAnyArrayOperation(
        domain: List<T>,
        array1: (T) -> R,
        array2: (T, T) -> R,
        array3: (T, T, T) -> R,
        arrayN: (List<T>) -> R
    ): R = when (domain.size) {
        KArraySort.DOMAIN_SIZE -> array1(domain.single())
        KArray2Sort.DOMAIN_SIZE -> array2(domain.first(), domain.last())
        KArray3Sort.DOMAIN_SIZE -> {
            val (d0, d1, d2) = domain
            array3(d0, d1, d2)
        }

        else -> arrayN(domain)
    }

    fun selectExpectedSort(lhs: KSort, rhs: KSort): KSort =
        if (rhs is KUninterpretedSort) rhs else lhs

    fun selectExpectedSort(sorts: List<KSort>): KSort =
        sorts.firstOrNull { it is KUninterpretedSort } ?: sorts.first()

    private fun <T : KDecl<*>> generateDecl(term: BitwuzlaTerm, generator: (String) -> T): T {
        val name = Native.bitwuzlaTermGetSymbol(term)
        val declName = name ?: generateBitwuzlaSymbol(term)
        return generator(declName)
    }

    private fun generateBitwuzlaSymbol(expr: BitwuzlaTerm): String {
        /* generate symbol in the same way as in bitwuzla model printer
        * https://github.com/bitwuzla/bitwuzla/blob/main/src/bzlaprintmodel.c#L263
        * */
        val id = Native.bitwuzlaTermHash(expr)
        return "uf$id"
    }

    /**
     * Bitwuzla does not distinguish between Bool and (BitVec 1).
     *
     *  By default, we convert all Bitwuzla (BitVec 1) terms as Bool expressions, but:
     *  1. user defined constant with (BitVec 1) sort may appear
     *  2. user defined constant with (Array X (BitVec 1)) sort may appear
     *  3. user defined function with (BitVec 1) in domain or range may appear
     *
     *  Such user defined constants may be used in expressions, where we expect equal sorts.
     *  For example, `x: (BitVec 1) == y: Bool` or `x: (Array T (BitVec 1)) == y: (Array T Bool)`.
     *  For such reason, we introduce additional expressions
     *  to convert from (BitVec 1) to Bool and vice versa.
     *
     * @see ensureBoolExpr
     * @see ensureBv1Expr
     * @see ensureArrayExprSortMatch
     * */
    @Suppress("UNCHECKED_CAST")
    fun KExpr<*>.convertToBoolIfNeeded(): KExpr<*> = when (sort) {
        ctx.bv1Sort -> ensureBoolExpr()
        is KArraySort<*, *> -> (this as KExpr<KArraySort<*, *>>)
            .ensureArrayExprSortMatch(
                domainExpected = { domain -> domain.map { if (it == ctx.bv1Sort) ctx.boolSort else it } },
                rangeExpected = { if (it == ctx.bv1Sort) ctx.boolSort else it }
            )
        else -> this
    }

    /**
     * Convert expression to expected sort.
     *
     *  Mainly used for convert from Bool to (BitVec 1) or (BitVec 32) to Uninterpreted Sort:
     *  1. In function app, when argument sort doesn't match
     *  2. When top level expression expectedSort doesn't match ([convertExpr])
     *
     *  Also works for Arrays.
     *
     * @see convertToBoolIfNeeded
     * */
    fun <T : KSort> KExpr<*>.convertToExpectedIfNeeded(expected: T): KExpr<T> = when (expected) {
        ctx.bv1Sort -> ensureBv1Expr().uncheckedCast()
        ctx.boolSort -> ensureBoolExpr().uncheckedCast()
        is KArraySortBase<*> -> {
            val array: KExpr<KArraySortBase<*>> = this.uncheckedCast()
            array.ensureArrayExprSortMatch(
                domainExpected = { expected.domainSorts },
                rangeExpected = { expected.range }
            ).uncheckedCast()
        }
        is KUninterpretedSort -> ensureUninterpretedSortExpr(expected).uncheckedCast()
        else -> {
            check(this.sort !is KUninterpretedSort) {
                "Unexpected cast from ${this.sort} to $expected"
            }
            this.uncheckedCast()
        }
    }

    fun KExpr<*>.ensureBvExpr(): KExpr<KBvSort> = with(ctx) {
        val expr = if (sort == boolSort) {
            ensureBv1Expr()
        } else {
            check(sort is KBvSort) { "Bv sort expected but $sort occurred" }
            this@ensureBvExpr
        }

        expr.uncheckedCast()
    }

    private val bv1One: KExpr<KBv1Sort> by lazy { ctx.mkBv(true) }
    private val bv1Zero: KExpr<KBv1Sort> by lazy { ctx.mkBv(false) }

    /**
     * Convert expression from (Array A B) to (Array X Y),
     * where A,B,X,Y are Bool or (BitVec 1), (BitVec 32) or Uninterpreted
     * */
    private inline fun <A : KArraySortBase<*>> KExpr<A>.ensureArrayExprSortMatch(
        domainExpected: (List<KSort>) -> List<KSort>,
        rangeExpected: (KSort) -> KSort
    ): KExpr<*> = with(ctx) {
        val expectedDomain = domainExpected(sort.domainSorts)
        val expectedRange = rangeExpected(sort.range)

        when {
            expectedDomain == sort.domainSorts && expectedRange == sort.range -> this@ensureArrayExprSortMatch
            this@ensureArrayExprSortMatch is ArrayAdapterExpr<*, *>
                    && arg.sort.domainSorts == expectedDomain
                    && arg.sort.range == expectedRange -> arg
            else -> {
                val expectedSort = mkAnyArraySort(expectedDomain, expectedRange)
                ArrayAdapterExpr(this@ensureArrayExprSortMatch, expectedSort)
            }
        }
    }

    /**
     * Convert expression from (BitVec 1) to Bool.
     * */
    private fun KExpr<*>.ensureBoolExpr(): KExpr<KBoolSort> = with(ctx) {
        when {
            sort == boolSort -> this@ensureBoolExpr.uncheckedCast()
            this@ensureBoolExpr is BoolToBv1AdapterExpr -> arg
            this@ensureBoolExpr == bv1One -> trueExpr
            this@ensureBoolExpr == bv1Zero -> falseExpr
            else -> Bv1ToBoolAdapterExpr(this@ensureBoolExpr.uncheckedCast())
        }
    }

    /**
     * Convert expression from Bool to (BitVec 1).
     * */
    private fun KExpr<*>.ensureBv1Expr(): KExpr<KBv1Sort> = with(ctx) {
        when {
            sort == bv1Sort -> this@ensureBv1Expr.uncheckedCast()
            this@ensureBv1Expr is Bv1ToBoolAdapterExpr -> arg
            this@ensureBv1Expr == trueExpr -> bv1One
            this@ensureBv1Expr == falseExpr -> bv1Zero
            else -> BoolToBv1AdapterExpr(this@ensureBv1Expr.uncheckedCast())
        }
    }

    /**
     * Convert expression from (BitVec 32) to uninterpreted sort.
     * */
    private fun KExpr<*>.ensureUninterpretedSortExpr(expected: KUninterpretedSort): KExpr<*> {
        if (sort == expected) return this

        if (this !is KBitVec32Value) {
            throw KSolverUnsupportedFeatureException(
                "Expression $this it too complex for conversion to $expected sort"
            )
        }

        // Use adapter term to keep caches consistent
        return Bv32ToUninterpretedSortAdapterExpr(this, expected)
    }

    private inner class BoolToBv1AdapterExpr(val arg: KExpr<KBoolSort>) : KExpr<KBv1Sort>(ctx) {
        override val sort: KBv1Sort = ctx.bv1Sort

        override fun print(printer: ExpressionPrinter) = with(printer) {
            append("(toBV1 ")
            append(arg)
            append(")")
        }

        override fun accept(transformer: KTransformerBase): KExpr<KBv1Sort> {
            check(transformer is AdapterTermRewriter) { "leaked adapter term" }
            return transformer.transform(this)
        }

        override fun internHashCode(): Int = hash(arg)
        override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg })
    }

    private inner class Bv1ToBoolAdapterExpr(val arg: KExpr<KBv1Sort>) : KExpr<KBoolSort>(ctx) {
        override val sort: KBoolSort = ctx.boolSort

        override fun print(printer: ExpressionPrinter) = with(printer) {
            append("(toBool ")
            append(arg)
            append(")")
        }

        override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> {
            check(transformer is AdapterTermRewriter) { "leaked adapter term" }
            return transformer.transform(this)
        }

        override fun internHashCode(): Int = hash(arg)
        override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg })
    }

    private inner class Bv32ToUninterpretedSortAdapterExpr(
        val arg: KBitVec32Value,
        override val sort: KUninterpretedSort
    ) : KExpr<KUninterpretedSort>(ctx) {
        override fun print(printer: ExpressionPrinter) = with(printer) {
            append("(to $sort ")
            append(arg)
            append(")")
        }

        override fun internHashCode(): Int = hash(arg, sort)
        override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg }, { sort })

        override fun accept(transformer: KTransformerBase): KExpr<KUninterpretedSort> {
            check(transformer is AdapterTermRewriter) { "leaked adapter term" }
            return transformer.transform(this)
        }
    }

    private inner class ArrayAdapterExpr<FromSort : KArraySortBase<*>, ToSort : KArraySortBase<*>>(
        val arg: KExpr<FromSort>,
        override val sort: ToSort
    ) : KExpr<ToSort>(ctx) {
        override fun print(printer: ExpressionPrinter) = with(printer) {
            append("(toArray ")
            append("$sort")
            append(" ")
            append(arg)
            append(")")
        }

        override fun accept(transformer: KTransformerBase): KExpr<ToSort> {
            check(transformer is AdapterTermRewriter) { "leaked adapter term" }
            return transformer.transform(this)
        }

        override fun internHashCode(): Int = hash(arg, sort)
        override fun internEquals(other: Any): Boolean =
            structurallyEqual(other, { arg }, { sort })
    }

    /**
     * Remove auxiliary terms introduced by [convertToBoolIfNeeded] and [convertToExpectedIfNeeded].
     * */
    private inner class AdapterTermRewriter(ctx: KContext) : KNonRecursiveTransformer(ctx) {
        // We can skip values transformation since values may not contain any adapter terms
        override fun <T : KSort> exprTransformationRequired(expr: KExpr<T>): Boolean =
            expr !is KInterpretedValue<T>

        /**
         * x: Bool
         * (toBv x) -> (ite x #b1 #b0)
         * */
        fun transform(expr: BoolToBv1AdapterExpr): KExpr<KBv1Sort> = with(ctx) {
            transformExprAfterTransformed(expr, expr.arg) { transformedArg ->
                when (transformedArg) {
                    trueExpr -> bv1Sort.trueValue()
                    falseExpr -> bv1Sort.falseValue()
                    else -> mkIte(transformedArg, bv1Sort.trueValue(), bv1Sort.falseValue())
                }
            }
        }

        /**
         * x: (BitVec 1)
         * (toBool x) -> (ite (x == #b1) true false)
         * */
        fun transform(expr: Bv1ToBoolAdapterExpr): KExpr<KBoolSort> = with(ctx) {
            transformExprAfterTransformed(expr, expr.arg) { transformedArg ->
                when (transformedArg) {
                    bv1Sort.trueValue() -> trueExpr
                    bv1Sort.falseValue() -> falseExpr
                    else -> mkIte(transformedArg eq bv1Sort.trueValue(), trueExpr, falseExpr)
                }
            }
        }

        /**
         * Replace (BitVec 32) value with an uninterpreted constant.
         * */
        fun transform(expr: Bv32ToUninterpretedSortAdapterExpr): KExpr<KUninterpretedSort> {
            val valueContext = uninterpretedSortValueContext
                ?: error("Uninterpreted sort value context is required to convert expr with ${expr.sort} sort")
            return valueContext.mkValue(expr.sort, expr.arg)
        }

        /**
         * x: (Array A B) -> (Array X Y)
         *
         * Resolve sort mismatch between two arrays.
         * For example,
         * ```
         * val x: (Array T Bool)
         * val y: (Array T (BitVec 1))
         * val expr = x eq y
         * ```
         * In this example, we need to convert y from `(Array T (BitVec 1))`
         * to `(Array T Bool)` because we need sort compatibility between x and y.
         * In general case the only way is to generate new array z such that
         * ```
         * convert: ((BitVec 1)) -> Bool
         * z: (Array T Bool)
         * (forall (i: T) (select z i) == convert(select y i))
         * ```
         * This array generation procedure can be represented as [io.ksmt.expr.KArrayLambda]
         * */
        fun <FromSort : KArraySortBase<*>, ToSort : KArraySortBase<*>> transform(
            expr: ArrayAdapterExpr<FromSort, ToSort>
        ): KExpr<ToSort> = with(ctx) {
            val fromSort = expr.arg.sort
            val toSort = expr.sort

            if (fromSort == toSort) {
                return@with expr.arg.uncheckedCast()
            }

            val indices = toSort.domainSorts.map {
                mkFreshConst("i", it)
            }
            val fromIndices = indices.zip(fromSort.domainSorts) { idx, sort ->
                idx.convertToExpectedIfNeeded(sort)
            }

            val value = mkAnyArraySelect(expr.arg, fromIndices)
            val toValue = value.convertToExpectedIfNeeded(toSort.range)

            val replacement: KExpr<ToSort> = mkAnyArrayLambda(
                indices.map { it.decl }, toValue
            ).uncheckedCast()

            AdapterTermRewriter(ctx).apply(replacement)
        }

        private fun <T : KSort> T.trueValue(): KExpr<T> = when (this) {
            is KBv1Sort -> bv1One.uncheckedCast()
            is KBoolSort -> ctx.trueExpr.uncheckedCast()
            else -> error("unexpected sort: $this")
        }

        private fun <T : KSort> T.falseValue(): KExpr<T> = when (this) {
            is KBv1Sort -> bv1Zero.uncheckedCast()
            is KBoolSort -> ctx.falseExpr.uncheckedCast()
            else -> error("unexpected sort: $this")
        }
    }

    inline fun <T : KSort> BitwuzlaTerm.convertBv(
        op: (KExpr<KBvSort>, KExpr<KBvSort>) -> KExpr<T>
    ) = convert { a0: KExpr<KSort>, a1: KExpr<KSort> ->
        op(a0.ensureBvExpr(), a1.ensureBvExpr()).convertToBoolIfNeeded()
    }

    inline fun <T : KSort> BitwuzlaTerm.convertBv(
        op: (KExpr<KBvSort>) -> KExpr<T>
    ) = convert { arg: KExpr<KSort> ->
        op(arg.ensureBvExpr()).convertToBoolIfNeeded()
    }

    inline fun <T : KSort, A0 : KSort> BitwuzlaTerm.convert(
        op: (KExpr<A0>) -> KExpr<T>
    ): ExprConversionResult = convert(getTermChildren(this), op)

    inline fun <T : KSort, A0 : KSort, A1 : KSort> BitwuzlaTerm.convert(
        op: (KExpr<A0>, KExpr<A1>) -> KExpr<T>
    ): ExprConversionResult = convert(getTermChildren(this), op)

    inline fun <T : KSort, A0 : KSort, A1 : KSort, A2 : KSort> BitwuzlaTerm.convert(
        op: (KExpr<A0>, KExpr<A1>, KExpr<A2>) -> KExpr<T>
    ): ExprConversionResult = convert(getTermChildren(this), op)

    inline fun <T : KSort, A0 : KSort, A1 : KSort, A2 : KSort, A3 : KSort> BitwuzlaTerm.convert(
        op: (KExpr<A0>, KExpr<A1>, KExpr<A2>, KExpr<A3>) -> KExpr<T>
    ): ExprConversionResult = convert(getTermChildren(this), op)

    inline fun <T : KSort, A : KSort> BitwuzlaTerm.convertList(
        op: (List<KExpr<A>>) -> KExpr<T>
    ): ExprConversionResult = convertList(getTermChildren(this), op)

    fun getTermChildren(term: BitwuzlaTerm): LongArray =
        Native.bitwuzlaTermGetChildren(term)
}
