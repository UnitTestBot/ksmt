package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KExpr
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.expr.transformer.KTransformer
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaKind
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaSort
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.solver.util.KExprConverterBase
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KSort

open class KBitwuzlaExprConverter(
    private val ctx: KContext,
    val bitwuzlaCtx: KBitwuzlaContext
) : KExprConverterBase<BitwuzlaTerm>() {

    private val adapterTermRewriter = AdapterTermRewriter(ctx)
    private val incompleteDeclarations = mutableSetOf<KDecl<*>>()

    /** New declarations introduced by Bitwuzla to return correct expressions.
     *
     *  For example, when converting array with partial interpretation,
     *  default value will be represented with new unnamed declaration.
     *
     *  @see generateDecl
     * */
    val incompleteDecls: Set<KDecl<*>>
        get() = incompleteDeclarations

    /**
     * Create KSmt expression from Bitwuzla term.
     * @param expectedSort expected sort of resulting expression
     *
     * @see convertToExpectedIfNeeded
     * @see convertToBoolIfNeeded
     * */
    fun <T : KSort> BitwuzlaTerm.convertExpr(expectedSort: T): KExpr<T> =
        convertFromNative<KSort>()
            .convertToExpectedIfNeeded(expectedSort)
            .let { adapterTermRewriter.apply(it) }


    override fun findConvertedNative(expr: BitwuzlaTerm): KExpr<*>? =
        bitwuzlaCtx.findConvertedExpr(expr)

    override fun saveConvertedNative(native: BitwuzlaTerm, converted: KExpr<*>) {
        bitwuzlaCtx.convertExpr(native) { converted }
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
                error("fun sorts are not allowed for conversion")
            }
            Native.bitwuzlaSortIsBv(sort) -> {
                val size = Native.bitwuzlaSortBvGetSize(sort)
                mkBvSort(size.toUInt())
            }
            else -> TODO("sort is not supported")
        }
    }

    @Suppress("LongMethod", "ComplexMethod")
    override fun convertNativeExpr(expr: BitwuzlaTerm): ExprConversionResult = with(ctx) {
        when (val kind = Native.bitwuzlaTermGetKind(expr)) {
            // constants, functions, values
            BitwuzlaKind.BITWUZLA_KIND_CONST -> convertConst(expr)
            BitwuzlaKind.BITWUZLA_KIND_APPLY -> convertFunctionApp(expr)
            BitwuzlaKind.BITWUZLA_KIND_VAL -> convertValue(expr)
            BitwuzlaKind.BITWUZLA_KIND_VAR -> TODO("var conversion is not implemented")

            // bool
            BitwuzlaKind.BITWUZLA_KIND_IFF,
            BitwuzlaKind.BITWUZLA_KIND_EQUAL -> expr.convert(::mkEq)
            BitwuzlaKind.BITWUZLA_KIND_DISTINCT -> expr.convertList(::mkDistinct)
            BitwuzlaKind.BITWUZLA_KIND_ITE -> expr.convert(::mkIte)
            BitwuzlaKind.BITWUZLA_KIND_IMPLIES -> expr.convert(::mkImplies)

            // array
            BitwuzlaKind.BITWUZLA_KIND_CONST_ARRAY -> expr.convert { value: KExpr<KSort> ->
                val sort = Native.bitwuzlaTermGetSort(expr).convertSort() as KArraySort<*, *>
                mkArrayConst(sort, value)
            }
            BitwuzlaKind.BITWUZLA_KIND_ARRAY_SELECT -> expr.convert(::mkArraySelect)
            BitwuzlaKind.BITWUZLA_KIND_ARRAY_STORE -> expr.convert(::mkArrayStore)

            // quantifiers
            BitwuzlaKind.BITWUZLA_KIND_EXISTS,
            BitwuzlaKind.BITWUZLA_KIND_FORALL -> TODO("quantifier conversion is not implemented")

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
            BitwuzlaKind.BITWUZLA_KIND_FP_TO_UBV -> TODO("FP are not supported yet")

            // unsupported
            BitwuzlaKind.BITWUZLA_NUM_KINDS,
            BitwuzlaKind.BITWUZLA_KIND_LAMBDA -> TODO("unsupported kind $kind")
        }
    }

    private fun KContext.convertFunctionApp(expr: BitwuzlaTerm): ExprConversionResult {
        val children = Native.bitwuzlaTermGetChildren(expr)
        check(children.isNotEmpty()) { "Apply has no function term" }
        val function = children[0]
        val appArgs = children.drop(1).toTypedArray()
        return expr.convertList(appArgs) { convertedArgs: List<KExpr<KSort>> ->
            check(Native.bitwuzlaTermIsFun(function)) { "function term expected" }
            val funcDecl = convertFuncDecl(function)
            val args = convertedArgs.zip(funcDecl.argSorts) { arg, expectedSort ->
                arg.convertToExpectedIfNeeded(expectedSort)
            }
            funcDecl.apply(args).convertToBoolIfNeeded()
        }
    }

    private fun KContext.convertFuncDecl(function: BitwuzlaTerm): KFuncDecl<*> {
        val knownFuncDecl = bitwuzlaCtx.convertConstantIfKnown(function)

        if (knownFuncDecl != null) {
            return knownFuncDecl as? KFuncDecl<*>
                ?: error("function expected. actual: $knownFuncDecl")
        }

        // new function
        val domain = Native.bitwuzlaTermFunGetDomainSorts(function).map { it.convertSort() }
        val range = Native.bitwuzlaTermFunGetCodomainSort(function).convertSort()
        return generateDecl(function) { mkFuncDecl(it, range, domain) }
    }

    private fun KContext.convertConst(expr: BitwuzlaTerm): ExprConversionResult = convert<KSort> {
        val knownConstDecl = bitwuzlaCtx.convertConstantIfKnown(expr)
        if (knownConstDecl != null) {
            @Suppress("UNCHECKED_CAST")
            return@convert mkConstApp(knownConstDecl).convertToBoolIfNeeded() as KExpr<KSort>
        }

        // newly generated constant
        val sort = Native.bitwuzlaTermGetSort(expr)
        if (!Native.bitwuzlaSortIsFun(sort) || Native.bitwuzlaSortIsArray(sort)) {
            val decl = generateDecl(expr) { mkConstDecl(it, sort.convertSort()) }
            @Suppress("UNCHECKED_CAST")
            return@convert decl.apply().convertToBoolIfNeeded() as KExpr<KSort>
        }

        // newly generated functional constant
        error("Constants with functional type are not supported")
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
            Native.bitwuzlaTermIsFp(expr) -> TODO("FP are not supported yet")
            else -> TODO("unsupported value")
        }
    }

    private fun KContext.convertBvValue(expr: BitwuzlaTerm): KBitVecValue<KBvSort> {
        val size = Native.bitwuzlaTermBvGetSize(expr)
        val convertedValue = if (Native.bitwuzlaTermIsBvValue(expr)) {
            // convert Bv value from native representation
            val nativeBits = Native.bitwuzlaBvConstNodeGetBits(expr)
            val nativeBitsSize = Native.bitwuzlaBvBitsGetWidth(nativeBits)

            check(size == nativeBitsSize) { "bv size mismatch" }

            val bits = if (size <= Long.SIZE_BITS) {
                val numericValue = Native.bitwuzlaBvBitsToUInt64(nativeBits).toULong()
                numericValue.toString(radix = 2).padStart(size, '0')
            } else {
                val bitChars = CharArray(size) { charIdx ->
                    val bitIdx = size - 1 - charIdx
                    val bit = Native.bitwuzlaBvBitsGetBit(nativeBits, bitIdx) != 0
                    if (bit) '1' else '0'
                }
                String(bitChars)
            }

            mkBv(bits, size.toUInt())
        } else {
            val value = Native.bitwuzlaGetBvValue(bitwuzlaCtx.bitwuzla, expr)
            mkBv(value, size.toUInt())
        }

        bitwuzlaCtx.saveInternalizedValue(convertedValue, expr)

        return convertedValue
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
        BitwuzlaKind.BITWUZLA_KIND_BV_REDXOR -> TODO("$kind")
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
        BitwuzlaKind.BITWUZLA_KIND_BV_INC -> TODO("$kind")
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

    private fun <T : KDecl<*>> generateDecl(term: BitwuzlaTerm, generator: (String) -> T): T {
        val name = Native.bitwuzlaTermGetSymbol(term)
        val declName = name ?: generateBitwuzlaSymbol(term)
        val decl = generator(declName)
        incompleteDeclarations += decl
        return decl
    }

    private fun generateBitwuzlaSymbol(expr: BitwuzlaTerm): String {
        /* generate symbol in the same way as in bitwuzla model printer
        * https://github.com/bitwuzla/bitwuzla/blob/main/src/bzlaprintmodel.c#L263
        * */
        val id = Native.bitwuzlaTermHash(expr)
        return "uf$id"
    }

    /** Bitwuzla does not distinguish between Bool and (BitVec 1).
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
     *  @see ensureBoolExpr
     *  @see ensureBv1Expr
     *  @see ensureArrayExprSortMatch
     * */
    @Suppress("UNCHECKED_CAST")
    fun KExpr<*>.convertToBoolIfNeeded(): KExpr<*> = when (with(ctx) { sort }) {
        ctx.bv1Sort -> ensureBoolExpr()
        is KArraySort<*, *> -> (this as KExpr<KArraySort<*, *>>)
            .ensureArrayExprSortMatch(
                domainExpected = { if (it == ctx.bv1Sort) ctx.boolSort else it },
                rangeExpected = { if (it == ctx.bv1Sort) ctx.boolSort else it }
            )
        else -> this
    }

    /** Convert expression to expected sort.
     *
     *  Mainly used for convert from Bool to (BitVec 1):
     *  1. in function app, when argument sort doesn't match
     *  2. when top level expression expectedSort doesn't match ([convertExpr])
     *  Also works for Arrays.
     *
     *  @see convertToBoolIfNeeded
     * */
    @Suppress("UNCHECKED_CAST")
    fun <T : KSort> KExpr<*>.convertToExpectedIfNeeded(expected: T): KExpr<T> = when (expected) {
        ctx.bv1Sort -> ensureBv1Expr() as KExpr<T>
        ctx.boolSort -> ensureBoolExpr() as KExpr<T>
        is KArraySort<*, *> -> {
            (this as? KExpr<KArraySort<*, *>> ?: error("array expected. actual is $this"))
                .ensureArrayExprSortMatch(
                    domainExpected = { expected.domain },
                    rangeExpected = { expected.range }
                ) as KExpr<T>
        }
        else -> this as KExpr<T>
    }

    /**
     * Convert expression from (Array A B) to (Array X Y),
     * where A,B,X,Y are Bool or (BitVec 1)
     * */
    private inline fun KExpr<KArraySort<*, *>>.ensureArrayExprSortMatch(
        domainExpected: (KSort) -> KSort,
        rangeExpected: (KSort) -> KSort
    ): KExpr<*> = with(ctx) {
        val expectedDomain = domainExpected(sort.domain)
        val expectedRange = rangeExpected(sort.range)
        when {
            expectedDomain == sort.domain && expectedRange == sort.range -> this@ensureArrayExprSortMatch
            this@ensureArrayExprSortMatch is ArrayAdapterExpr<*, *, *, *>
                    && arg.sort.domain == expectedDomain
                    && arg.sort.range == expectedRange -> arg
            else -> {
                check(expectedDomain !is KArraySort<*, *> && expectedRange !is KArraySort<*, *>) {
                    "Bitwuzla doesn't support nested arrays"
                }
                ArrayAdapterExpr(this@ensureArrayExprSortMatch, expectedDomain, expectedRange)
            }
        }
    }

    /**
     * Convert expression from (BitVec 1) to Bool.
     * */
    @Suppress("UNCHECKED_CAST")
    private fun KExpr<*>.ensureBoolExpr(): KExpr<KBoolSort> = with(ctx) {
        when {
            sort == boolSort -> this@ensureBoolExpr as KExpr<KBoolSort>
            this@ensureBoolExpr is BoolToBv1AdapterExpr -> arg
            else -> Bv1ToBoolAdapterExpr(this@ensureBoolExpr as KExpr<KBv1Sort>)
        }
    }

    /**
     * Convert expression from Bool to (BitVec 1).
     * */
    @Suppress("UNCHECKED_CAST")
    private fun KExpr<*>.ensureBv1Expr(): KExpr<KBv1Sort> = with(ctx) {
        when {
            sort == bv1Sort -> this@ensureBv1Expr as KExpr<KBv1Sort>
            this@ensureBv1Expr is Bv1ToBoolAdapterExpr -> arg
            else -> BoolToBv1AdapterExpr(this@ensureBv1Expr as KExpr<KBoolSort>)
        }
    }

    fun KExpr<*>.ensureBvExpr(): KExpr<KBvSort> = with(ctx) {
        val expr = if (sort == boolSort) {
            ensureBv1Expr()
        } else {
            check(sort is KBvSort) { "Bv sort expected but $sort occurred" }
            this@ensureBvExpr
        }
        @Suppress("UNCHECKED_CAST")
        expr as KExpr<KBvSort>
    }

    private inner class BoolToBv1AdapterExpr(val arg: KExpr<KBoolSort>) : KExpr<KBv1Sort>(ctx) {
        override fun sort(): KBv1Sort = ctx.bv1Sort

        override fun print(builder: StringBuilder) {
            builder.append("(toBV1 ")
            arg.print(builder)
            builder.append(')')
        }

        override fun accept(transformer: KTransformer): KExpr<KBv1Sort> {
            check(transformer is AdapterTermRewriter) { "leaked adapter term" }
            return transformer.transform(this)
        }
    }

    private inner class Bv1ToBoolAdapterExpr(val arg: KExpr<KBv1Sort>) : KExpr<KBoolSort>(ctx) {
        override fun sort(): KBoolSort = ctx.boolSort

        override fun print(builder: StringBuilder) {
            builder.append("(toBool ")
            arg.print(builder)
            builder.append(')')
        }

        override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
            check(transformer is AdapterTermRewriter) { "leaked adapter term" }
            return transformer.transform(this)
        }
    }

    private inner class ArrayAdapterExpr<FromDomain : KSort, FromRange : KSort, ToDomain : KSort, ToRange : KSort>(
        val arg: KExpr<KArraySort<FromDomain, FromRange>>,
        val toDomainSort: ToDomain,
        val toRangeSort: ToRange
    ) : KExpr<KArraySort<ToDomain, ToRange>>(ctx) {
        override fun sort(): KArraySort<ToDomain, ToRange> = ctx.mkArraySort(toDomainSort, toRangeSort)

        override fun print(builder: StringBuilder) {
            builder.append("(toArray ")
            sort().print(builder)
            builder.append(' ')
            arg.print(builder)
            builder.append(')')
        }

        override fun accept(transformer: KTransformer): KExpr<KArraySort<ToDomain, ToRange>> {
            check(transformer is AdapterTermRewriter) { "leaked adapter term" }
            return transformer.transform(this)
        }
    }

    /**
     * Remove auxiliary terms introduced by [convertToBoolIfNeeded] and [convertToExpectedIfNeeded].
     * */
    private inner class AdapterTermRewriter(ctx: KContext) : KNonRecursiveTransformer(ctx) {
        /**
         * x: Bool
         * (toBv x) -> (ite x #b1 #b0)
         * */
        fun transform(expr: BoolToBv1AdapterExpr): KExpr<KBv1Sort> = with(ctx) {
            transformExprAfterTransformed(expr, listOf(expr.arg)) { transformedArg ->
                mkIte(transformedArg.single(), bv1Sort.trueValue(), bv1Sort.falseValue())
            }
        }

        /**
         * x: (BitVec 1)
         * (toBool x) -> (ite (x == #b1) true false)
         * */
        fun transform(expr: Bv1ToBoolAdapterExpr): KExpr<KBoolSort> = with(ctx) {
            transformExprAfterTransformed(expr, listOf(expr.arg)) { transformedArg ->
                mkIte(transformedArg.single() eq bv1Sort.trueValue(), trueExpr, falseExpr)
            }
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
         * This array generation procedure can be represented as [org.ksmt.expr.KArrayLambda]
         * */
        @Suppress("UNCHECKED_CAST")
        fun <FromDomain : KSort, FromRange : KSort, ToDomain : KSort, ToRange : KSort> transform(
            expr: ArrayAdapterExpr<FromDomain, FromRange, ToDomain, ToRange>
        ): KExpr<KArraySort<ToDomain, ToRange>> = with(ctx) {
            val fromSort = expr.arg.sort
            if (fromSort.domain == expr.toDomainSort && fromSort.range == expr.toRangeSort) {
                return@with expr.arg as KExpr<KArraySort<ToDomain, ToRange>>
            }
            val replacement = when (fromSort.domain) {
                bv1Sort, boolSort -> {
                    // avoid lambda expression when possible
                    check(expr.toDomainSort == boolSort || expr.toDomainSort == bv1Sort) {
                        "unexpected cast from ${fromSort.domain} to ${expr.toDomainSort}"
                    }

                    val falseValue = expr.arg.select(fromSort.domain.falseValue())
                        .convertToExpectedIfNeeded(expr.toRangeSort)
                    val trueValue = expr.arg.select(fromSort.domain.trueValue())
                        .convertToExpectedIfNeeded(expr.toRangeSort)

                    val resultArraySort = mkArraySort(expr.toDomainSort, expr.toRangeSort)

                    mkArrayConst(resultArraySort, falseValue).store(expr.toDomainSort.trueValue(), trueValue)
                }
                else -> {
                    check(fromSort.domain == expr.toDomainSort) {
                        "unexpected cast from ${fromSort.domain} to ${expr.toDomainSort}"
                    }

                    val index = expr.toDomainSort.mkFreshConst("index")
                    val bodyExpr = expr.arg.select(index as KExpr<FromDomain>)
                    val body: KExpr<ToRange> = when (expr.toRangeSort) {
                        bv1Sort -> bodyExpr.ensureBv1Expr() as KExpr<ToRange>
                        boolSort -> bodyExpr.ensureBoolExpr() as KExpr<ToRange>
                        else -> error("unexpected domain: ${expr.toRangeSort}")
                    }

                    mkArrayLambda(index.decl, body)
                }
            }
            AdapterTermRewriter(ctx).apply(replacement)
        }

        private val bv1One: KExpr<KBv1Sort> by lazy { ctx.mkBv(true) }
        private val bv1Zero: KExpr<KBv1Sort> by lazy { ctx.mkBv(false) }

        @Suppress("UNCHECKED_CAST")
        private fun <T : KSort> T.trueValue(): KExpr<T> = when (this) {
            is KBv1Sort -> bv1One as KExpr<T>
            is KBoolSort -> ctx.trueExpr as KExpr<T>
            else -> error("unexpected sort: $this")
        }

        @Suppress("UNCHECKED_CAST")
        private fun <T : KSort> T.falseValue(): KExpr<T> = when (this) {
            is KBv1Sort -> bv1Zero as KExpr<T>
            is KBoolSort -> ctx.falseExpr as KExpr<T>
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
    ): ExprConversionResult {
        val args = Native.bitwuzlaTermGetChildren(this)
        return convert(args, op)
    }

    inline fun <T : KSort, A0 : KSort, A1 : KSort> BitwuzlaTerm.convert(
        op: (KExpr<A0>, KExpr<A1>) -> KExpr<T>
    ): ExprConversionResult {
        val args = Native.bitwuzlaTermGetChildren(this)
        return convert(args, op)
    }

    inline fun <T : KSort, A0 : KSort, A1 : KSort, A2 : KSort> BitwuzlaTerm.convert(
        op: (KExpr<A0>, KExpr<A1>, KExpr<A2>) -> KExpr<T>
    ): ExprConversionResult {
        val args = Native.bitwuzlaTermGetChildren(this)
        return convert(args, op)
    }

    inline fun <T : KSort, A : KSort> BitwuzlaTerm.convertList(
        op: (List<KExpr<A>>) -> KExpr<T>
    ): ExprConversionResult {
        val args = Native.bitwuzlaTermGetChildren(this)
        return convertList(args, op)
    }
}
