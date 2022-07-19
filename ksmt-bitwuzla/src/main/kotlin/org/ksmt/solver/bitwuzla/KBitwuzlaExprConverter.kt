package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KExpr
import org.ksmt.expr.KTransformer
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaKind
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaSort
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBV1Sort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort

open class KBitwuzlaExprConverter(
    private val ctx: KContext,
    val bitwuzlaCtx: KBitwuzlaContext
) {
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
        convert<KSort>()
            .convertToExpectedIfNeeded(expectedSort)
            .accept(adapterTermRewriter)

    @Suppress("UNCHECKED_CAST")
    private fun <T : KSort> BitwuzlaTerm.convert(): KExpr<T> = bitwuzlaCtx.convertExpr(this) {
        convertExprHelper(this)
    } as? KExpr<T> ?: error("expression is not properly converted")

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

    @Suppress("MagicNumber", "LongMethod", "ComplexMethod")
    open fun convertExprHelper(expr: BitwuzlaTerm): KExpr<*> = with(ctx) {
        when (val kind = Native.bitwuzlaTermGetKind(expr)) {
            // constants, functions, values
            BitwuzlaKind.BITWUZLA_KIND_CONST -> {
                val knownConstDecl = bitwuzlaCtx.convertConstantIfKnown(expr)
                if (knownConstDecl != null) return mkConstApp(knownConstDecl).convertToBoolIfNeeded()

                // newly generated constant
                val sort = Native.bitwuzlaTermGetSort(expr)
                if (!Native.bitwuzlaSortIsFun(sort) || Native.bitwuzlaSortIsArray(sort)) {
                    val decl = generateDecl(expr) { mkConstDecl(it, sort.convertSort()) }
                    return decl.apply().convertToBoolIfNeeded()
                }

                // newly generated functional constant
                error("Constants with functional type are not supported")
            }
            BitwuzlaKind.BITWUZLA_KIND_APPLY -> {
                val children = Native.bitwuzlaTermGetChildren(expr)
                check(children.isNotEmpty()) { "Apply has no function term" }
                val function = children[0]
                check(Native.bitwuzlaTermIsFun(function)) { "function term expected" }
                val funcDecl = bitwuzlaCtx.convertConstantIfKnown(function)
                    ?.let { it as? KFuncDecl<*> ?: error("function expected. actual: $it") }
                    ?: run {
                        // new function
                        val domain = Native.bitwuzlaTermFunGetDomainSorts(function).map { it.convertSort() }
                        val range = Native.bitwuzlaTermFunGetCodomainSort(function).convertSort()
                        generateDecl(function) { mkFuncDecl(it, range, domain) }
                    }
                val args = children.drop(1).zip(funcDecl.argSorts) { arg, expectedSort ->
                    arg.convert<KSort>().convertToExpectedIfNeeded(expectedSort)
                }
                return funcDecl.apply(args).convertToBoolIfNeeded()
            }
            BitwuzlaKind.BITWUZLA_KIND_VAL -> when {
                bitwuzlaCtx.trueTerm == expr -> trueExpr
                bitwuzlaCtx.falseTerm == expr -> falseExpr
                Native.bitwuzlaTermIsBv(expr) -> {
                    val size = Native.bitwuzlaTermBvGetSize(expr)
                    val value = Native.bitwuzlaGetBvValue(bitwuzlaCtx.bitwuzla, expr)
                    mkBV(value, size.toUInt())
                }
                Native.bitwuzlaTermIsFp(expr) -> TODO("FP are not supported yet")
                else -> TODO("unsupported value")
            }
            BitwuzlaKind.BITWUZLA_KIND_VAR -> TODO("var conversion is not implemented")

            // bool
            BitwuzlaKind.BITWUZLA_KIND_EQUAL -> {
                val args = Native.bitwuzlaTermGetChildren(expr).map { it.convert<KSort>() }
                check(args.size == 2) { "unexpected number of EQ arguments: ${args.size}" }
                mkEq(args[0], args[1])
            }
            BitwuzlaKind.BITWUZLA_KIND_ITE -> {
                val children = Native.bitwuzlaTermGetChildren(expr)
                check(children.size == 3) { "unexpected number of ITE arguments: ${children.size}" }
                mkIte(children[0].convert(), children[1].convert(), children[2].convert())
            }
            BitwuzlaKind.BITWUZLA_KIND_IMPLIES -> {
                val args = Native.bitwuzlaTermGetChildren(expr)
                check(args.size == 2) { "unexpected number of Implies arguments: ${args.size}" }
                mkImplies(args[0].convert(), args[1].convert())
            }
            BitwuzlaKind.BITWUZLA_KIND_IFF -> TODO()
            BitwuzlaKind.BITWUZLA_KIND_AND,
            BitwuzlaKind.BITWUZLA_KIND_OR,
            BitwuzlaKind.BITWUZLA_KIND_NOT,
            BitwuzlaKind.BITWUZLA_KIND_XOR -> convertBoolExpr(expr, kind)

            // array
            BitwuzlaKind.BITWUZLA_KIND_CONST_ARRAY -> {
                val children = Native.bitwuzlaTermGetChildren(expr)
                check(children.size == 1) { "incorrect const array arguments" }
                val value = children[0].convert<KSort>()
                val sort = Native.bitwuzlaTermGetSort(expr).convertSort() as KArraySort<*, *>
                mkArrayConst(sort, value)
            }
            BitwuzlaKind.BITWUZLA_KIND_ARRAY_SELECT -> {
                val children = Native.bitwuzlaTermGetChildren(expr)
                check(children.size == 2) { "incorrect array select arguments" }
                val array = children[0].convert<KArraySort<KSort, KSort>>()
                val index = children[1].convert<KSort>()
                array.select(index)
            }
            BitwuzlaKind.BITWUZLA_KIND_ARRAY_STORE -> {
                val children = Native.bitwuzlaTermGetChildren(expr)
                check(children.size == 3) { "incorrect array store arguments" }
                val array = children[0].convert<KArraySort<*, *>>()
                val index = children[1].convert<KSort>()
                val value = children[2].convert<KSort>()
                array.store(index, value)
            }

            // quantifiers
            BitwuzlaKind.BITWUZLA_KIND_EXISTS,
            BitwuzlaKind.BITWUZLA_KIND_FORALL -> TODO("quantifier conversion is not implemented")

            // bit-vec
            BitwuzlaKind.BITWUZLA_KIND_BV_ADD,
            BitwuzlaKind.BITWUZLA_KIND_BV_AND,
            BitwuzlaKind.BITWUZLA_KIND_BV_ASHR,
            BitwuzlaKind.BITWUZLA_KIND_BV_COMP,
            BitwuzlaKind.BITWUZLA_KIND_BV_CONCAT,
            BitwuzlaKind.BITWUZLA_KIND_BV_DEC,
            BitwuzlaKind.BITWUZLA_KIND_BV_INC,
            BitwuzlaKind.BITWUZLA_KIND_BV_MUL,
            BitwuzlaKind.BITWUZLA_KIND_BV_NAND,
            BitwuzlaKind.BITWUZLA_KIND_BV_NEG,
            BitwuzlaKind.BITWUZLA_KIND_BV_NOR,
            BitwuzlaKind.BITWUZLA_KIND_BV_NOT,
            BitwuzlaKind.BITWUZLA_KIND_BV_OR,
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
            BitwuzlaKind.BITWUZLA_KIND_BV_XNOR,
            BitwuzlaKind.BITWUZLA_KIND_BV_XOR,
            BitwuzlaKind.BITWUZLA_KIND_BV_EXTRACT,
            BitwuzlaKind.BITWUZLA_KIND_BV_REPEAT,
            BitwuzlaKind.BITWUZLA_KIND_BV_ROLI,
            BitwuzlaKind.BITWUZLA_KIND_BV_RORI,
            BitwuzlaKind.BITWUZLA_KIND_BV_SIGN_EXTEND,
            BitwuzlaKind.BITWUZLA_KIND_BV_ZERO_EXTEND -> {
                if (Native.bitwuzlaTermGetSort(expr) == bitwuzlaCtx.boolSort) {
                    convertBoolExpr(expr, kind)
                } else {
                    convertBVExpr(expr, kind)
                }
            }

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
            BitwuzlaKind.BITWUZLA_KIND_DISTINCT,
            BitwuzlaKind.BITWUZLA_KIND_LAMBDA -> TODO("unsupported kind $kind")
        }
    }

    @Suppress("LongMethod")
    open fun convertBoolExpr(expr: BitwuzlaTerm, kind: BitwuzlaKind): KExpr<*> = when (kind) {
        BitwuzlaKind.BITWUZLA_KIND_BV_AND, BitwuzlaKind.BITWUZLA_KIND_AND -> with(ctx) {
            val args = Native.bitwuzlaTermGetChildren(expr).map { it.convert<KBoolSort>() }
            mkAnd(args)
        }
        BitwuzlaKind.BITWUZLA_KIND_BV_OR, BitwuzlaKind.BITWUZLA_KIND_OR -> with(ctx) {
            val args = Native.bitwuzlaTermGetChildren(expr).map { it.convert<KBoolSort>() }
            mkOr(args)
        }
        BitwuzlaKind.BITWUZLA_KIND_BV_NOT, BitwuzlaKind.BITWUZLA_KIND_NOT -> with(ctx) {
            val arg = Native.bitwuzlaTermGetChildren(expr)
                .map { it.convert<KBoolSort>() }
                .singleOrNull()
                ?: error("single argument expected for not operation")
            mkNot(arg)
        }
        BitwuzlaKind.BITWUZLA_KIND_BV_XOR, BitwuzlaKind.BITWUZLA_KIND_XOR -> TODO("xor is not implemented")
        BitwuzlaKind.BITWUZLA_KIND_BV_NAND,
        BitwuzlaKind.BITWUZLA_KIND_BV_NOR,
        BitwuzlaKind.BITWUZLA_KIND_BV_XNOR,
        BitwuzlaKind.BITWUZLA_KIND_BV_ADD,
        BitwuzlaKind.BITWUZLA_KIND_BV_DEC,
        BitwuzlaKind.BITWUZLA_KIND_BV_INC,
        BitwuzlaKind.BITWUZLA_KIND_BV_NEG,
        BitwuzlaKind.BITWUZLA_KIND_BV_MUL,
        BitwuzlaKind.BITWUZLA_KIND_BV_REDAND,
        BitwuzlaKind.BITWUZLA_KIND_BV_REDOR,
        BitwuzlaKind.BITWUZLA_KIND_BV_REDXOR,
        BitwuzlaKind.BITWUZLA_KIND_BV_ROL,
        BitwuzlaKind.BITWUZLA_KIND_BV_ROR,
        BitwuzlaKind.BITWUZLA_KIND_BV_ASHR,
        BitwuzlaKind.BITWUZLA_KIND_BV_COMP,
        BitwuzlaKind.BITWUZLA_KIND_BV_CONCAT,
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
        BitwuzlaKind.BITWUZLA_KIND_BV_ZERO_EXTEND -> TODO("bool operation $kind is not supported yet")
        else -> error("unexpected bool kind $kind")
    }

    open fun convertBVExpr(expr: BitwuzlaTerm, kind: BitwuzlaKind): KExpr<*> = when (kind) {
        BitwuzlaKind.BITWUZLA_KIND_BV_ADD,
        BitwuzlaKind.BITWUZLA_KIND_BV_AND,
        BitwuzlaKind.BITWUZLA_KIND_BV_ASHR,
        BitwuzlaKind.BITWUZLA_KIND_BV_COMP,
        BitwuzlaKind.BITWUZLA_KIND_BV_CONCAT,
        BitwuzlaKind.BITWUZLA_KIND_BV_DEC,
        BitwuzlaKind.BITWUZLA_KIND_BV_INC,
        BitwuzlaKind.BITWUZLA_KIND_BV_MUL,
        BitwuzlaKind.BITWUZLA_KIND_BV_NAND,
        BitwuzlaKind.BITWUZLA_KIND_BV_NEG,
        BitwuzlaKind.BITWUZLA_KIND_BV_NOR,
        BitwuzlaKind.BITWUZLA_KIND_BV_NOT,
        BitwuzlaKind.BITWUZLA_KIND_BV_OR,
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
        BitwuzlaKind.BITWUZLA_KIND_BV_XNOR,
        BitwuzlaKind.BITWUZLA_KIND_BV_XOR,
        BitwuzlaKind.BITWUZLA_KIND_BV_EXTRACT,
        BitwuzlaKind.BITWUZLA_KIND_BV_REPEAT,
        BitwuzlaKind.BITWUZLA_KIND_BV_ROLI,
        BitwuzlaKind.BITWUZLA_KIND_BV_RORI,
        BitwuzlaKind.BITWUZLA_KIND_BV_SIGN_EXTEND,
        BitwuzlaKind.BITWUZLA_KIND_BV_ZERO_EXTEND -> TODO("BV are not supported yet")
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
    private fun KExpr<*>.convertToBoolIfNeeded(): KExpr<*> = when (with(ctx) { sort }) {
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
    private fun <T : KSort> KExpr<*>.convertToExpectedIfNeeded(expected: T): KExpr<T> = when (expected) {
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
            else -> Bv1ToBoolAdapterExpr(this@ensureBoolExpr as KExpr<KBV1Sort>)
        }
    }

    /**
     * Convert expression from Bool to (BitVec 1).
     * */
    @Suppress("UNCHECKED_CAST")
    private fun KExpr<*>.ensureBv1Expr(): KExpr<KBV1Sort> = with(ctx) {
        when {
            sort == bv1Sort -> this@ensureBv1Expr as KExpr<KBV1Sort>
            this@ensureBv1Expr is Bv1ToBoolAdapterExpr -> arg
            else -> BoolToBv1AdapterExpr(this@ensureBv1Expr as KExpr<KBoolSort>)
        }
    }

    private inner class BoolToBv1AdapterExpr(val arg: KExpr<KBoolSort>) : KExpr<KBV1Sort>(ctx) {
        override fun sort(): KBV1Sort = ctx.bv1Sort
        override fun print(): String = "(toBV1 $arg)"

        override fun accept(transformer: KTransformer): KExpr<KBV1Sort> =
            (transformer as AdapterTermRewriter).transform(this)
    }

    private inner class Bv1ToBoolAdapterExpr(val arg: KExpr<KBV1Sort>) : KExpr<KBoolSort>(ctx) {
        override fun sort(): KBoolSort = ctx.boolSort
        override fun print(): String = "(toBool $arg)"

        override fun accept(transformer: KTransformer): KExpr<KBoolSort> =
            (transformer as AdapterTermRewriter).transform(this)
    }

    private inner class ArrayAdapterExpr<FromDomain : KSort, FromRange : KSort, ToDomain : KSort, ToRange : KSort>(
        val arg: KExpr<KArraySort<FromDomain, FromRange>>,
        val toDomainSort: ToDomain,
        val toRangeSort: ToRange
    ) : KExpr<KArraySort<ToDomain, ToRange>>(ctx) {
        override fun sort(): KArraySort<ToDomain, ToRange> = ctx.mkArraySort(toDomainSort, toRangeSort)
        override fun print(): String = "(toArray ${sort()} $arg)"

        override fun accept(transformer: KTransformer): KExpr<KArraySort<ToDomain, ToRange>> =
            (transformer as AdapterTermRewriter).transform(this)
    }

    /** Remove auxiliary terms introduced by [convertToBoolIfNeeded] and [convertToExpectedIfNeeded].
     * */
    private inner class AdapterTermRewriter(override val ctx: KContext) : KTransformer {
        /**
         * x: Bool
         * (toBv x) -> (ite x #b1 #b0)
         * */
        fun transform(expr: BoolToBv1AdapterExpr): KExpr<KBV1Sort> = with(ctx) {
            val arg = expr.arg.accept(this@AdapterTermRewriter)
            return mkIte(arg, bv1Sort.trueValue(), bv1Sort.falseValue())
        }

        /**
         * x: (BitVec 1)
         * (toBool x) -> (ite (x == #b1) true false)
         * */
        fun transform(expr: Bv1ToBoolAdapterExpr): KExpr<KBoolSort> = with(ctx) {
            val arg = expr.arg.accept(this@AdapterTermRewriter)
            return mkIte(arg eq bv1Sort.trueValue(), trueExpr, falseExpr)
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
            when (fromSort.domain) {
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
            }.accept(this@AdapterTermRewriter)
        }

        private val bv1One: KExpr<KBV1Sort> by lazy { ctx.mkBV(true) }
        private val bv1Zero: KExpr<KBV1Sort> by lazy { ctx.mkBV(false) }

        @Suppress("UNCHECKED_CAST")
        private fun <T : KSort> T.trueValue(): KExpr<T> = when (this) {
            is KBV1Sort -> bv1One as KExpr<T>
            is KBoolSort -> ctx.trueExpr as KExpr<T>
            else -> error("unexpected sort: $this")
        }

        @Suppress("UNCHECKED_CAST")
        private fun <T : KSort> T.falseValue(): KExpr<T> = when (this) {
            is KBV1Sort -> bv1Zero as KExpr<T>
            is KBoolSort -> ctx.falseExpr as KExpr<T>
            else -> error("unexpected sort: $this")
        }
    }
}
