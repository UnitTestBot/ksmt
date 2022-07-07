package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaKind
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaSort
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

open class KBitwuzlaExprConverter(
    val ctx: KContext,
    val bitwuzlaCtx: KBitwuzlaContext
) {
    /*
    * Create KSmt expression from Bitwuzla term
    * todo: booleans in bitwuzla are represented as BV1
    *  convert all bv1 -> bool?
    *  keep bv1 as bv?
    *  more complex rules?
    * */
    @Suppress("UNCHECKED_CAST")
    fun <T : KSort> BitwuzlaTerm.convert(): KExpr<T> = bitwuzlaCtx.convertExpr(this) {
        convertExprHelper(this)
    } as? KExpr<T> ?: error("expr is not properly converted")

    /*
    * Create KSmt sort from Bitwuzla sort
    * */
    @Suppress("UNCHECKED_CAST")
    fun <T : KSort> BitwuzlaSort.convertSort(): T = bitwuzlaCtx.convertSort(this) {
        convertSortHelper(this)
    } as? T ?: error("sort is not properly converted")

    open fun convertSortHelper(sort: BitwuzlaSort): KSort = with(ctx) {
        when {
            Native.bitwuzlaSortIsEqual(sort, Native.bitwuzlaMkBoolSort(bitwuzlaCtx.bitwuzla)) -> {
                boolSort
            }
            Native.bitwuzlaSortIsArray(sort) -> {
                val domain = Native.bitwuzlaSortArrayGetIndex(sort).convertSort<KSort>()
                val range = Native.bitwuzlaSortArrayGetElement(sort).convertSort<KSort>()
                mkArraySort(domain, range)
            }
            Native.bitwuzlaSortIsFun(sort) -> {
                error("fun sorts are not allowed for conversion")
            }
            Native.bitwuzlaSortIsBv(sort) -> TODO("BV are not supported yet")
            else -> TODO("sort is not supported")
        }
    }

    open fun convertExprHelper(expr: BitwuzlaTerm): KExpr<*> = with(ctx) {
        when (val kind = Native.bitwuzlaTermGetKind(expr)) {
            // constants, functions, values
            BitwuzlaKind.BITWUZLA_KIND_CONST -> {
                val name = Native.bitwuzlaTermGetSymbol(expr) ?: generateBitwuzlaSymbol(expr)
                val sort = Native.bitwuzlaTermGetSort(expr)
                val decl = if (Native.bitwuzlaSortIsFun(sort) && !Native.bitwuzlaSortIsArray(sort)) {
                    val domain = Native.bitwuzlaSortFunGetDomainSorts(sort).map { it.convertSort<KSort>() }
                    val range = Native.bitwuzlaSortFunGetCodomain(sort).convertSort<KSort>()
                    mkFuncDecl(name, range, domain)
                } else {
                    mkConstDecl(name, sort.convertSort())
                }
                mkConstApp(decl)
            }
            BitwuzlaKind.BITWUZLA_KIND_APPLY -> {
                val children = Native.bitwuzlaTermGetChildren(expr)
                check(children.isNotEmpty()) { "Apply has no function term" }
                val function = children[0]
                check(Native.bitwuzlaTermIsFun(function)) { "function term expected" }
                val decl = (function.convert<KSort>() as KApp<*, *>).decl()
                val args = children.drop(1).map { it.convert<KSort>() }
                decl.apply(args)
            }
            BitwuzlaKind.BITWUZLA_KIND_VAL -> when {
                Native.bitwuzlaMkTrue(bitwuzlaCtx.bitwuzla) == expr -> trueExpr
                Native.bitwuzlaMkFalse(bitwuzlaCtx.bitwuzla) == expr -> falseExpr
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
            BitwuzlaKind.BITWUZLA_KIND_AND -> TODO()
            BitwuzlaKind.BITWUZLA_KIND_OR -> TODO()
            BitwuzlaKind.BITWUZLA_KIND_NOT -> TODO()
            BitwuzlaKind.BITWUZLA_KIND_EQUAL -> TODO()
            BitwuzlaKind.BITWUZLA_KIND_IFF -> TODO()
            BitwuzlaKind.BITWUZLA_KIND_ITE -> TODO()
            BitwuzlaKind.BITWUZLA_KIND_IMPLIES -> TODO()
            BitwuzlaKind.BITWUZLA_KIND_XOR -> TODO()

            // array
            BitwuzlaKind.BITWUZLA_KIND_CONST_ARRAY -> {
                val children = Native.bitwuzlaTermGetChildren(expr)
                check(children.size == 1) { "incorrect const array arguments" }
                val value = children[0].convert<KSort>()
                val sort = Native.bitwuzlaTermGetSort(expr).convertSort<KArraySort<KSort, KSort>>()
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
                val array = children[0].convert<KArraySort<KSort, KSort>>()
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
            BitwuzlaKind.BITWUZLA_KIND_BV_ZERO_EXTEND -> TODO("BV are not supported yet")

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

    private fun generateBitwuzlaSymbol(expr: BitwuzlaTerm): String {
        /* generate symbol in the same way as in bitwuzla model printer
        * https://github.com/bitwuzla/bitwuzla/blob/main/src/bzlaprintmodel.c#L263
        * */
        val id = Native.bitwuzlaTermHash(expr)
        return "uf$id"
    }
}
