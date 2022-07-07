package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.decl.KDeclVisitor
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.*
import org.ksmt.sort.*
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaKind
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaBVBase
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaSort
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native

open class KBitwuzlaExprInternalizer(
    override val ctx: KContext,
    val bitwuzlaCtx: KBitwuzlaContext
) : KTransformer {
    open val sortInternalizer: SortInternalizer by lazy { SortInternalizer(bitwuzlaCtx) }
    open val declSortInternalizer: DeclSortInternalizer by lazy { DeclSortInternalizer(bitwuzlaCtx, sortInternalizer) }

    /*
    * Create Bitwuzla term from KSmt expression
    * */
    fun <T : KSort> KExpr<T>.internalize(): BitwuzlaTerm {
        bitwuzlaCtx.ensureActive()
        accept(this@KBitwuzlaExprInternalizer)
        return bitwuzlaCtx[this] ?: error("expression is not properly internalized")
    }

    /*
    * Create Bitwuzla sort from KSmt sort
    * */
    fun <T : KSort> T.internalize(): BitwuzlaSort = accept(sortInternalizer)

    /*
    * Create Bitwuzla function sort for KSmt declaration.
    * If declaration is a constant then nonfunction sort is returned
    * */
    fun <T : KSort> KDecl<T>.bitwuzlaSort(): BitwuzlaSort = accept(declSortInternalizer)


    override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> =
        error("Unsupported expr $expr")


    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> = expr.internalizeExpr {
        val const = bitwuzlaCtx.mkConstant(decl, decl.bitwuzlaSort())
        val args = args.map { it.internalize() }
        val termArgs = (listOf(const) + args).toTypedArray()
        Native.bitwuzla_mk_term(bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_APPLY, termArgs)
    }

    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = expr.internalizeExpr {
        with(ctx) {
            bitwuzlaCtx.mkConstant(decl, sort.internalize())
        }
    }

    override fun transform(expr: KAndExpr): KExpr<KBoolSort> = expr.internalizeExpr {
        when (args.size) {
            0 -> ctx.trueExpr.internalize()
            1 -> args[0].internalize()
            else -> Native.bitwuzla_mk_term(
                bitwuzlaCtx.bitwuzla,
                BitwuzlaKind.BITWUZLA_KIND_AND,
                args.map { it.internalize() }.toTypedArray()
            )
        }
    }

    override fun transform(expr: KOrExpr): KExpr<KBoolSort> = expr.internalizeExpr {
        when (args.size) {
            0 -> ctx.falseExpr.internalize()
            1 -> args[0].internalize()
            else -> Native.bitwuzla_mk_term(
                bitwuzlaCtx.bitwuzla,
                BitwuzlaKind.BITWUZLA_KIND_OR,
                args.map { it.internalize() }.toTypedArray()
            )
        }
    }

    override fun transform(expr: KNotExpr): KExpr<KBoolSort> = expr.internalizeExpr {
        Native.bitwuzla_mk_term1(bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_NOT, arg.internalize())
    }

    override fun transform(expr: KTrue): KExpr<KBoolSort> = expr.internalizeExpr {
        Native.bitwuzla_mk_true(bitwuzlaCtx.bitwuzla)
    }

    override fun transform(expr: KFalse): KExpr<KBoolSort> = expr.internalizeExpr {
        Native.bitwuzla_mk_false(bitwuzlaCtx.bitwuzla)
    }

    override fun <T : KSort> transform(expr: KEqExpr<T>): KExpr<KBoolSort> = expr.internalizeExpr {
        Native.bitwuzla_mk_term2(
            bitwuzlaCtx.bitwuzla,
            BitwuzlaKind.BITWUZLA_KIND_EQUAL,
            lhs.internalize(),
            rhs.internalize()
        )
    }

    override fun <T : KSort> transform(expr: KIteExpr<T>): KExpr<T> = expr.internalizeExpr {
        Native.bitwuzla_mk_term3(
            bitwuzlaCtx.bitwuzla,
            BitwuzlaKind.BITWUZLA_KIND_ITE,
            condition.internalize(),
            trueBranch.internalize(),
            falseBranch.internalize()
        )
    }

    override fun transform(expr: KBitVec8Expr): KExpr<KBV8Sort> = transformBvNumber(expr)
    override fun transform(expr: KBitVec16Expr): KExpr<KBV16Sort> = transformBvNumber(expr)
    override fun transform(expr: KBitVec32Expr): KExpr<KBV32Sort> = transformBvNumber(expr)
    override fun transform(expr: KBitVec64Expr): KExpr<KBV64Sort> = transformBvNumber(expr)

    fun <T : KBitVecNumberExpr<S, *>, S : KBVSort> transformBvNumber(expr: T): T = expr.internalizeExpr {
        with(ctx) {
            Native.bitwuzla_mk_bv_value_uint64(
                bitwuzlaCtx.bitwuzla,
                sort.internalize(),
                numberValue.toLong()
            )
        }
    }

    override fun transform(expr: KBitVecCustomExpr): KExpr<KBVSort> = expr.internalizeExpr {
        with(ctx) {
            Native.bitwuzla_mk_bv_value(
                bitwuzlaCtx.bitwuzla,
                sort.internalize(),
                value,
                BitwuzlaBVBase.BITWUZLA_BV_BASE_BIN
            )
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>): KExpr<KArraySort<D, R>> =
        expr.internalizeExpr {
            Native.bitwuzla_mk_term3(
                bitwuzlaCtx.bitwuzla,
                BitwuzlaKind.BITWUZLA_KIND_ARRAY_STORE,
                array.internalize(),
                index.internalize(),
                value.internalize()
            )
        }

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> = expr.internalizeExpr {
        Native.bitwuzla_mk_term2(
            bitwuzlaCtx.bitwuzla,
            BitwuzlaKind.BITWUZLA_KIND_ARRAY_SELECT,
            array.internalize(),
            index.internalize()
        )
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayConst<D, R>): KExpr<KArraySort<D, R>> =
        expr.internalizeExpr {
            Native.bitwuzla_mk_const_array(
                bitwuzlaCtx.bitwuzla,
                sort.internalize(),
                value.internalize()
            )
        }

    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> = expr.internalizeQuantifier { args ->
        Native.bitwuzla_mk_term(bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_EXISTS, args)
    }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> = expr.internalizeQuantifier { args ->
        Native.bitwuzla_mk_term(bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_FORALL, args)
    }

    inline fun <T : KQuantifier> T.internalizeQuantifier(crossinline internalizer: T.(Array<BitwuzlaTerm>) -> BitwuzlaTerm) =
        internalizeExpr {
            bitwuzlaCtx.withConstantScope {
                val internalizedBounds = bounds.map {
                    mkFreshConstant(it, it.sort.internalize())
                }
                val boundVars = bounds.map { bitwuzlaCtx.mkFreshVar(it.sort.internalize()) }
                val body = body.internalize()
                if (bounds.isEmpty()) return@internalizeExpr body
                val bodyWithVars = Native.bitwuzla_substitute_term(
                    bitwuzlaCtx.bitwuzla,
                    body,
                    internalizedBounds.toTypedArray(),
                    boundVars.toTypedArray()
                )
                val args = (boundVars + bodyWithVars).toTypedArray()
                internalizer(args)
            }
        }

    inline fun <T : KExpr<*>> T.internalizeExpr(crossinline internalizer: T.() -> BitwuzlaTerm): T {
        bitwuzlaCtx.internalizeExpr(this) {
            internalizer()
        }
        return this
    }

    open class SortInternalizer(val bitwuzlaCtx: KBitwuzlaContext) : KSortVisitor<BitwuzlaSort> {
        override fun visit(sort: KBoolSort): BitwuzlaSort = bitwuzlaCtx.internalizeSort(sort) {
            Native.bitwuzla_mk_bool_sort(bitwuzlaCtx.bitwuzla)
        }

        override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): BitwuzlaSort =
            bitwuzlaCtx.internalizeSort(sort) {
                val domain = sort.domain.accept(this@SortInternalizer)
                val range = sort.range.accept(this@SortInternalizer)
                Native.bitwuzla_mk_array_sort(bitwuzlaCtx.bitwuzla, domain, range)
            }

        override fun <S : KBVSort> visit(sort: S): BitwuzlaSort =
            bitwuzlaCtx.internalizeSort(sort) {
                Native.bitwuzla_mk_bv_sort(bitwuzlaCtx.bitwuzla, sort.sizeBits.toInt())
            }

        override fun visit(sort: KIntSort): BitwuzlaSort = error("Unsupported sort $sort")
        override fun visit(sort: KRealSort): BitwuzlaSort = error("Unsupported sort $sort")
    }

    open class DeclSortInternalizer(
        val bitwuzlaCtx: KBitwuzlaContext,
        val sortInternalizer: SortInternalizer
    ) : KDeclVisitor<BitwuzlaSort> {
        override fun <S : KSort> visit(decl: KFuncDecl<S>): BitwuzlaSort = bitwuzlaCtx.internalizeDeclSort(decl) {
            val domain = decl.argSorts.map { it.accept(sortInternalizer) }.toTypedArray()
            val range = decl.sort.accept(sortInternalizer)
            if (domain.isEmpty()) return@internalizeDeclSort range
            Native.bitwuzla_mk_fun_sort(bitwuzlaCtx.bitwuzla, domain.size, domain, range)
        }
    }
}
