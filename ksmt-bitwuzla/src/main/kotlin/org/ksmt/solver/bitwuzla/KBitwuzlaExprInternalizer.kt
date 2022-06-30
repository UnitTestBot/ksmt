package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.decl.KDeclVisitor
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.*
import org.ksmt.sort.*
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaKind
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaSort
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native

open class KBitwuzlaExprInternalizer(
    override val ctx: KContext,
    val bitwuzlaCtx: KBitwuzlaContext
) : KTransformer {
    open val sortInternalizer: SortInternalizer by lazy { SortInternalizer(bitwuzlaCtx) }
    open val declSortInternalizer: DeclSortInternalizer by lazy { DeclSortInternalizer(bitwuzlaCtx, sortInternalizer) }

    fun <T : KSort> KExpr<T>.internalize(): BitwuzlaTerm {
        bitwuzlaCtx.ensureActive()
        accept(this@KBitwuzlaExprInternalizer)
        return bitwuzlaCtx[this] ?: error("expression is not properly internalized")
    }

    fun <T : KSort> T.internalize(): BitwuzlaSort = accept(sortInternalizer)
    fun <T : KSort> KDecl<T>.bitwuzlaSort(): BitwuzlaSort = accept(declSortInternalizer)


    override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> =
        error("Unsupported expr $expr")


    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> = expr.internalizeExpr {
        val const = bitwuzlaCtx.mkConstant(expr.decl.name, expr.decl.bitwuzlaSort())
        val args = args.map { it.internalize() }
        val termArgs = (listOf(const) + args).toTypedArray()
        Native.bitwuzla_mk_term(bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_APPLY, termArgs)
    }

    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = expr.internalizeExpr {
        with(ctx) {
            bitwuzlaCtx.mkConstant(expr.decl.name, expr.sort.internalize())
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

    override fun <T : KBVSort<KBVSize>> transform(expr: BitVecExpr<T>): KExpr<T> = expr.internalizeExpr {
        TODO("BV is not supported yet")
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
            with(ctx) {
                val body = body.internalize()
                if (bounds.isEmpty()) return@internalizeExpr body
                val internalizedBounds = bounds.map {
                    Native.bitwuzla_mk_const(bitwuzlaCtx.bitwuzla, it.sort.internalize(), it.name)
                }
                val boundVars = bounds.map { bitwuzlaCtx.mkFreshVar(it.sort.internalize()) }
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
                val range = sort.domain.accept(this@SortInternalizer)
                Native.bitwuzla_mk_array_sort(bitwuzlaCtx.bitwuzla, domain, range)
            }

        override fun <S : KBVSize> visit(sort: KBVSort<S>): BitwuzlaSort {
            TODO("BV is not supported yet")
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
            Native.bitwuzla_mk_fun_sort(bitwuzlaCtx.bitwuzla, domain.size, domain, range)
        }
    }
}