package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.decl.KDeclVisitor
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KAndExpr
import org.ksmt.expr.KArrayConst
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KArraySelect
import org.ksmt.expr.KArrayStore
import org.ksmt.expr.KBitVec16Expr
import org.ksmt.expr.KBitVec1Value
import org.ksmt.expr.KBitVec32Expr
import org.ksmt.expr.KBitVec64Expr
import org.ksmt.expr.KBitVec8Expr
import org.ksmt.expr.KBitVecCustomExpr
import org.ksmt.expr.KBitVecNumberExpr
import org.ksmt.expr.KConst
import org.ksmt.expr.KDistinctExpr
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFalse
import org.ksmt.expr.KFunctionApp
import org.ksmt.expr.KImpliesExpr
import org.ksmt.expr.KIteExpr
import org.ksmt.expr.KNotExpr
import org.ksmt.expr.KOrExpr
import org.ksmt.expr.KQuantifier
import org.ksmt.expr.KTransformer
import org.ksmt.expr.KTrue
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.KXorExpr
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaBVBase
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaKind
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaSort
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBv16Sort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBv64Sort
import org.ksmt.sort.KBv8Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KSortVisitor

open class KBitwuzlaExprInternalizer(
    override val ctx: KContext,
    val bitwuzlaCtx: KBitwuzlaContext
) : KTransformer {
    open val sortInternalizer: SortInternalizer by lazy { SortInternalizer(bitwuzlaCtx) }
    open val functionSortInternalizer: FunctionSortInternalizer by lazy {
        FunctionSortInternalizer(bitwuzlaCtx, sortInternalizer)
    }

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
    fun <T : KSort> KDecl<T>.bitwuzlaFunctionSort(): BitwuzlaSort = accept(functionSortInternalizer)

    /**
     * [KBitwuzlaExprInternalizer] overrides transform for all supported Bitwuzla expressions.
     * Therefore, if basic expr transform is invoked expression is unsupported.
     * */
    override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> =
        error("Unsupported expr $expr")


    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> = expr.internalizeExpr {
        val const = bitwuzlaCtx.mkConstant(decl, decl.bitwuzlaFunctionSort())
        val args = args.map { it.internalize() }
        val termArgs = (listOf(const) + args).toTypedArray()
        Native.bitwuzlaMkTerm(bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_APPLY, termArgs)
    }

    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = expr.internalizeExpr {
        with(ctx) {
            bitwuzlaCtx.mkConstant(decl, sort.internalize())
        }
    }

    override fun transform(expr: KAndExpr): KExpr<KBoolSort> = expr.internalizeExpr {
        when (args.size) {
            0 -> bitwuzlaCtx.trueTerm
            1 -> args[0].internalize()
            else -> Native.bitwuzlaMkTerm(
                bitwuzlaCtx.bitwuzla,
                BitwuzlaKind.BITWUZLA_KIND_AND,
                args.map { it.internalize() }.toTypedArray()
            )
        }
    }

    override fun transform(expr: KOrExpr): KExpr<KBoolSort> = expr.internalizeExpr {
        when (args.size) {
            0 -> bitwuzlaCtx.falseTerm
            1 -> args[0].internalize()
            else -> Native.bitwuzlaMkTerm(
                bitwuzlaCtx.bitwuzla,
                BitwuzlaKind.BITWUZLA_KIND_OR,
                args.map { it.internalize() }.toTypedArray()
            )
        }
    }

    override fun transform(expr: KNotExpr): KExpr<KBoolSort> = expr.internalizeExpr {
        Native.bitwuzlaMkTerm1(bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_NOT, arg.internalize())
    }

    override fun transform(expr: KImpliesExpr): KExpr<KBoolSort> = expr.internalizeExpr {
        Native.bitwuzlaMkTerm2(
            bitwuzlaCtx.bitwuzla,
            BitwuzlaKind.BITWUZLA_KIND_IMPLIES,
            p.internalize(),
            q.internalize()
        )
    }

    override fun transform(expr: KXorExpr): KExpr<KBoolSort> = expr.internalizeExpr {
        Native.bitwuzlaMkTerm2(
            bitwuzlaCtx.bitwuzla,
            BitwuzlaKind.BITWUZLA_KIND_XOR,
            a.internalize(),
            b.internalize()
        )
    }

    override fun transform(expr: KTrue): KExpr<KBoolSort> = expr.internalizeExpr {
        bitwuzlaCtx.trueTerm
    }

    override fun transform(expr: KFalse): KExpr<KBoolSort> = expr.internalizeExpr {
        bitwuzlaCtx.falseTerm
    }

    override fun <T : KSort> transform(expr: KEqExpr<T>): KExpr<KBoolSort> = expr.internalizeExpr {
        Native.bitwuzlaMkTerm2(
            bitwuzlaCtx.bitwuzla,
            BitwuzlaKind.BITWUZLA_KIND_EQUAL,
            lhs.internalize(),
            rhs.internalize()
        )
    }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>): KExpr<KBoolSort> = expr.internalizeExpr {
        Native.bitwuzlaMkTerm(
            bitwuzlaCtx.bitwuzla,
            BitwuzlaKind.BITWUZLA_KIND_DISTINCT,
            args.map { it.internalize() }.toTypedArray()
        )
    }

    override fun <T : KSort> transform(expr: KIteExpr<T>): KExpr<T> = expr.internalizeExpr {
        Native.bitwuzlaMkTerm3(
            bitwuzlaCtx.bitwuzla,
            BitwuzlaKind.BITWUZLA_KIND_ITE,
            condition.internalize(),
            trueBranch.internalize(),
            falseBranch.internalize()
        )
    }

    override fun transform(expr: KBitVec1Value): KExpr<KBv1Sort> = expr.internalizeExpr {
        if (expr.value) bitwuzlaCtx.trueTerm else bitwuzlaCtx.falseTerm
    }

    override fun transform(expr: KBitVec8Expr): KExpr<KBv8Sort> = transformBvNumber(expr)
    override fun transform(expr: KBitVec16Expr): KExpr<KBv16Sort> = transformBvNumber(expr)
    override fun transform(expr: KBitVec32Expr): KExpr<KBv32Sort> = transformBvNumber(expr)
    override fun transform(expr: KBitVec64Expr): KExpr<KBv64Sort> = transformBvNumber(expr)

    fun <T : KBitVecNumberExpr<S, *>, S : KBvSort> transformBvNumber(expr: T): T = expr.internalizeExpr {
        with(ctx) {
            Native.bitwuzlaMkBvValueUint64(
                bitwuzlaCtx.bitwuzla,
                sort.internalize(),
                numberValue.toLong()
            )
        }
    }

    override fun transform(expr: KBitVecCustomExpr): KExpr<KBvSort> = expr.internalizeExpr {
        with(ctx) {
            Native.bitwuzlaMkBvValue(
                bitwuzlaCtx.bitwuzla,
                sort.internalize(),
                value,
                BitwuzlaBVBase.BITWUZLA_BV_BASE_BIN
            )
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>): KExpr<KArraySort<D, R>> =
        expr.internalizeExpr {
            Native.bitwuzlaMkTerm3(
                bitwuzlaCtx.bitwuzla,
                BitwuzlaKind.BITWUZLA_KIND_ARRAY_STORE,
                array.internalize(),
                index.internalize(),
                value.internalize()
            )
        }

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> = expr.internalizeExpr {
        Native.bitwuzlaMkTerm2(
            bitwuzlaCtx.bitwuzla,
            BitwuzlaKind.BITWUZLA_KIND_ARRAY_SELECT,
            array.internalize(),
            index.internalize()
        )
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayConst<D, R>): KExpr<KArraySort<D, R>> =
        expr.internalizeExpr {
            Native.bitwuzlaMkConstArray(
                bitwuzlaCtx.bitwuzla,
                sort.internalize(),
                value.internalize()
            )
        }

    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>> =
        expr.internalizeExpr {
            bitwuzlaCtx.withConstantScope {
                val indexVar = mkVar(indexVarDecl, indexVarDecl.sort.internalize())
                val body = body.internalize()
                val bodyKind = Native.bitwuzlaTermGetKind(body)
                if (bodyKind == BitwuzlaKind.BITWUZLA_KIND_ARRAY_SELECT) {
                    val selectArgs = Native.bitwuzlaTermGetChildren(body)
                    if (selectArgs[1] == indexVar) {
                        /** Recognize and support special case of lambda expressions
                         * which can be produced by [KBitwuzlaExprConverter].
                         *
                         * (lambda (i) (select array i)) -> array
                         */
                        return@internalizeExpr selectArgs[0]
                    }
                }
                error("array lambda expressions are not supported in Bitwuzla")
            }
        }

    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> = expr.internalizeQuantifier { args ->
        Native.bitwuzlaMkTerm(bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_EXISTS, args)
    }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> = expr.internalizeQuantifier { args ->
        Native.bitwuzlaMkTerm(bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_FORALL, args)
    }

    inline fun <T : KQuantifier> T.internalizeQuantifier(
        crossinline internalizer: T.(Array<BitwuzlaTerm>) -> BitwuzlaTerm
    ) = internalizeExpr {
        bitwuzlaCtx.withConstantScope {
            val boundVars = bounds.map {
                mkVar(it, it.sort.internalize())
            }
            val body = body.internalize()
            if (bounds.isEmpty()) return@internalizeExpr body
            val args = (boundVars + body).toTypedArray()
            internalizer(args)
        }
    }

    inline fun <T : KExpr<*>> T.internalizeExpr(crossinline internalizer: T.() -> BitwuzlaTerm): T {
        bitwuzlaCtx.internalizeExpr(this) {
            internalizer()
        }
        return this
    }

    open class SortInternalizer(private val bitwuzlaCtx: KBitwuzlaContext) : KSortVisitor<BitwuzlaSort> {
        override fun visit(sort: KBoolSort): BitwuzlaSort = bitwuzlaCtx.internalizeSort(sort) {
            bitwuzlaCtx.boolSort
        }

        override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): BitwuzlaSort =
            bitwuzlaCtx.internalizeSort(sort) {
                check(sort.range !is KArraySort<*, *> && sort.domain !is KArraySort<*, *>) {
                    "Bitwuzla doesn't support nested arrays"
                }
                val domain = sort.domain.accept(this@SortInternalizer)
                val range = sort.range.accept(this@SortInternalizer)
                Native.bitwuzlaMkArraySort(bitwuzlaCtx.bitwuzla, domain, range)
            }

        override fun <S : KBvSort> visit(sort: S): BitwuzlaSort =
            bitwuzlaCtx.internalizeSort(sort) {
                val size = sort.sizeBits.toInt()
                if (size == 1) {
                    bitwuzlaCtx.boolSort
                } else {
                    Native.bitwuzlaMkBvSort(bitwuzlaCtx.bitwuzla, sort.sizeBits.toInt())
                }
            }

        /**
         * Bitwuzla doesn't support integers and reals.
         * */
        override fun visit(sort: KIntSort): BitwuzlaSort = error("Unsupported sort $sort")
        override fun visit(sort: KRealSort): BitwuzlaSort = error("Unsupported sort $sort")
    }

    open class FunctionSortInternalizer(
        private val bitwuzlaCtx: KBitwuzlaContext,
        private val sortInternalizer: SortInternalizer
    ) : KDeclVisitor<BitwuzlaSort> {
        override fun <S : KSort> visit(decl: KFuncDecl<S>): BitwuzlaSort = bitwuzlaCtx.internalizeDeclSort(decl) {
            check(decl.argSorts.all { it !is KArraySort<*, *> }) {
                "Bitwuzla doesn't support functions with arrays in domain"
            }
            check(decl.argSorts.isEmpty() || decl.sort !is KArraySort<*, *>) {
                "Bitwuzla doesn't support functions with arrays in range"
            }
            val domain = decl.argSorts.map { it.accept(sortInternalizer) }.toTypedArray()
            val range = decl.sort.accept(sortInternalizer)
            if (domain.isEmpty()) return@internalizeDeclSort range
            Native.bitwuzlaMkFunSort(bitwuzlaCtx.bitwuzla, domain.size, domain, range)
        }
    }
}
