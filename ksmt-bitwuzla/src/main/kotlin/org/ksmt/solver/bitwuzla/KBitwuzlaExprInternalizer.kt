package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.decl.KBitVec16ValueDecl
import org.ksmt.decl.KBitVec1ValueDecl
import org.ksmt.decl.KBitVec32ValueDecl
import org.ksmt.decl.KBitVec64ValueDecl
import org.ksmt.decl.KBitVec8ValueDecl
import org.ksmt.decl.KBitVecCustomSizeValueDecl
import org.ksmt.decl.KBv2IntDecl
import org.ksmt.decl.KBvAddDecl
import org.ksmt.decl.KBvAddNoOverflowDecl
import org.ksmt.decl.KBvAddNoUnderflowDecl
import org.ksmt.decl.KBvAndDecl
import org.ksmt.decl.KBvArithShiftRightDecl
import org.ksmt.decl.KBvDivNoOverflowDecl
import org.ksmt.decl.KBvLogicalShiftRightDecl
import org.ksmt.decl.KBvMulDecl
import org.ksmt.decl.KBvMulNoOverflowDecl
import org.ksmt.decl.KBvMulNoUnderflowDecl
import org.ksmt.decl.KBvNAndDecl
import org.ksmt.decl.KBvNegNoOverflowDecl
import org.ksmt.decl.KBvNegationDecl
import org.ksmt.decl.KBvNorDecl
import org.ksmt.decl.KBvNotDecl
import org.ksmt.decl.KBvOrDecl
import org.ksmt.decl.KBvReductionAndDecl
import org.ksmt.decl.KBvReductionOrDecl
import org.ksmt.decl.KBvRotateLeftDecl
import org.ksmt.decl.KBvRotateLeftIndexedDecl
import org.ksmt.decl.KBvRotateRightDecl
import org.ksmt.decl.KBvRotateRightIndexedDecl
import org.ksmt.decl.KBvShiftLeftDecl
import org.ksmt.decl.KBvSignedDivDecl
import org.ksmt.decl.KBvSignedGreaterDecl
import org.ksmt.decl.KBvSignedGreaterOrEqualDecl
import org.ksmt.decl.KBvSignedLessDecl
import org.ksmt.decl.KBvSignedLessOrEqualDecl
import org.ksmt.decl.KBvSignedModDecl
import org.ksmt.decl.KBvSignedRemDecl
import org.ksmt.decl.KBvSubDecl
import org.ksmt.decl.KBvSubNoOverflowDecl
import org.ksmt.decl.KBvSubNoUnderflowDecl
import org.ksmt.decl.KBvUnsignedDivDecl
import org.ksmt.decl.KBvUnsignedGreaterDecl
import org.ksmt.decl.KBvUnsignedGreaterOrEqualDecl
import org.ksmt.decl.KBvUnsignedLessDecl
import org.ksmt.decl.KBvUnsignedLessOrEqualDecl
import org.ksmt.decl.KBvUnsignedRemDecl
import org.ksmt.decl.KBvXNorDecl
import org.ksmt.decl.KBvXorDecl
import org.ksmt.decl.KConcatDecl
import org.ksmt.decl.KDecl
import org.ksmt.decl.KDeclVisitor
import org.ksmt.decl.KExtractDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.decl.KRepeatDecl
import org.ksmt.decl.KSignExtDecl
import org.ksmt.decl.KZeroExtDecl
import org.ksmt.expr.KAndExpr
import org.ksmt.expr.KArrayConst
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KArraySelect
import org.ksmt.expr.KArrayStore
import org.ksmt.expr.KBitVec16Value
import org.ksmt.expr.KBitVec1Value
import org.ksmt.expr.KBitVec32Value
import org.ksmt.expr.KBitVec64Value
import org.ksmt.expr.KBitVec8Value
import org.ksmt.expr.KBitVecCustomValue
import org.ksmt.expr.KBitVecNumberValue
import org.ksmt.expr.KBv2IntExpr
import org.ksmt.expr.KBvAddExpr
import org.ksmt.expr.KBvAddNoOverflowExpr
import org.ksmt.expr.KBvAddNoUnderflowExpr
import org.ksmt.expr.KBvAndExpr
import org.ksmt.expr.KBvArithShiftRightExpr
import org.ksmt.expr.KBvDivNoOverflowExpr
import org.ksmt.expr.KBvLogicalShiftRightExpr
import org.ksmt.expr.KBvMulExpr
import org.ksmt.expr.KBvMulNoOverflowExpr
import org.ksmt.expr.KBvMulNoUnderflowExpr
import org.ksmt.expr.KBvNAndExpr
import org.ksmt.expr.KBvNegNoOverflowExpr
import org.ksmt.expr.KBvNegationExpr
import org.ksmt.expr.KBvNorExpr
import org.ksmt.expr.KBvNotExpr
import org.ksmt.expr.KBvOrExpr
import org.ksmt.expr.KBvReductionAndExpr
import org.ksmt.expr.KBvReductionOrExpr
import org.ksmt.expr.KBvRotateLeftExpr
import org.ksmt.expr.KBvRotateRightExpr
import org.ksmt.expr.KBvShiftLeftExpr
import org.ksmt.expr.KBvSignedDivExpr
import org.ksmt.expr.KBvSignedGreaterExpr
import org.ksmt.expr.KBvSignedGreaterOrEqualExpr
import org.ksmt.expr.KBvSignedLessExpr
import org.ksmt.expr.KBvSignedLessOrEqualExpr
import org.ksmt.expr.KBvSignedModExpr
import org.ksmt.expr.KBvSignedRemExpr
import org.ksmt.expr.KBvSubExpr
import org.ksmt.expr.KBvSubNoOverflowExpr
import org.ksmt.expr.KBvSubNoUnderflowExpr
import org.ksmt.expr.KBvUnsignedDivExpr
import org.ksmt.expr.KBvUnsignedGreaterExpr
import org.ksmt.expr.KBvUnsignedGreaterOrEqualExpr
import org.ksmt.expr.KBvUnsignedLessExpr
import org.ksmt.expr.KBvUnsignedLessOrEqualExpr
import org.ksmt.expr.KBvUnsignedRemExpr
import org.ksmt.expr.KBvXNorExpr
import org.ksmt.expr.KBvXorExpr
import org.ksmt.expr.KConcatExpr
import org.ksmt.expr.KConst
import org.ksmt.expr.KDistinctExpr
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KExtractExpr
import org.ksmt.expr.KFalse
import org.ksmt.expr.KFunctionApp
import org.ksmt.expr.KImpliesExpr
import org.ksmt.expr.KIteExpr
import org.ksmt.expr.KNotExpr
import org.ksmt.expr.KOrExpr
import org.ksmt.expr.KQuantifier
import org.ksmt.expr.KRepeatExpr
import org.ksmt.expr.KSignExtensionExpr
import org.ksmt.expr.KTransformer
import org.ksmt.expr.KTrue
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.KZeroExtensionExpr
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

    override fun transform(expr: KBitVec8Value): KExpr<KBv8Sort> = transformBvNumber(expr)
    override fun transform(expr: KBitVec16Value): KExpr<KBv16Sort> = transformBvNumber(expr)
    override fun transform(expr: KBitVec32Value): KExpr<KBv32Sort> = transformBvNumber(expr)
    override fun transform(expr: KBitVec64Value): KExpr<KBv64Sort> = transformBvNumber(expr)

    fun <T : KBitVecNumberValue<S, *>, S : KBvSort> transformBvNumber(expr: T): T = expr.internalizeExpr {
        with(ctx) {
            Native.bitwuzlaMkBvValueUint64(
                bitwuzlaCtx.bitwuzla,
                sort.internalize(),
                numberValue.toLong()
            )
        }
    }

    override fun transform(expr: KBitVecCustomValue): KExpr<KBvSort> = expr.internalizeExpr {
        with(ctx) {
            Native.bitwuzlaMkBvValue(
                bitwuzlaCtx.bitwuzla,
                sort.internalize(),
                decimalStringValue,
                BitwuzlaBVBase.BITWUZLA_BV_BASE_BIN
            )
        }
    }

    override fun transform(expr: KBvNotExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvReductionAndExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvReductionOrExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvAndExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvOrExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvXorExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvNAndExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvNorExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvXNorExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvNegationExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvAddExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvSubExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvMulExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvUnsignedDivExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvSignedDivExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvUnsignedRemExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvSignedRemExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvSignedModExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvUnsignedLessExpr): KExpr<KBoolSort> = TODO()

    override fun transform(expr: KBvSignedLessExpr): KExpr<KBoolSort> = TODO()

    override fun transform(expr: KBvUnsignedLessOrEqualExpr): KExpr<KBoolSort> = TODO()

    override fun transform(expr: KBvSignedLessOrEqualExpr): KExpr<KBoolSort> = TODO()

    override fun transform(expr: KBvUnsignedGreaterOrEqualExpr): KExpr<KBoolSort> = TODO()

    override fun transform(expr: KBvSignedGreaterOrEqualExpr): KExpr<KBoolSort> = TODO()

    override fun transform(expr: KBvUnsignedGreaterExpr): KExpr<KBoolSort> = TODO()

    override fun transform(expr: KBvSignedGreaterExpr): KExpr<KBoolSort> = TODO()

    override fun transform(expr: KConcatExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KExtractExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KSignExtensionExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KZeroExtensionExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KRepeatExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvShiftLeftExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvLogicalShiftRightExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvArithShiftRightExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvRotateLeftExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBvRotateRightExpr): KExpr<KBvSort> = TODO()

    override fun transform(expr: KBv2IntExpr): KExpr<KIntSort> = TODO()

    override fun transform(expr: KBvAddNoOverflowExpr): KExpr<KBoolSort> = TODO()

    override fun transform(expr: KBvAddNoUnderflowExpr): KExpr<KBoolSort> = TODO()

    override fun transform(expr: KBvSubNoOverflowExpr): KExpr<KBoolSort> = TODO()

    override fun transform(expr: KBvSubNoUnderflowExpr): KExpr<KBoolSort> = TODO()

    override fun transform(expr: KBvDivNoOverflowExpr): KExpr<KBoolSort> = TODO()

    override fun transform(expr: KBvNegNoOverflowExpr): KExpr<KBoolSort> = TODO()

    override fun transform(expr: KBvMulNoOverflowExpr): KExpr<KBoolSort> = TODO()

    override fun transform(expr: KBvMulNoUnderflowExpr): KExpr<KBoolSort> = TODO()

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

        override fun visit(decl: KBitVec1ValueDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBitVec8ValueDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBitVec16ValueDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBitVec32ValueDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBitVec64ValueDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBitVecCustomSizeValueDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvNotDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvReductionAndDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvReductionOrDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvAndDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvOrDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvXorDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvNAndDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvNorDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvXNorDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvNegationDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvAddDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvSubDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvMulDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvUnsignedDivDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvSignedDivDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvUnsignedRemDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvSignedRemDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvSignedModDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvUnsignedLessDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvSignedLessDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvSignedLessOrEqualDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvUnsignedLessOrEqualDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvUnsignedGreaterOrEqualDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvSignedGreaterOrEqualDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvUnsignedGreaterDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvSignedGreaterDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KConcatDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KExtractDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KSignExtDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KZeroExtDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KRepeatDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvShiftLeftDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvLogicalShiftRightDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvArithShiftRightDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvRotateLeftDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvRotateLeftIndexedDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvRotateRightDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvRotateRightIndexedDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBv2IntDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvAddNoOverflowDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvAddNoUnderflowDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvSubNoOverflowDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvSubNoUnderflowDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvDivNoOverflowDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvNegNoOverflowDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvMulNoOverflowDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }

        override fun visit(decl: KBvMulNoUnderflowDecl): BitwuzlaSort {
            TODO("Not yet implemented")
        }
    }
}
