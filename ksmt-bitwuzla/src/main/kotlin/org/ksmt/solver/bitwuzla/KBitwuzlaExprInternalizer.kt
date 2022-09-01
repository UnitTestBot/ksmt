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
import org.ksmt.expr.KBitVec16Value
import org.ksmt.expr.KBitVec1Value
import org.ksmt.expr.KBitVec32Value
import org.ksmt.expr.KBitVec64Value
import org.ksmt.expr.KBitVec8Value
import org.ksmt.expr.KBitVecCustomValue
import org.ksmt.expr.KBitVecNumberValue
import org.ksmt.expr.KBvAddExpr
import org.ksmt.expr.KBvAddNoOverflowExpr
import org.ksmt.expr.KBvAddNoUnderflowExpr
import org.ksmt.expr.KBvAndExpr
import org.ksmt.expr.KBvArithShiftRightExpr
import org.ksmt.expr.KBvConcatExpr
import org.ksmt.expr.KBvDivNoOverflowExpr
import org.ksmt.expr.KBvExtractExpr
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
import org.ksmt.expr.KBvRepeatExpr
import org.ksmt.expr.KBvRotateLeftExpr
import org.ksmt.expr.KBvRotateLeftIndexedExpr
import org.ksmt.expr.KBvRotateRightExpr
import org.ksmt.expr.KBvRotateRightIndexedExpr
import org.ksmt.expr.KBvShiftLeftExpr
import org.ksmt.expr.KBvSignExtensionExpr
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
import org.ksmt.expr.KBvZeroExtensionExpr
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
import org.ksmt.expr.KTrue
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.KXorExpr
import org.ksmt.solver.KSolverUnsupportedFeatureException
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaBVBase
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaKind
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaSort
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.solver.util.KExprInternalizerBase
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv16Sort
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBv64Sort
import org.ksmt.sort.KBv8Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KSortVisitor
import org.ksmt.sort.KUninterpretedSort

open class KBitwuzlaExprInternalizer(
    override val ctx: KContext,
    val bitwuzlaCtx: KBitwuzlaContext
) : KExprInternalizerBase<BitwuzlaTerm>() {

    open val sortInternalizer: SortInternalizer by lazy { SortInternalizer(bitwuzlaCtx) }
    open val functionSortInternalizer: FunctionSortInternalizer by lazy {
        FunctionSortInternalizer(bitwuzlaCtx, sortInternalizer)
    }

    override fun findInternalizedExpr(expr: KExpr<*>): BitwuzlaTerm? {
        return bitwuzlaCtx[expr]
    }

    override fun saveInternalizedExpr(expr: KExpr<*>, internalized: BitwuzlaTerm) {
        saveExprInternalizationResult(expr, internalized)
    }

    /*
    * Create Bitwuzla term from KSmt expression
    * */
    fun <T : KSort> KExpr<T>.internalize(): BitwuzlaTerm {
        bitwuzlaCtx.ensureActive()
        return internalizeExpr()
    }

    /*
    * Create Bitwuzla sort from KSmt sort
    * */
    fun <T : KSort> T.internalizeSort(): BitwuzlaSort = accept(sortInternalizer)

    /*
    * Create Bitwuzla function sort for KSmt declaration.
    * If declaration is a constant then nonfunction sort is returned
    * */
    fun <T : KSort> KDecl<T>.bitwuzlaFunctionSort(): BitwuzlaSort = accept(functionSortInternalizer)

    fun saveExprInternalizationResult(expr: KExpr<*>, term: BitwuzlaTerm) {
        bitwuzlaCtx.internalizeExpr(expr) { term }
        val kind = Native.bitwuzlaTermGetKind(term)

        /**
         * Save internalized values for [KBitwuzlaExprConverter] needs
         * @see [KBitwuzlaContext.saveInternalizedValue]
         * */
        if (kind != BitwuzlaKind.BITWUZLA_KIND_VAL) return

        if (bitwuzlaCtx.convertValue(term) != null) return

        if (term != bitwuzlaCtx.trueTerm && term != bitwuzlaCtx.falseTerm) {
            bitwuzlaCtx.saveInternalizedValue(expr, term)
        }
    }

    /**
     * [KBitwuzlaExprInternalizer] overrides transform for all supported Bitwuzla expressions.
     * Therefore, if basic expr transform is invoked expression is unsupported.
     * */
    override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> =
        error("Unsupported expr $expr")

    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> = with(expr) {
        transformList(args) { args: Array<BitwuzlaTerm> ->
            val const = bitwuzlaCtx.mkConstant(decl, decl.bitwuzlaFunctionSort())
            val termArgs = (listOf(const) + args).toTypedArray()
            Native.bitwuzlaMkTerm(bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_APPLY, termArgs)
        }
    }

    override fun <T : KSort> transform(expr: KConst<T>) = expr.transform {
        with(ctx) {
            bitwuzlaCtx.mkConstant(expr.decl, expr.sort.internalizeSort())
        }
    }

    override fun transform(expr: KAndExpr) = with(expr) {
        transformList(args) { args: Array<BitwuzlaTerm> ->
            when (args.size) {
                0 -> bitwuzlaCtx.trueTerm
                1 -> args[0]
                else -> Native.bitwuzlaMkTerm(
                    bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_AND, args
                )
            }
        }
    }

    override fun transform(expr: KOrExpr) = with(expr) {
        transformList(args) { args: Array<BitwuzlaTerm> ->
            when (args.size) {
                0 -> bitwuzlaCtx.falseTerm
                1 -> args[0]
                else -> Native.bitwuzlaMkTerm(
                    bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_OR, args
                )
            }
        }
    }

    override fun transform(expr: KNotExpr) = with(expr) {
        transform(arg, BitwuzlaKind.BITWUZLA_KIND_NOT)
    }

    override fun transform(expr: KImpliesExpr) = with(expr) {
        transform(p, q, BitwuzlaKind.BITWUZLA_KIND_IMPLIES)
    }

    override fun transform(expr: KXorExpr) = with(expr) {
        transform(a, b, BitwuzlaKind.BITWUZLA_KIND_XOR)
    }

    override fun transform(expr: KTrue) = expr.transform { bitwuzlaCtx.trueTerm }

    override fun transform(expr: KFalse) = expr.transform { bitwuzlaCtx.falseTerm }

    override fun <T : KSort> transform(expr: KEqExpr<T>) = with(expr) {
        transform(lhs, rhs, BitwuzlaKind.BITWUZLA_KIND_EQUAL)
    }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>) = with(expr) {
        transformList(args) { args: Array<BitwuzlaTerm> ->
            Native.bitwuzlaMkTerm(
                bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_DISTINCT, args
            )
        }
    }

    override fun <T : KSort> transform(expr: KIteExpr<T>) = with(expr) {
        transform(condition, trueBranch, falseBranch, BitwuzlaKind.BITWUZLA_KIND_ITE)
    }

    override fun transform(expr: KBitVec1Value) = with(expr) {
        transform { if (value) bitwuzlaCtx.trueTerm else bitwuzlaCtx.falseTerm }
    }

    override fun transform(expr: KBitVec8Value): KExpr<KBv8Sort> = transformBvNumber(expr)
    override fun transform(expr: KBitVec16Value): KExpr<KBv16Sort> = transformBvNumber(expr)
    override fun transform(expr: KBitVec32Value): KExpr<KBv32Sort> = transformBvNumber(expr)
    override fun transform(expr: KBitVec64Value): KExpr<KBv64Sort> = transformBvNumber(expr)

    fun <T : KBitVecNumberValue<S, *>, S : KBvSort> transformBvNumber(expr: T): T = with(expr) {
        transform {
            with(ctx) {
                Native.bitwuzlaMkBvValueUint64(
                    bitwuzlaCtx.bitwuzla,
                    sort.internalizeSort(),
                    numberValue.toLong()
                ).also { bitwuzlaCtx.saveInternalizedValue(expr, it) }
            }
        }
    }

    override fun transform(expr: KBitVecCustomValue) = with(expr) {
        transform {
            with(ctx) {
                Native.bitwuzlaMkBvValue(
                    bitwuzlaCtx.bitwuzla,
                    sort.internalizeSort(),
                    binaryStringValue,
                    BitwuzlaBVBase.BITWUZLA_BV_BASE_BIN
                ).also { bitwuzlaCtx.saveInternalizedValue(expr, it) }
            }
        }
    }

    override fun <T : KBvSort> transform(expr: KBvNotExpr<T>) = with(expr) {
        transform(value, BitwuzlaKind.BITWUZLA_KIND_BV_NOT)
    }

    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>) = with(expr) {
        transform(value, BitwuzlaKind.BITWUZLA_KIND_BV_REDAND)
    }

    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>) = with(expr) {
        transform(value, BitwuzlaKind.BITWUZLA_KIND_BV_REDOR)
    }

    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_AND)
    }

    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_OR)
    }

    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_XOR)
    }

    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_NAND)
    }

    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_NOR)
    }

    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_XNOR)
    }

    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>) = with(expr) {
        transform(value, BitwuzlaKind.BITWUZLA_KIND_BV_NEG)
    }

    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_ADD)
    }

    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_SUB)
    }

    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_MUL)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_UDIV)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_SDIV)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_UREM)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_SREM)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_SMOD)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_ULT)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_SLT)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_ULE)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_SLE)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_UGE)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_SGE)
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_UGT)
    }

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_SGT)
    }

    override fun transform(expr: KBvConcatExpr) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_CONCAT)
    }

    override fun transform(expr: KBvExtractExpr) = with(expr) {
        transform(value) { arg: BitwuzlaTerm ->
            Native.bitwuzlaMkTerm1Indexed2(
                bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_BV_EXTRACT,
                arg, expr.high, expr.low
            )
        }
    }

    override fun transform(expr: KBvSignExtensionExpr) = with(expr) {
        transform(value) { arg: BitwuzlaTerm ->
            Native.bitwuzlaMkTerm1Indexed1(
                bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_BV_SIGN_EXTEND,
                arg, expr.i
            )
        }
    }

    override fun transform(expr: KBvZeroExtensionExpr) = with(expr) {
        transform(value) { arg: BitwuzlaTerm ->
            Native.bitwuzlaMkTerm1Indexed1(
                bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_BV_ZERO_EXTEND,
                arg, expr.i
            )
        }
    }

    override fun transform(expr: KBvRepeatExpr) = with(expr) {
        transform(value) { arg: BitwuzlaTerm ->
            Native.bitwuzlaMkTerm1Indexed1(
                bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_BV_REPEAT,
                arg, expr.i
            )
        }
    }

    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_SHL)
    }

    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_SHR)
    }

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_ASHR)
    }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_ROL)
    }

    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_ROR)
    }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>) = with(expr) {
        transform(value) { arg: BitwuzlaTerm ->
            Native.bitwuzlaMkTerm1Indexed1(
                bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_BV_ROLI,
                arg, expr.i
            )
        }
    }

    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>) = with(expr) {
        transform(value) { arg: BitwuzlaTerm ->
            Native.bitwuzlaMkTerm1Indexed1(
                bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_BV_RORI,
                arg, expr.i
            )
        }
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>) = with(expr) {
        val kind = if (isSigned) {
            BitwuzlaKind.BITWUZLA_KIND_BV_SADD_OVERFLOW
        } else {
            BitwuzlaKind.BITWUZLA_KIND_BV_UADD_OVERFLOW
        }
        transform(arg0, arg1, kind)
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>) = with(expr) {
        TODO("no direct support for $expr")
    }

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_SSUB_OVERFLOW)
    }

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>) = with(expr) {
        TODO("no direct support for $expr")
    }

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_SDIV_OVERFLOW)
    }

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>) = with(expr) {
        TODO("no direct support for $expr")
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>) = with(expr) {
        val kind = if (isSigned) {
            BitwuzlaKind.BITWUZLA_KIND_BV_SMUL_OVERFLOW
        } else {
            BitwuzlaKind.BITWUZLA_KIND_BV_UMUL_OVERFLOW
        }
        transform(arg0, arg1, kind)
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>) = with(expr) {
        TODO("no direct support for $expr")
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>) = with(expr) {
        transform(array, index, value, BitwuzlaKind.BITWUZLA_KIND_ARRAY_STORE)
    }

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>) = with(expr) {
        transform(array, index, BitwuzlaKind.BITWUZLA_KIND_ARRAY_SELECT)
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayConst<D, R>) = with(expr) {
        transform(value) { value: BitwuzlaTerm ->
            Native.bitwuzlaMkConstArray(bitwuzlaCtx.bitwuzla, sort.internalizeSort(), value)
        }
    }

    @Suppress("LABEL_NAME_CLASH")
    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>) = with(expr) {
        transform {
            bitwuzlaCtx.withConstantScope {
                val indexVar = mkVar(indexVarDecl, indexVarDecl.sort.internalizeSort())
                val bodyInternalizer = KBitwuzlaExprInternalizer(ctx, bitwuzlaCtx)
                val body = with(bodyInternalizer) {
                    body.internalize()
                }
                val bodyKind = Native.bitwuzlaTermGetKind(body)
                if (bodyKind == BitwuzlaKind.BITWUZLA_KIND_ARRAY_SELECT) {
                    val selectArgs = Native.bitwuzlaTermGetChildren(body)
                    if (selectArgs[1] == indexVar) {
                        /** Recognize and support special case of lambda expressions
                         * which can be produced by [KBitwuzlaExprConverter].
                         *
                         * (lambda (i) (select array i)) -> array
                         */
                        return@transform selectArgs[0]
                    }
                }
                throw KSolverUnsupportedFeatureException("array lambda expressions are not supported in Bitwuzla")
            }
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
    ) = transform {
        bitwuzlaCtx.withConstantScope {
            val boundVars = bounds.map {
                mkVar(it, it.sort.internalizeSort())
            }
            val bodyInternalizer = KBitwuzlaExprInternalizer(ctx, bitwuzlaCtx)
            val body = with(bodyInternalizer) {
                body.internalize()
            }
            if (bounds.isEmpty()) return@transform body
            val args = (boundVars + body).toTypedArray()
            internalizer(args)
        }
    }

    open class SortInternalizer(private val bitwuzlaCtx: KBitwuzlaContext) : KSortVisitor<BitwuzlaSort> {
        override fun visit(sort: KBoolSort): BitwuzlaSort = bitwuzlaCtx.internalizeSort(sort) {
            bitwuzlaCtx.boolSort
        }

        override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): BitwuzlaSort =
            bitwuzlaCtx.internalizeSort(sort) {
                if (sort.range is KArraySort<*, *> || sort.domain is KArraySort<*, *>) {
                    throw KSolverUnsupportedFeatureException("Bitwuzla doesn't support nested arrays")
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
        override fun visit(sort: KIntSort): BitwuzlaSort =
            throw KSolverUnsupportedFeatureException("Unsupported sort $sort")

        override fun visit(sort: KRealSort): BitwuzlaSort =
            throw KSolverUnsupportedFeatureException("Unsupported sort $sort")

        override fun visit(sort: KUninterpretedSort): BitwuzlaSort =
            throw KSolverUnsupportedFeatureException("Unsupported sort $sort")

        override fun <S : KFpSort> visit(sort: S): BitwuzlaSort =
            TODO("We do not support KFP sort yet")

        override fun visit(sort: KFpRoundingModeSort): BitwuzlaSort {
            TODO("We do not support KFpRoundingModeSort yet")
        }
    }

    open class FunctionSortInternalizer(
        private val bitwuzlaCtx: KBitwuzlaContext,
        private val sortInternalizer: SortInternalizer
    ) : KDeclVisitor<BitwuzlaSort> {
        override fun <S : KSort> visit(decl: KFuncDecl<S>): BitwuzlaSort = bitwuzlaCtx.internalizeDeclSort(decl) {
            if (decl.argSorts.any { it is KArraySort<*, *> }) {
                throw KSolverUnsupportedFeatureException("Bitwuzla doesn't support functions with arrays in domain")
            }
            if (decl.argSorts.isNotEmpty() && decl.sort is KArraySort<*, *>) {
                throw KSolverUnsupportedFeatureException("Bitwuzla doesn't support functions with arrays in range")
            }
            val domain = decl.argSorts.map { it.accept(sortInternalizer) }.toTypedArray()
            val range = decl.sort.accept(sortInternalizer)
            if (domain.isEmpty()) return@internalizeDeclSort range
            Native.bitwuzlaMkFunSort(bitwuzlaCtx.bitwuzla, domain.size, domain, range)
        }
    }

    fun <S : KExpr<*>> S.transform(
        arg: KExpr<*>,
        kind: BitwuzlaKind
    ): S = transform(arg) { a0: BitwuzlaTerm -> Native.bitwuzlaMkTerm1(bitwuzlaCtx.bitwuzla, kind, a0) }


    fun <S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        kind: BitwuzlaKind
    ): S = transform(arg0, arg1) { a0: BitwuzlaTerm, a1: BitwuzlaTerm ->
        Native.bitwuzlaMkTerm2(bitwuzlaCtx.bitwuzla, kind, a0, a1)
    }

    fun <S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        arg2: KExpr<*>,
        kind: BitwuzlaKind
    ): S = transform(arg0, arg1, arg2) { a0: BitwuzlaTerm, a1: BitwuzlaTerm, a2: BitwuzlaTerm ->
        Native.bitwuzlaMkTerm3(bitwuzlaCtx.bitwuzla, kind, a0, a1, a2)
    }
}
