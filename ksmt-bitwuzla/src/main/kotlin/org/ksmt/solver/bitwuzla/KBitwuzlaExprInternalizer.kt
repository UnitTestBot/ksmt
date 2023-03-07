package org.ksmt.solver.bitwuzla

import it.unimi.dsi.fastutil.objects.Object2LongOpenHashMap
import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.decl.KDeclVisitor
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KAddArithExpr
import org.ksmt.expr.KAndBinaryExpr
import org.ksmt.expr.KAndExpr
import org.ksmt.expr.KArray2Lambda
import org.ksmt.expr.KArray2Select
import org.ksmt.expr.KArray2Store
import org.ksmt.expr.KArray3Lambda
import org.ksmt.expr.KArray3Select
import org.ksmt.expr.KArray3Store
import org.ksmt.expr.KArrayConst
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KArrayLambdaBase
import org.ksmt.expr.KArrayNLambda
import org.ksmt.expr.KArrayNSelect
import org.ksmt.expr.KArrayNStore
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
import org.ksmt.expr.KBvToFpExpr
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
import org.ksmt.expr.KDivArithExpr
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFalse
import org.ksmt.expr.KFp128Value
import org.ksmt.expr.KFp16Value
import org.ksmt.expr.KFp32Value
import org.ksmt.expr.KFp64Value
import org.ksmt.expr.KFpAbsExpr
import org.ksmt.expr.KFpAddExpr
import org.ksmt.expr.KFpCustomSizeValue
import org.ksmt.expr.KFpDivExpr
import org.ksmt.expr.KFpEqualExpr
import org.ksmt.expr.KFpFromBvExpr
import org.ksmt.expr.KFpFusedMulAddExpr
import org.ksmt.expr.KFpGreaterExpr
import org.ksmt.expr.KFpGreaterOrEqualExpr
import org.ksmt.expr.KFpIsInfiniteExpr
import org.ksmt.expr.KFpIsNaNExpr
import org.ksmt.expr.KFpIsNegativeExpr
import org.ksmt.expr.KFpIsNormalExpr
import org.ksmt.expr.KFpIsPositiveExpr
import org.ksmt.expr.KFpIsSubnormalExpr
import org.ksmt.expr.KFpIsZeroExpr
import org.ksmt.expr.KFpLessExpr
import org.ksmt.expr.KFpLessOrEqualExpr
import org.ksmt.expr.KFpMaxExpr
import org.ksmt.expr.KFpMinExpr
import org.ksmt.expr.KFpMulExpr
import org.ksmt.expr.KFpNegationExpr
import org.ksmt.expr.KFpRemExpr
import org.ksmt.expr.KFpRoundToIntegralExpr
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.expr.KFpRoundingModeExpr
import org.ksmt.expr.KFpSqrtExpr
import org.ksmt.expr.KFpSubExpr
import org.ksmt.expr.KFpToBvExpr
import org.ksmt.expr.KFpToFpExpr
import org.ksmt.expr.KFpToIEEEBvExpr
import org.ksmt.expr.KFpToRealExpr
import org.ksmt.expr.KFpValue
import org.ksmt.expr.KFunctionApp
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.expr.KGeArithExpr
import org.ksmt.expr.KGtArithExpr
import org.ksmt.expr.KImpliesExpr
import org.ksmt.expr.KInt32NumExpr
import org.ksmt.expr.KInt64NumExpr
import org.ksmt.expr.KIntBigNumExpr
import org.ksmt.expr.KInterpretedValue
import org.ksmt.expr.KIsIntRealExpr
import org.ksmt.expr.KIteExpr
import org.ksmt.expr.KLeArithExpr
import org.ksmt.expr.KLtArithExpr
import org.ksmt.expr.KModIntExpr
import org.ksmt.expr.KMulArithExpr
import org.ksmt.expr.KNotExpr
import org.ksmt.expr.KOrBinaryExpr
import org.ksmt.expr.KOrExpr
import org.ksmt.expr.KPowerArithExpr
import org.ksmt.expr.KQuantifier
import org.ksmt.expr.KRealNumExpr
import org.ksmt.expr.KRealToFpExpr
import org.ksmt.expr.KRemIntExpr
import org.ksmt.expr.KSubArithExpr
import org.ksmt.expr.KToIntRealExpr
import org.ksmt.expr.KToRealIntExpr
import org.ksmt.expr.KTrue
import org.ksmt.expr.KUnaryMinusArithExpr
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.KXorExpr
import org.ksmt.solver.KSolverUnsupportedFeatureException
import org.ksmt.solver.bitwuzla.KBitwuzlaContext.Companion.mkTermCache
import org.ksmt.solver.bitwuzla.bindings.Bitwuzla
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaKind
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaRoundingMode
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaSort
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.solver.util.KExprLongInternalizerBase
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv16Sort
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBv64Sort
import org.ksmt.sort.KBv8Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFp128Sort
import org.ksmt.sort.KFp16Sort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KSortVisitor
import org.ksmt.sort.KUninterpretedSort
import java.math.BigInteger

@Suppress("LargeClass")
open class KBitwuzlaExprInternalizer(
    val bitwuzlaCtx: KBitwuzlaContext,
    private val scopedVars: Object2LongOpenHashMap<KDecl<*>>? = null,
    private val scopedExpressions: Object2LongOpenHashMap<KExpr<*>>? = null
) : KExprLongInternalizerBase() {

    @JvmField
    val bitwuzla: Bitwuzla = bitwuzlaCtx.bitwuzla

    open val sortInternalizer: SortInternalizer by lazy { SortInternalizer(bitwuzlaCtx) }
    open val functionSortInternalizer: FunctionSortInternalizer by lazy {
        FunctionSortInternalizer(bitwuzlaCtx, sortInternalizer)
    }

    override fun findInternalizedExpr(expr: KExpr<*>): BitwuzlaTerm {
        if (scopedExpressions != null) {
            return scopedExpressions.getLong(expr)
        }
        return bitwuzlaCtx.findExprTerm(expr)
    }

    override fun saveInternalizedExpr(expr: KExpr<*>, internalized: BitwuzlaTerm) {
        if (scopedExpressions != null) {
            scopedExpressions.put(expr, internalized)
        } else {
            saveExprInternalizationResult(expr, internalized)
        }
    }

    /**
    * Create Bitwuzla term from KSMT expression
    * */
    fun <T : KSort> KExpr<T>.internalize(): BitwuzlaTerm = tryInternalize({
        bitwuzlaCtx.ensureActive()
        internalizeExpr()
    }, rewriteWithAxiomsRequired = {
        // Expression axioms are not supported here
        throw KSolverUnsupportedFeatureException(it)
    })


    class AssertionWithAxioms(
        val assertion: BitwuzlaTerm,
        val axioms: List<BitwuzlaTerm>
    )

    /**
     * Some expressions are not representable in Bitwuzla, but in some cases
     * we can rewrite them using axioms preserving formula satisfiability.
     *
     * For example, there is no way to create [KFpToIEEEBvExpr] but we can create
     * a corresponding inverse function [KFpFromBvExpr]. We can replace an expression
     * of the form (fpToIEEEBv x) with a fresh variable y and add an axiom that
     * (= x (fpFromBv y)).
     * */
    fun KExpr<KBoolSort>.internalizeAssertion(): AssertionWithAxioms = tryInternalize({
        bitwuzlaCtx.ensureActive()

        // Try to internalize without axioms first, since it is the most common case.
        AssertionWithAxioms(
            assertion = internalizeExpr(),
            axioms = emptyList()
        )
    }, rewriteWithAxiomsRequired = {
        // We have an expression that can be rewritten using axioms

        // Reset internalizer
        exprStack.clear()

        // Rewrite whole assertion using axioms
        val rewriterWithAxioms = KBitwuzlaInternalizationAxioms(ctx)
        val exprWithAxioms = rewriterWithAxioms.rewriteWithAxioms(this)

        // Rerun internalizer
        AssertionWithAxioms(
            assertion = exprWithAxioms.expr.internalizeExpr(),
            axioms = exprWithAxioms.axioms.map { it.internalizeExpr() }
        )
    })

    /**
    * Create Bitwuzla sort from KSMT sort
    * */
    fun <T : KSort> T.internalizeSort(): BitwuzlaSort =
        sortInternalizer.internalizeSort(this)

    /**
    * Create Bitwuzla function sort for KSMT declaration.
     *
    * If [this] declaration is a constant then non-function sort is returned
    * */
    fun <T : KSort> KDecl<T>.bitwuzlaFunctionSort(): BitwuzlaSort =
        functionSortInternalizer.internalizeDeclSort(this)

    private fun saveExprInternalizationResult(expr: KExpr<*>, term: BitwuzlaTerm) {
        bitwuzlaCtx.saveExprTerm(expr, term)

        // Save only constants
        if (expr !is KInterpretedValue<*>) return

        val kind = Native.bitwuzlaTermGetBitwuzlaKind(term)

        /*
         * Save internalized values for [KBitwuzlaExprConverter] needs
         * @see [KBitwuzlaContext.saveInternalizedValue]
         */
        if (kind != BitwuzlaKind.BITWUZLA_KIND_VAL) return

        if (bitwuzlaCtx.convertValue(term) != null) return

        if (term != bitwuzlaCtx.trueTerm && term != bitwuzlaCtx.falseTerm) {
            bitwuzlaCtx.saveInternalizedValue(expr, term)
        }
    }

    private inline fun mkConstant(decl: KDecl<*>, sort: () -> BitwuzlaSort): BitwuzlaTerm {
        if (scopedVars != null) {
            val scopedVar = scopedVars.getLong(decl)
            if (scopedVar != NOT_INTERNALIZED) {
                return scopedVar
            }
        }
        return bitwuzlaCtx.mkConstant(decl, sort())
    }

    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> = with(expr) {
        transformList(args) { args: LongArray ->
            val const = mkConstant(decl) { decl.bitwuzlaFunctionSort() }

            val termArgs = args.addFirst(const)
            Native.bitwuzlaMkTerm(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_APPLY, termArgs)
        }
    }

    override fun <T : KSort> transform(expr: KConst<T>) = expr.transform {
        mkConstant(expr.decl) { expr.sort.internalizeSort() }
    }

    override fun transform(expr: KAndExpr) = with(expr) {
        transformList(args) { args: LongArray ->
            when (args.size) {
                0 -> bitwuzlaCtx.trueTerm
                1 -> args[0]
                else -> Native.bitwuzlaMkTerm(
                    bitwuzla, BitwuzlaKind.BITWUZLA_KIND_AND, args
                )
            }
        }
    }

    override fun transform(expr: KAndBinaryExpr) = with(expr) {
        transform(lhs, rhs, BitwuzlaKind.BITWUZLA_KIND_AND)
    }

    override fun transform(expr: KOrExpr) = with(expr) {
        transformList(args) { args: LongArray ->
            when (args.size) {
                0 -> bitwuzlaCtx.falseTerm
                1 -> args[0]
                else -> Native.bitwuzlaMkTerm(
                    bitwuzla, BitwuzlaKind.BITWUZLA_KIND_OR, args
                )
            }
        }
    }

    override fun transform(expr: KOrBinaryExpr) = with(expr) {
        transform(lhs, rhs, BitwuzlaKind.BITWUZLA_KIND_OR)
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
        transform(lhs, rhs) { l: BitwuzlaTerm, r: BitwuzlaTerm ->
            mkEqTerm(lhs.sort, l, r)
        }
    }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>) = with(expr) {
        transformList(args) { args: LongArray ->
            mkDistinctTerm(expr.args.first().sort, args)
        }
    }

    override fun <T : KSort> transform(expr: KIteExpr<T>) = with(expr) {
        transform(condition, trueBranch, falseBranch) { c: BitwuzlaTerm, t: BitwuzlaTerm, f: BitwuzlaTerm ->
            mkIteTerm(sort, c, t, f)
        }
    }

    private fun mkIteTerm(sort: KSort, c: BitwuzlaTerm, t: BitwuzlaTerm, f: BitwuzlaTerm): BitwuzlaTerm {
        if (sort is KArraySort<*, *>) {
            val tIsArray = Native.bitwuzlaTermIsArray(t)
            val fIsArray = Native.bitwuzlaTermIsArray(f)
            if (tIsArray != fIsArray) {
                return mkArrayLambdaTerm(sort.domain) { boundVar ->
                    val tValue = mkArraySelectTerm(tIsArray, t, boundVar)
                    val fValue = mkArraySelectTerm(fIsArray, f, boundVar)
                    Native.bitwuzlaMkTerm3(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_ITE, c, tValue, fValue)
                }
            }
        }
        return Native.bitwuzlaMkTerm3(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_ITE, c, t, f)
    }

    private fun mkEqTerm(sort: KSort, l: BitwuzlaTerm, r: BitwuzlaTerm): BitwuzlaTerm {
        if (sort is KArraySort<*, *>) {
            val lIsArray = Native.bitwuzlaTermIsArray(l)
            val rIsArray = Native.bitwuzlaTermIsArray(r)
            if (lIsArray != rIsArray) {
                val wrappingIsPossible = arrayAsFunctionWrappingIsPossible(lIsArray, l)
                        && arrayAsFunctionWrappingIsPossible(rIsArray, r)

                return if (wrappingIsPossible) {
                    val lFunction = wrapArrayAsFunction(sort, lIsArray, l)
                    val rFunction = wrapArrayAsFunction(sort, rIsArray, r)
                    Native.bitwuzlaMkTerm2(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_EQUAL, lFunction, rFunction)
                } else {
                    val lFunction = wrapArrayAsHighArityFunction(sort, lIsArray, l)
                    val rFunction = wrapArrayAsHighArityFunction(sort, rIsArray, r)
                    Native.bitwuzlaMkTerm2(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_EQUAL, lFunction, rFunction)
                }
            }
        }
        return Native.bitwuzlaMkTerm2(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_EQUAL, l, r)
    }

    private fun mkDistinctTerm(sort: KSort, args: LongArray): BitwuzlaTerm {
        if (sort is KArraySort<*, *>) {
            val allArgsAreArrays = args.map { Native.bitwuzlaTermIsArray(it) }
            val anyDifferentArgs = allArgsAreArrays.count { it }.let { it != 0 && it != allArgsAreArrays.size }
            if (anyDifferentArgs) {
                val wrappingIsPossible = args.zip(allArgsAreArrays)
                    .all { (arg, isArray) -> arrayAsFunctionWrappingIsPossible(isArray, arg) }

                return if (wrappingIsPossible) {
                    val functions = args.zip(allArgsAreArrays) { arg, isArray ->
                        wrapArrayAsFunction(sort, isArray, arg)
                    }
                    Native.bitwuzlaMkTerm(
                        bitwuzla, BitwuzlaKind.BITWUZLA_KIND_DISTINCT, functions.toLongArray()
                    )
                } else {
                    val functions = args.zip(allArgsAreArrays) { arg, isArray ->
                        wrapArrayAsHighArityFunction(sort, isArray, arg)
                    }
                    Native.bitwuzlaMkTerm(
                        bitwuzla, BitwuzlaKind.BITWUZLA_KIND_DISTINCT, functions.toLongArray()
                    )
                }
            }
        }

        return Native.bitwuzlaMkTerm(
            bitwuzla, BitwuzlaKind.BITWUZLA_KIND_DISTINCT, args
        )
    }

    private fun arrayAsFunctionWrappingIsPossible(isArray: Boolean, expr: BitwuzlaTerm): Boolean {
        // no wrapping needed
        if (!isArray) return true

        val exprKind = Native.bitwuzlaTermGetBitwuzlaKind(expr)

        // constant arrays can't be wrapped as lambda expressions
        return exprKind != BitwuzlaKind.BITWUZLA_KIND_CONST_ARRAY
    }

    private fun wrapArrayAsFunction(
        sort: KArraySort<*, *>,
        isArray: Boolean,
        expr: BitwuzlaTerm
    ): BitwuzlaTerm {
        // Expr is already a function
        if (!isArray) return expr

        return mkArrayLambdaTerm(sort.domain) { boundVar ->
            mkArraySelectTerm(isArray, expr, boundVar)
        }.also {
            check(!Native.bitwuzlaTermIsArray(it)) { "Array term was not eliminated" }
        }
    }

    /**
     * Since lambda expressions with constant body are always simplified
     * to constant arrays we have no normal way to replace constant array as function.
     *
     * Trick: replace constant array as 2-ary function, since it can't be rewritten
     * as constant array.
     * */
    private fun wrapArrayAsHighArityFunction(
        sort: KArraySort<*, *>,
        isArray: Boolean,
        expr: BitwuzlaTerm
    ): BitwuzlaTerm {
        return mkArrayLambdaTerm(listOf(sort.domain, sort.domain)) { (boundVar) ->
            mkArraySelectTerm(isArray, expr, boundVar)
        }.also {
            check(!Native.bitwuzlaTermIsArray(it)) { "Array term was not eliminated" }
        }
    }

    override fun transform(expr: KBitVec1Value) = with(expr) {
        transform { if (value) bitwuzlaCtx.trueTerm else bitwuzlaCtx.falseTerm }
    }

    override fun transform(expr: KBitVec8Value): KExpr<KBv8Sort> = transformBv32Number(expr)
    override fun transform(expr: KBitVec16Value): KExpr<KBv16Sort> = transformBv32Number(expr)
    override fun transform(expr: KBitVec32Value): KExpr<KBv32Sort> = transformBv32Number(expr)
    override fun transform(expr: KBitVec64Value): KExpr<KBv64Sort> = transformBv64Number(expr)

    fun <T : KBitVecNumberValue<S, *>, S : KBvSort> transformBv32Number(expr: T): T = with(expr) {
        transform {
            Native.bitwuzlaMkBvValueUint32(
                bitwuzla,
                sort.internalizeSort(),
                numberValue.toInt()
            ).also { bitwuzlaCtx.saveInternalizedValue(expr, it) }
        }
    }

    fun <T : KBitVecNumberValue<S, *>, S : KBvSort> transformBv64Number(expr: T): T = with(expr) {
        transform {
            transformBvLongNumber(numberValue.toLong(), sort.sizeBits.toInt())
                .also { bitwuzlaCtx.saveInternalizedValue(expr, it) }
        }
    }

    override fun transform(expr: KBitVecCustomValue) = with(expr) {
        transform {
            transformCustomBvNumber(value, expr.sizeBits.toInt())
                .also { bitwuzlaCtx.saveInternalizedValue(expr, it) }
        }
    }

    private fun transformBvLongNumber(value: Long, size: Int): BitwuzlaTerm {
        val intParts = intArrayOf((value ushr Int.SIZE_BITS).toInt(), value.toInt())
        return Native.bitwuzlaMkBvValueUint32Array(bitwuzla, size, intParts)
    }

    private fun transformCustomBvNumber(value: BigInteger, size: Int): BitwuzlaTerm =
        if (size <= Long.SIZE_BITS) {
            transformBvLongNumber(value.toLong(), size)
        } else {
            val intParts = bigIntegerToBvBits(value, size)
            Native.bitwuzlaMkBvValueUint32Array(bitwuzla, size, intParts)
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
                bitwuzla, BitwuzlaKind.BITWUZLA_KIND_BV_EXTRACT,
                arg,
                high, low
            )
        }
    }

    override fun transform(expr: KBvSignExtensionExpr) = with(expr) {
        transform(value) { arg: BitwuzlaTerm ->
            Native.bitwuzlaMkTerm1Indexed1(
                bitwuzla, BitwuzlaKind.BITWUZLA_KIND_BV_SIGN_EXTEND,
                arg,
                extensionSize
            )
        }
    }

    override fun transform(expr: KBvZeroExtensionExpr) = with(expr) {
        transform(value) { arg: BitwuzlaTerm ->
            Native.bitwuzlaMkTerm1Indexed1(
                bitwuzla, BitwuzlaKind.BITWUZLA_KIND_BV_ZERO_EXTEND,
                arg,
                extensionSize
            )
        }
    }

    override fun transform(expr: KBvRepeatExpr) = with(expr) {
        transform(value) { arg: BitwuzlaTerm ->
            Native.bitwuzlaMkTerm1Indexed1(
                bitwuzla, BitwuzlaKind.BITWUZLA_KIND_BV_REPEAT,
                arg,
                repeatNumber
            )
        }
    }

    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>) = with(expr) {
        transform(arg, shift, BitwuzlaKind.BITWUZLA_KIND_BV_SHL)
    }

    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>) = with(expr) {
        transform(arg, shift, BitwuzlaKind.BITWUZLA_KIND_BV_SHR)
    }

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>) = with(expr) {
        transform(arg, shift, BitwuzlaKind.BITWUZLA_KIND_BV_ASHR)
    }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>) = with(expr) {
        transform(arg, rotation, BitwuzlaKind.BITWUZLA_KIND_BV_ROL)
    }

    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>) = with(expr) {
        transform(arg, rotation, BitwuzlaKind.BITWUZLA_KIND_BV_ROR)
    }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>) = with(expr) {
        transform(value) { arg: BitwuzlaTerm ->
            Native.bitwuzlaMkTerm1Indexed1(
                bitwuzla, BitwuzlaKind.BITWUZLA_KIND_BV_ROLI,
                arg,
                rotationNumber
            )
        }
    }

    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>) = with(expr) {
        transform(value) { arg: BitwuzlaTerm ->
            Native.bitwuzlaMkTerm1Indexed1(
                bitwuzla, BitwuzlaKind.BITWUZLA_KIND_BV_RORI,
                arg,
                rotationNumber
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

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>) =
        TODO("no direct support for $expr")

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_SSUB_OVERFLOW)
    }

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>) =
        TODO("no direct support for $expr")

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>) = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_BV_SDIV_OVERFLOW)
    }

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>) =
        TODO("no direct support for $expr")

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>) = with(expr) {
        val kind = if (isSigned) {
            BitwuzlaKind.BITWUZLA_KIND_BV_SMUL_OVERFLOW
        } else {
            BitwuzlaKind.BITWUZLA_KIND_BV_UMUL_OVERFLOW
        }

        transform(arg0, arg1, kind)
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>) =
        TODO("no direct support for $expr")

    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>) = with(expr) {
        transform(array, index, value) { a: BitwuzlaTerm, i: BitwuzlaTerm, v: BitwuzlaTerm ->
            if (Native.bitwuzlaTermIsArray(a)) {
                Native.bitwuzlaMkTerm3(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_ARRAY_STORE, a, i, v)
            } else {
                mkArrayLambdaTerm(index.sort) { lambdaVar ->
                    // (store a i v) ==> (ite (= x i) v (select a x))
                    val nestedValue = mkArraySelectTerm(array = a, idx = lambdaVar, isArray = false)
                    val condition = Native.bitwuzlaMkTerm2(
                        bitwuzla, BitwuzlaKind.BITWUZLA_KIND_EQUAL, lambdaVar, i
                    )
                    Native.bitwuzlaMkTerm3(
                        bitwuzla, BitwuzlaKind.BITWUZLA_KIND_ITE, condition, v, nestedValue
                    )
                }
            }
        }
    }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Store<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = with(expr) {
        transform(
            array,
            index0,
            index1,
            value
        ) { a: BitwuzlaTerm, i0: BitwuzlaTerm, i1: BitwuzlaTerm, v: BitwuzlaTerm ->
            // (store a i j v) ==> (ite (and (= x0 i) (= x1 j)) v (select a x0 x1))
            mkArrayLambdaTerm(sort.domainSorts) { lambdaVars ->
                val selectArgs = lambdaVars.addFirst(a)
                val nestedValue = Native.bitwuzlaMkTerm(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_APPLY, selectArgs)

                val condition0 = Native.bitwuzlaMkTerm2(
                    bitwuzla, BitwuzlaKind.BITWUZLA_KIND_EQUAL, lambdaVars[0], i0
                )
                val condition1 = Native.bitwuzlaMkTerm2(
                    bitwuzla, BitwuzlaKind.BITWUZLA_KIND_EQUAL, lambdaVars[1], i1
                )
                val condition = Native.bitwuzlaMkTerm2(
                    bitwuzla, BitwuzlaKind.BITWUZLA_KIND_AND, condition0, condition1
                )

                Native.bitwuzlaMkTerm3(
                    bitwuzla, BitwuzlaKind.BITWUZLA_KIND_ITE, condition, v, nestedValue
                )
            }
        }
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Store<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> {
        val exprArgs = buildList {
            add(expr.array)
            add(expr.index0)
            add(expr.index1)
            add(expr.index2)
            add(expr.value)
        }
        return expr.transformList(exprArgs) { transformedArgs ->
            mkArrayStoreTerm(transformedArgs, expr.sort)
        }
    }

    override fun <R : KSort> transform(expr: KArrayNStore<R>): KExpr<KArrayNSort<R>> {
        val exprArgs = buildList {
            add(expr.value)
            addAll(expr.indices)
            add(expr.array)
        }

        return expr.transformList(exprArgs) { transformedArgs: LongArray ->
            mkArrayStoreTerm(transformedArgs, expr.sort)
        }
    }

    // args in the format: [array] + indices + [value]
    private fun mkArrayStoreTerm(args: LongArray, sort: KArraySortBase<*>): BitwuzlaTerm {
        val array: BitwuzlaTerm = args[0]
        val value: BitwuzlaTerm = args[args.lastIndex]

        return mkArrayLambdaTerm(sort.domainSorts) { lambdaVars: LongArray ->
            val selectArgs = lambdaVars.addFirst(array)
            val nestedValue = Native.bitwuzlaMkTerm(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_APPLY, selectArgs)

            val conditions = LongArray(lambdaVars.size) {
                val index = args[it + 1] // +1 for array argument
                val lambdaIndex = lambdaVars[it]

                Native.bitwuzlaMkTerm2(
                    bitwuzla, BitwuzlaKind.BITWUZLA_KIND_EQUAL, lambdaIndex, index
                )
            }
            val condition = Native.bitwuzlaMkTerm(
                bitwuzla, BitwuzlaKind.BITWUZLA_KIND_AND, conditions
            )

            Native.bitwuzlaMkTerm3(
                bitwuzla, BitwuzlaKind.BITWUZLA_KIND_ITE, condition, value, nestedValue
            )
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>) = with(expr) {
        transform(array, index) { a: BitwuzlaTerm, i: BitwuzlaTerm ->
            mkArraySelectTerm(Native.bitwuzlaTermIsArray(a), a, i)
        }
    }

    private fun mkArraySelectTerm(isArray: Boolean, array: BitwuzlaTerm, idx: BitwuzlaTerm): BitwuzlaTerm =
        if (isArray) {
            Native.bitwuzlaMkTerm2(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_ARRAY_SELECT, array, idx)
        } else {
            Native.bitwuzlaMkTerm2(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_APPLY, array, idx)
        }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Select<D0, D1, R>
    ): KExpr<R> = with(expr) {
        transform(array, index0, index1) { a: BitwuzlaTerm, i0: BitwuzlaTerm, i1: BitwuzlaTerm ->
            Native.bitwuzlaMkTerm3(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_APPLY, a, i0, i1)
        }
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Select<D0, D1, D2, R>
    ): KExpr<R> = with(expr) {
        transform(
            array,
            index0,
            index1,
            index2
        ) { a: BitwuzlaTerm, i0: BitwuzlaTerm, i1: BitwuzlaTerm, i2: BitwuzlaTerm ->
            val args = longArrayOf(a, i0, i1, i2)
            Native.bitwuzlaMkTerm(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_APPLY, args)
        }
    }

    override fun <R : KSort> transform(expr: KArrayNSelect<R>): KExpr<R> {
        val exprArgs = buildList {
            add(expr.array)
            addAll(expr.indices)
        }
        return expr.transformList(exprArgs) { args ->
            Native.bitwuzlaMkTerm(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_APPLY, args)
        }
    }

    override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KArrayConst<A, R>) = with(expr) {
        transform(value) { value: BitwuzlaTerm ->
            if (sort is KArraySort<*, *>) {
                Native.bitwuzlaMkConstArray(bitwuzla, sort.internalizeSort(), value)
            } else {
                mkArrayLambdaTerm(sort.domainSorts) { value }
            }
        }
    }

    override fun <D : KSort, R : KSort> transform(
        expr: KArrayLambda<D, R>
    ) = expr.transformArrayLambda()

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Lambda<D0, D1, R>
    ) = expr.transformArrayLambda()

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Lambda<D0, D1, D2, R>
    ) = expr.transformArrayLambda()

    override fun <R : KSort> transform(
        expr: KArrayNLambda<R>
    ) = expr.transformArrayLambda()

    private fun <E : KArrayLambdaBase<*, *>> E.transformArrayLambda(): E = transform {
        internalizeQuantifierBody(indexVarDeclarations, body) { internalizedBounds, internalizedBody ->
            mkLambdaTerm(internalizedBounds, internalizedBody)
        }
    }

    private inline fun mkArrayLambdaTerm(
        boundVarSort: KSort,
        body: (BitwuzlaTerm) -> BitwuzlaTerm
    ): BitwuzlaTerm {
        val boundDecl = bitwuzlaCtx.ctx.mkFreshConstDecl("x", boundVarSort)
        val boundVar = bitwuzlaCtx.mkVar(boundDecl, boundVarSort.internalizeSort())
        val lambdaBody = body(boundVar)
        return mkLambdaTerm(boundVar, lambdaBody)
    }

    private inline fun mkArrayLambdaTerm(
        bounds: List<KSort>,
        body: (LongArray) -> BitwuzlaTerm
    ): BitwuzlaTerm {
        val boundVars = LongArray(bounds.size) {
            val boundDecl = bitwuzlaCtx.ctx.mkFreshConstDecl("x", bounds[it])
            bitwuzlaCtx.mkVar(boundDecl, bounds[it].internalizeSort())
        }
        val lambdaBody = body(boundVars)
        return mkLambdaTerm(boundVars, lambdaBody)
    }

    private fun mkLambdaTerm(boundVar: BitwuzlaTerm, body: BitwuzlaTerm): BitwuzlaTerm =
        Native.bitwuzlaMkTerm2(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_LAMBDA, boundVar, body)

    private fun mkLambdaTerm(
        bounds: LongArray,
        body: BitwuzlaTerm
    ): BitwuzlaTerm {
        val args = bounds.addLast(body)
        return Native.bitwuzlaMkTerm(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_LAMBDA, args)
    }

    override fun transform(
        expr: KExistentialQuantifier
    ): KExpr<KBoolSort> = expr.internalizeQuantifier { args ->
        Native.bitwuzlaMkTerm(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_EXISTS, args)
    }

    override fun transform(
        expr: KUniversalQuantifier
    ): KExpr<KBoolSort> = expr.internalizeQuantifier { args ->
        Native.bitwuzlaMkTerm(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_FORALL, args)
    }

    override fun transform(expr: KBv2IntExpr): KExpr<KIntSort> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun <T : KFpSort> transform(expr: KRealToFpExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun transform(expr: KModIntExpr): KExpr<KIntSort> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun transform(expr: KRemIntExpr): KExpr<KIntSort> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun transform(expr: KToRealIntExpr): KExpr<KRealSort> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun transform(expr: KInt32NumExpr): KExpr<KIntSort> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun transform(expr: KInt64NumExpr): KExpr<KIntSort> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun transform(expr: KIntBigNumExpr): KExpr<KIntSort> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun transform(expr: KToIntRealExpr): KExpr<KIntSort> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun transform(expr: KIsIntRealExpr): KExpr<KBoolSort> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun transform(expr: KRealNumExpr): KExpr<KRealSort> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun <T : KArithSort> transform(expr: KAddArithExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun <T : KArithSort> transform(expr: KMulArithExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun <T : KArithSort> transform(expr: KSubArithExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun <T : KArithSort> transform(expr: KUnaryMinusArithExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun <T : KArithSort> transform(expr: KDivArithExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun <T : KArithSort> transform(expr: KPowerArithExpr<T>): KExpr<T> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun <T : KArithSort> transform(expr: KLtArithExpr<T>): KExpr<KBoolSort> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun <T : KArithSort> transform(expr: KLeArithExpr<T>): KExpr<KBoolSort> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun <T : KArithSort> transform(expr: KGtArithExpr<T>): KExpr<KBoolSort> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    override fun <T : KArithSort> transform(expr: KGeArithExpr<T>): KExpr<KBoolSort> {
        throw KSolverUnsupportedFeatureException("int and real theories are not supported in Bitwuzla")
    }

    private fun <T : KFpValue<*>> transformFpValue(expr: T): T = with(expr) {
        transform(biasedExponent, significand) { exponent: BitwuzlaTerm, significand: BitwuzlaTerm ->
            val sign = if (signBit) bitwuzlaCtx.trueTerm else bitwuzlaCtx.falseTerm
            Native.bitwuzlaMkFpValue(bitwuzla, sign, exponent, significand)
        }
    }

    override fun transform(expr: KFp16Value): KExpr<KFp16Sort> = transformFpValue(expr)

    override fun transform(expr: KFp32Value): KExpr<KFp32Sort> = transformFpValue(expr)

    override fun transform(expr: KFp64Value): KExpr<KFp64Sort> = transformFpValue(expr)

    override fun transform(expr: KFp128Value): KExpr<KFp128Sort> = transformFpValue(expr)

    override fun transform(expr: KFpCustomSizeValue): KExpr<KFpSort> = transformFpValue(expr)

    override fun transform(expr: KFpRoundingModeExpr): KExpr<KFpRoundingModeSort> = with(expr) {
        transform {
            val rmMode = when (value) {
                KFpRoundingMode.RoundNearestTiesToEven -> BitwuzlaRoundingMode.BITWUZLA_RM_RNE
                KFpRoundingMode.RoundNearestTiesToAway -> BitwuzlaRoundingMode.BITWUZLA_RM_RNA
                KFpRoundingMode.RoundTowardPositive -> BitwuzlaRoundingMode.BITWUZLA_RM_RTP
                KFpRoundingMode.RoundTowardNegative -> BitwuzlaRoundingMode.BITWUZLA_RM_RTN
                KFpRoundingMode.RoundTowardZero -> BitwuzlaRoundingMode.BITWUZLA_RM_RTZ
            }
            Native.bitwuzlaMkRmValue(bitwuzla, rmMode)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpAbsExpr<T>): KExpr<T> = with(expr) {
        transform(value, BitwuzlaKind.BITWUZLA_KIND_FP_ABS)
    }

    override fun <T : KFpSort> transform(expr: KFpNegationExpr<T>): KExpr<T> = with(expr) {
        transform(value, BitwuzlaKind.BITWUZLA_KIND_FP_NEG)
    }

    override fun <T : KFpSort> transform(expr: KFpAddExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_FP_ADD)
    }

    override fun <T : KFpSort> transform(expr: KFpSubExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_FP_SUB)
    }

    override fun <T : KFpSort> transform(expr: KFpMulExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_FP_MUL)
    }

    override fun <T : KFpSort> transform(expr: KFpDivExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_FP_DIV)
    }

    override fun <T : KFpSort> transform(expr: KFpFusedMulAddExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, arg0, arg1, arg2, BitwuzlaKind.BITWUZLA_KIND_FP_FMA)
    }

    override fun <T : KFpSort> transform(expr: KFpSqrtExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, value, BitwuzlaKind.BITWUZLA_KIND_FP_SQRT)
    }

    override fun <T : KFpSort> transform(expr: KFpRemExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_FP_REM)
    }

    override fun <T : KFpSort> transform(expr: KFpRoundToIntegralExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, value, BitwuzlaKind.BITWUZLA_KIND_FP_RTI)
    }

    override fun <T : KFpSort> transform(expr: KFpMinExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_FP_MIN)
    }

    override fun <T : KFpSort> transform(expr: KFpMaxExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_FP_MAX)
    }

    override fun <T : KFpSort> transform(expr: KFpLessOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_FP_LEQ)
    }

    override fun <T : KFpSort> transform(expr: KFpLessExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_FP_LT)
    }

    override fun <T : KFpSort> transform(expr: KFpGreaterOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_FP_GEQ)
    }

    override fun <T : KFpSort> transform(expr: KFpGreaterExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_FP_GT)
    }

    override fun <T : KFpSort> transform(expr: KFpEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1, BitwuzlaKind.BITWUZLA_KIND_FP_EQ)
    }

    override fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, BitwuzlaKind.BITWUZLA_KIND_FP_IS_NORMAL)
    }

    override fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, BitwuzlaKind.BITWUZLA_KIND_FP_IS_SUBNORMAL)
    }

    override fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, BitwuzlaKind.BITWUZLA_KIND_FP_IS_ZERO)
    }

    override fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, BitwuzlaKind.BITWUZLA_KIND_FP_IS_INF)
    }

    override fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, BitwuzlaKind.BITWUZLA_KIND_FP_IS_NAN)
    }

    override fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, BitwuzlaKind.BITWUZLA_KIND_FP_IS_NEG)
    }

    override fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value, BitwuzlaKind.BITWUZLA_KIND_FP_IS_POS)
    }

    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort> = with(expr) {
        transform(roundingMode, value) { rm: BitwuzlaTerm, value: BitwuzlaTerm ->
            val operation = if (isSigned) BitwuzlaKind.BITWUZLA_KIND_FP_TO_SBV else BitwuzlaKind.BITWUZLA_KIND_FP_TO_UBV
            Native.bitwuzlaMkTerm2Indexed1(bitwuzla, operation, rm, value, bvSize)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> {
        throw TryRewriteExpressionUsingAxioms("KFpToIEEEBvExpr is not supported in bitwuzla")
    }

    override fun <T : KFpSort> transform(expr: KFpToFpExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, value) { rm: BitwuzlaTerm, value: BitwuzlaTerm ->
            Native.bitwuzlaMkTerm2Indexed2(
                bitwuzla,
                BitwuzlaKind.BITWUZLA_KIND_FP_TO_FP_FROM_FP,
                rm,
                value,
                sort.exponentBits.toInt(),
                sort.significandBits.toInt()
            )
        }
    }

    override fun <T : KFpSort> transform(expr: KFpFromBvExpr<T>): KExpr<T> = with(expr) {
        transform(
            sign,
            biasedExponent,
            significand
        ) { sign: BitwuzlaTerm, exp: BitwuzlaTerm, significand: BitwuzlaTerm ->
            Native.bitwuzlaMkTerm3(
                bitwuzla, BitwuzlaKind.BITWUZLA_KIND_FP_FP, sign, exp, significand
            )
        }
    }

    override fun <T : KFpSort> transform(expr: KBvToFpExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, value) { rm: BitwuzlaTerm, value: BitwuzlaTerm ->
            val operation = if (signed) {
                BitwuzlaKind.BITWUZLA_KIND_FP_TO_FP_FROM_SBV
            } else {
                BitwuzlaKind.BITWUZLA_KIND_FP_TO_FP_FROM_UBV
            }
            Native.bitwuzlaMkTerm2Indexed2(
                bitwuzla, operation, rm, value, sort.exponentBits.toInt(), sort.significandBits.toInt()
            )
        }
    }

    override fun <A : KArraySortBase<R>, R : KSort> transform(
        expr: KFunctionAsArray<A, R>
    ): KExpr<A> = expr.transform {
        mkConstant(expr.function) { expr.function.bitwuzlaFunctionSort() }
    }

    private inline fun <T : KQuantifier> T.internalizeQuantifier(
        crossinline internalizer: T.(LongArray) -> BitwuzlaTerm
    ): T = transform {
        internalizeQuantifierBody(bounds, body) { internalizedBounds, internalizedBody ->
            if (internalizedBounds.isEmpty()) {
                internalizedBody
            } else {
                val args = internalizedBounds.addLast(internalizedBody)
                internalizer(args)
            }
        }
    }

    private inline fun internalizeQuantifierBody(
        bounds: List<KDecl<*>>,
        body: KExpr<*>,
        crossinline transformInternalized: (LongArray, BitwuzlaTerm) -> BitwuzlaTerm
    ): BitwuzlaTerm {
        if (bounds.any { it.hasUninterpretedSorts() }) {
            throw KSolverUnsupportedFeatureException("Bitwuzla doesn't support quantifiers with uninterpreted sorts")
        }

        val nestedVarScope = mkTermCache<KDecl<*>>().apply {
            scopedVars?.let { putAll(it) }
        }

        /**
         * Internalizer will produce var instead of normal constants
         * for all bound variables.
         * */
        val internalizedBounds = LongArray(bounds.size) { idx ->
            val boundDecl = bounds[idx]
            bitwuzlaCtx.mkVar(boundDecl, boundDecl.sort.internalizeSort()).also {
                nestedVarScope.put(boundDecl, it)
            }
        }

        /**
         * Internalizer will not use expressions cache because
         * bound variables may overlap with already cached constants
         * */
        val nestedExpressionCache = mkTermCache<KExpr<*>>()

        val bodyInternalizer = KBitwuzlaExprInternalizer(bitwuzlaCtx, nestedVarScope, nestedExpressionCache)
        val internalizedBody = with(bodyInternalizer) { body.internalize() }

        return transformInternalized(internalizedBounds, internalizedBody)
    }

    open class SortInternalizer(private val bitwuzlaCtx: KBitwuzlaContext) : KSortVisitor<Unit> {
        private var internalizedSort: BitwuzlaSort = NOT_INTERNALIZED

        override fun visit(sort: KBoolSort) {
            internalizedSort = bitwuzlaCtx.boolSort
        }

        override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>) {
            checkNoNestedArrays(sort.domain)
            checkNoNestedArrays(sort.range)

            val domain = internalizeSort(sort.domain)
            val range = internalizeSort(sort.range)

            internalizedSort = Native.bitwuzlaMkArraySort(bitwuzlaCtx.bitwuzla, domain, range)
        }

        override fun <D0 : KSort, D1 : KSort, R : KSort> visit(sort: KArray2Sort<D0, D1, R>) {
            checkNoNestedArrays(sort.domain0)
            checkNoNestedArrays(sort.domain1)
            checkNoNestedArrays(sort.range)

            val domain = longArrayOf(
                internalizeSort(sort.domain0),
                internalizeSort(sort.domain1)
            )
            val range = internalizeSort(sort.range)

            internalizedSort = Native.bitwuzlaMkFunSort(bitwuzlaCtx.bitwuzla, domain.size, domain, range)
        }

        override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(sort: KArray3Sort<D0, D1, D2, R>) {
            checkNoNestedArrays(sort.domain0)
            checkNoNestedArrays(sort.domain1)
            checkNoNestedArrays(sort.domain2)
            checkNoNestedArrays(sort.range)

            val domain = longArrayOf(
                internalizeSort(sort.domain0),
                internalizeSort(sort.domain1),
                internalizeSort(sort.domain2)
            )
            val range = internalizeSort(sort.range)

            internalizedSort = Native.bitwuzlaMkFunSort(bitwuzlaCtx.bitwuzla, domain.size, domain, range)
        }

        override fun <R : KSort> visit(sort: KArrayNSort<R>) {
            sort.domainSorts.forEach { checkNoNestedArrays(it) }
            checkNoNestedArrays(sort.range)

            val domain = sort.domainSorts.let { sorts ->
                LongArray(sorts.size) { internalizeSort(sorts[it]) }
            }
            val range = internalizeSort(sort.range)

            internalizedSort = Native.bitwuzlaMkFunSort(bitwuzlaCtx.bitwuzla, domain.size, domain, range)
        }

        private fun checkNoNestedArrays(sort: KSort) {
            if (sort is KArraySortBase<*>) {
                throw KSolverUnsupportedFeatureException("Bitwuzla doesn't support nested arrays")
            }
        }

        override fun <S : KBvSort> visit(sort: S) {
            val size = sort.sizeBits.toInt()

            internalizedSort = if (size == 1) {
                bitwuzlaCtx.boolSort
            } else {
                Native.bitwuzlaMkBvSort(bitwuzlaCtx.bitwuzla, size)
            }
        }

        /**
         * Bitwuzla doesn't support integers and reals.
         * */
        override fun visit(sort: KIntSort) =
            throw KSolverUnsupportedFeatureException("Unsupported sort $sort")

        override fun visit(sort: KRealSort) =
            throw KSolverUnsupportedFeatureException("Unsupported sort $sort")

        /**
         * Replace Uninterpreted sorts with (BitVec 32).
         * The sort universe size is limited by 2^32 values which should be enough.
         * */
        override fun visit(sort: KUninterpretedSort) {
            internalizedSort = Native.bitwuzlaMkBvSort(bitwuzlaCtx.bitwuzla, UNINTERPRETED_SORT_REPLACEMENT_BV_SIZE)
        }

        override fun <S : KFpSort> visit(sort: S) {
            internalizedSort = Native.bitwuzlaMkFpSort(
                bitwuzlaCtx.bitwuzla,
                expSize = sort.exponentBits.toInt(),
                sigSize = sort.significandBits.toInt()
            )
        }

        override fun visit(sort: KFpRoundingModeSort) {
            internalizedSort = Native.bitwuzlaMkRmSort(bitwuzlaCtx.bitwuzla)
        }

        fun internalizeSort(sort: KSort): BitwuzlaSort = bitwuzlaCtx.internalizeSort(sort) {
            sort.accept(this)
            internalizedSort
        }

        companion object {
            const val UNINTERPRETED_SORT_REPLACEMENT_BV_SIZE = 32
        }
    }

    open class FunctionSortInternalizer(
        private val bitwuzlaCtx: KBitwuzlaContext,
        private val sortInternalizer: SortInternalizer
    ) : KDeclVisitor<Unit> {
        private var declSort: BitwuzlaSort = NOT_INTERNALIZED

        override fun <S : KSort> visit(decl: KFuncDecl<S>) {
            if (decl.argSorts.any { it is KArraySortBase<*> }) {
                throw KSolverUnsupportedFeatureException("Bitwuzla doesn't support functions with arrays in domain")
            }

            if (decl.argSorts.isNotEmpty() && decl.sort is KArraySortBase<*>) {
                throw KSolverUnsupportedFeatureException("Bitwuzla doesn't support functions with arrays in range")
            }

            val domain = decl.argSorts.let { sorts ->
                LongArray(sorts.size) { sortInternalizer.internalizeSort(sorts[it]) }
            }
            val range = sortInternalizer.internalizeSort(decl.sort)

            declSort = if (domain.isEmpty()) {
                range
            } else {
                Native.bitwuzlaMkFunSort(bitwuzlaCtx.bitwuzla, domain.size, domain, range)
            }
        }

        override fun <S : KSort> visit(decl: KConstDecl<S>) {
            declSort = sortInternalizer.internalizeSort(decl.sort)
        }

        fun internalizeDeclSort(decl: KDecl<*>): BitwuzlaSort = bitwuzlaCtx.internalizeDeclSort(decl) {
            decl.accept(this)
            declSort
        }
    }

    fun <S : KExpr<*>> S.transform(
        arg: KExpr<*>,
        kind: BitwuzlaKind
    ): S = transform(arg) { a0: BitwuzlaTerm -> Native.bitwuzlaMkTerm1(bitwuzla, kind, a0) }


    fun <S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        kind: BitwuzlaKind
    ): S = transform(arg0, arg1) { a0: BitwuzlaTerm, a1: BitwuzlaTerm ->
        Native.bitwuzlaMkTerm2(bitwuzla, kind, a0, a1)
    }

    fun <S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        arg2: KExpr<*>,
        kind: BitwuzlaKind
    ): S = transform(arg0, arg1, arg2) { a0: BitwuzlaTerm, a1: BitwuzlaTerm, a2: BitwuzlaTerm ->
        Native.bitwuzlaMkTerm3(bitwuzla, kind, a0, a1, a2)
    }

    fun <S : KExpr<*>> S.transform(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        arg2: KExpr<*>,
        arg3: KExpr<*>,
        kind: BitwuzlaKind
    ): S = transform(arg0, arg1, arg2, arg3) { a0: BitwuzlaTerm, a1: BitwuzlaTerm, a2: BitwuzlaTerm, a3: BitwuzlaTerm ->
        val args = longArrayOf(a0, a1, a2, a3)
        Native.bitwuzlaMkTerm(bitwuzla, kind, args)
    }

    /**
     * Abort current internalization because we found an unsupported expression,
     * that can be rewritten using axioms.
     *
     * See [internalizeAssertion].
     * */
    class TryRewriteExpressionUsingAxioms(override val message: String) : Exception(message) {
        // Not a real exception -> we can avoid stacktrace collection
        override fun fillInStackTrace(): Throwable = this
    }

    private inline fun <T> tryInternalize(
        body: () -> T,
        rewriteWithAxiomsRequired: (String) -> T
    ): T = try {
        body()
    } catch (ex: TryRewriteExpressionUsingAxioms) {
        rewriteWithAxiomsRequired(ex.message)
    }

    private val uninterpretedSortsDetector = object : KSortVisitor<Boolean> {
        override fun visit(sort: KUninterpretedSort): Boolean = true

        override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): Boolean =
            sort.domain.accept(this) || sort.range.accept(this)

        override fun <D0 : KSort, D1 : KSort, R : KSort> visit(sort: KArray2Sort<D0, D1, R>): Boolean =
            sort.domain0.accept(this) || sort.domain1.accept(this) || sort.range.accept(this)

        override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(sort: KArray3Sort<D0, D1, D2, R>): Boolean =
            sort.domain0.accept(this) || sort.domain1.accept(this)
                || sort.domain2.accept(this) || sort.range.accept(this)

        override fun <R : KSort> visit(sort: KArrayNSort<R>): Boolean =
            sort.domainSorts.any { it.accept(this) } || sort.range.accept(this)

        override fun visit(sort: KBoolSort): Boolean = false
        override fun visit(sort: KIntSort): Boolean = false
        override fun visit(sort: KRealSort): Boolean = false
        override fun <S : KBvSort> visit(sort: S): Boolean = false
        override fun <S : KFpSort> visit(sort: S): Boolean = false
        override fun visit(sort: KFpRoundingModeSort): Boolean = false
    }

    private fun KDecl<*>.hasUninterpretedSorts(): Boolean =
        sort.accept(uninterpretedSortsDetector)

    private fun LongArray.addLast(element: Long): LongArray {
        val result = copyOf(size + 1)
        result[size] = element
        return result
    }

    private fun LongArray.addFirst(element: Long): LongArray {
        val result = LongArray(size + 1)
        copyInto(result, destinationOffset = 1)
        result[0] = element
        return result
    }
}
