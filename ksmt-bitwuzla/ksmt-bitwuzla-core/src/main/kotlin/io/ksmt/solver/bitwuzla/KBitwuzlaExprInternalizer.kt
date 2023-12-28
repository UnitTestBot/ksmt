package io.ksmt.solver.bitwuzla

import io.ksmt.decl.KConstDecl
import io.ksmt.decl.KDecl
import io.ksmt.decl.KDeclVisitor
import io.ksmt.decl.KFuncDecl
import io.ksmt.expr.KAddArithExpr
import io.ksmt.expr.KAndBinaryExpr
import io.ksmt.expr.KAndExpr
import io.ksmt.expr.KArray2Lambda
import io.ksmt.expr.KArray2Select
import io.ksmt.expr.KArray2Store
import io.ksmt.expr.KArray3Lambda
import io.ksmt.expr.KArray3Select
import io.ksmt.expr.KArray3Store
import io.ksmt.expr.KArrayConst
import io.ksmt.expr.KArrayLambda
import io.ksmt.expr.KArrayLambdaBase
import io.ksmt.expr.KArrayNLambda
import io.ksmt.expr.KArrayNSelect
import io.ksmt.expr.KArrayNStore
import io.ksmt.expr.KArraySelect
import io.ksmt.expr.KArrayStore
import io.ksmt.expr.KBitVec16Value
import io.ksmt.expr.KBitVec1Value
import io.ksmt.expr.KBitVec32Value
import io.ksmt.expr.KBitVec64Value
import io.ksmt.expr.KBitVec8Value
import io.ksmt.expr.KBitVecCustomValue
import io.ksmt.expr.KBitVecNumberValue
import io.ksmt.expr.KBv2IntExpr
import io.ksmt.expr.KBvAddExpr
import io.ksmt.expr.KBvAddNoOverflowExpr
import io.ksmt.expr.KBvAddNoUnderflowExpr
import io.ksmt.expr.KBvAndExpr
import io.ksmt.expr.KBvArithShiftRightExpr
import io.ksmt.expr.KBvConcatExpr
import io.ksmt.expr.KBvDivNoOverflowExpr
import io.ksmt.expr.KBvExtractExpr
import io.ksmt.expr.KBvLogicalShiftRightExpr
import io.ksmt.expr.KBvMulExpr
import io.ksmt.expr.KBvMulNoOverflowExpr
import io.ksmt.expr.KBvMulNoUnderflowExpr
import io.ksmt.expr.KBvNAndExpr
import io.ksmt.expr.KBvNegNoOverflowExpr
import io.ksmt.expr.KBvNegationExpr
import io.ksmt.expr.KBvNorExpr
import io.ksmt.expr.KBvNotExpr
import io.ksmt.expr.KBvOrExpr
import io.ksmt.expr.KBvReductionAndExpr
import io.ksmt.expr.KBvReductionOrExpr
import io.ksmt.expr.KBvRepeatExpr
import io.ksmt.expr.KBvRotateLeftExpr
import io.ksmt.expr.KBvRotateLeftIndexedExpr
import io.ksmt.expr.KBvRotateRightExpr
import io.ksmt.expr.KBvRotateRightIndexedExpr
import io.ksmt.expr.KBvShiftLeftExpr
import io.ksmt.expr.KBvSignExtensionExpr
import io.ksmt.expr.KBvSignedDivExpr
import io.ksmt.expr.KBvSignedGreaterExpr
import io.ksmt.expr.KBvSignedGreaterOrEqualExpr
import io.ksmt.expr.KBvSignedLessExpr
import io.ksmt.expr.KBvSignedLessOrEqualExpr
import io.ksmt.expr.KBvSignedModExpr
import io.ksmt.expr.KBvSignedRemExpr
import io.ksmt.expr.KBvSubExpr
import io.ksmt.expr.KBvSubNoOverflowExpr
import io.ksmt.expr.KBvSubNoUnderflowExpr
import io.ksmt.expr.KBvToFpExpr
import io.ksmt.expr.KBvUnsignedDivExpr
import io.ksmt.expr.KBvUnsignedGreaterExpr
import io.ksmt.expr.KBvUnsignedGreaterOrEqualExpr
import io.ksmt.expr.KBvUnsignedLessExpr
import io.ksmt.expr.KBvUnsignedLessOrEqualExpr
import io.ksmt.expr.KBvUnsignedRemExpr
import io.ksmt.expr.KBvXNorExpr
import io.ksmt.expr.KBvXorExpr
import io.ksmt.expr.KBvZeroExtensionExpr
import io.ksmt.expr.KConst
import io.ksmt.expr.KDistinctExpr
import io.ksmt.expr.KDivArithExpr
import io.ksmt.expr.KEqExpr
import io.ksmt.expr.KExistentialQuantifier
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFalse
import io.ksmt.expr.KFp128Value
import io.ksmt.expr.KFp16Value
import io.ksmt.expr.KFp32Value
import io.ksmt.expr.KFp64Value
import io.ksmt.expr.KFpAbsExpr
import io.ksmt.expr.KFpAddExpr
import io.ksmt.expr.KFpCustomSizeValue
import io.ksmt.expr.KFpDivExpr
import io.ksmt.expr.KFpEqualExpr
import io.ksmt.expr.KFpFromBvExpr
import io.ksmt.expr.KFpFusedMulAddExpr
import io.ksmt.expr.KFpGreaterExpr
import io.ksmt.expr.KFpGreaterOrEqualExpr
import io.ksmt.expr.KFpIsInfiniteExpr
import io.ksmt.expr.KFpIsNaNExpr
import io.ksmt.expr.KFpIsNegativeExpr
import io.ksmt.expr.KFpIsNormalExpr
import io.ksmt.expr.KFpIsPositiveExpr
import io.ksmt.expr.KFpIsSubnormalExpr
import io.ksmt.expr.KFpIsZeroExpr
import io.ksmt.expr.KFpLessExpr
import io.ksmt.expr.KFpLessOrEqualExpr
import io.ksmt.expr.KFpMaxExpr
import io.ksmt.expr.KFpMinExpr
import io.ksmt.expr.KFpMulExpr
import io.ksmt.expr.KFpNegationExpr
import io.ksmt.expr.KFpRemExpr
import io.ksmt.expr.KFpRoundToIntegralExpr
import io.ksmt.expr.KFpRoundingMode
import io.ksmt.expr.KFpRoundingModeExpr
import io.ksmt.expr.KFpSqrtExpr
import io.ksmt.expr.KFpSubExpr
import io.ksmt.expr.KFpToBvExpr
import io.ksmt.expr.KFpToFpExpr
import io.ksmt.expr.KFpToIEEEBvExpr
import io.ksmt.expr.KFpToRealExpr
import io.ksmt.expr.KFpValue
import io.ksmt.expr.KFunctionApp
import io.ksmt.expr.KFunctionAsArray
import io.ksmt.expr.KGeArithExpr
import io.ksmt.expr.KGtArithExpr
import io.ksmt.expr.KImpliesExpr
import io.ksmt.expr.KInt32NumExpr
import io.ksmt.expr.KInt64NumExpr
import io.ksmt.expr.KIntBigNumExpr
import io.ksmt.expr.KInterpretedValue
import io.ksmt.expr.KIsIntRealExpr
import io.ksmt.expr.KIteExpr
import io.ksmt.expr.KLeArithExpr
import io.ksmt.expr.KLtArithExpr
import io.ksmt.expr.KModIntExpr
import io.ksmt.expr.KMulArithExpr
import io.ksmt.expr.KNotExpr
import io.ksmt.expr.KOrBinaryExpr
import io.ksmt.expr.KOrExpr
import io.ksmt.expr.KPowerArithExpr
import io.ksmt.expr.KQuantifier
import io.ksmt.expr.KRealNumExpr
import io.ksmt.expr.KRealToFpExpr
import io.ksmt.expr.KRemIntExpr
import io.ksmt.expr.KSubArithExpr
import io.ksmt.expr.KToIntRealExpr
import io.ksmt.expr.KToRealIntExpr
import io.ksmt.expr.KTrue
import io.ksmt.expr.KUnaryMinusArithExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.expr.KUniversalQuantifier
import io.ksmt.expr.KXorExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvAddNoUnderflowExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvMulNoUnderflowExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvNegNoOverflowExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvSubNoUnderflowExpr
import io.ksmt.solver.KSolverUnsupportedFeatureException
import org.ksmt.solver.bitwuzla.bindings.Bitwuzla
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaKind
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaRoundingMode
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaSort
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native
import io.ksmt.solver.util.KExprLongInternalizerBase
import io.ksmt.sort.KArithSort
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBv16Sort
import io.ksmt.sort.KBv32Sort
import io.ksmt.sort.KBv64Sort
import io.ksmt.sort.KBv8Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFp128Sort
import io.ksmt.sort.KFp16Sort
import io.ksmt.sort.KFp32Sort
import io.ksmt.sort.KFp64Sort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KSortVisitor
import io.ksmt.sort.KUninterpretedSort
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTermArray
import java.math.BigInteger

@Suppress("LargeClass")
open class KBitwuzlaExprInternalizer(val bitwuzlaCtx: KBitwuzlaContext) : KExprLongInternalizerBase() {

    @JvmField
    val bitwuzla: Bitwuzla = bitwuzlaCtx.bitwuzla

    open val sortInternalizer: SortInternalizer by lazy { SortInternalizer(bitwuzlaCtx) }
    open val functionSortInternalizer: FunctionSortInternalizer by lazy {
        FunctionSortInternalizer(bitwuzlaCtx, sortInternalizer)
    }

    private val quantifiedDeclarationsScopeOwner = arrayListOf<KExpr<*>>()
    private val quantifiedDeclarationsScope = arrayListOf<Set<KDecl<*>>?>()
    private var quantifiedDeclarations: Set<KDecl<*>>? = null

    override fun findInternalizedExpr(expr: KExpr<*>): BitwuzlaTerm {
        return bitwuzlaCtx.findExprTerm(expr)
    }

    override fun saveInternalizedExpr(expr: KExpr<*>, internalized: BitwuzlaTerm) {
        saveExprInternalizationResult(expr, internalized)
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

        resetInternalizer()

        // Rewrite whole assertion using axioms
        val rewriterWithAxioms = KBitwuzlaInternalizationAxioms(ctx)
        val exprWithAxioms = rewriterWithAxioms.rewriteWithAxioms(this)

        // Rerun internalizer
        AssertionWithAxioms(
            assertion = exprWithAxioms.expr.internalizeExpr(),
            axioms = exprWithAxioms.axioms.map { it.internalizeExpr() }
        )
    })

    private fun resetInternalizer() {
        exprStack.clear()

        quantifiedDeclarations = null
        quantifiedDeclarationsScope.clear()
        quantifiedDeclarationsScopeOwner.clear()
    }

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

        // Don't reverse cache uninterpreted values because we represent them as Bv32
        if (expr is KUninterpretedSortValue) return

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
        val isQuantified = quantifiedDeclarations?.contains(decl) ?: false
        return bitwuzlaCtx.mkConstant(decl, sort(), isQuantified)
    }

    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> = with(expr) {
        transformList(args) { args: BitwuzlaTermArray ->
            val const = mkConstant(decl) { decl.bitwuzlaFunctionSort() }

            val termArgs = args.addFirst(const)
            Native.bitwuzlaMkTerm(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_APPLY, termArgs)
        }
    }

    override fun <T : KSort> transform(expr: KConst<T>) = expr.transform {
        mkConstant(expr.decl) { expr.sort.internalizeSort() }
    }

    override fun transform(expr: KAndExpr) = with(expr) {
        transformList(args, ::mkAndTerm)
    }

    fun mkAndTerm(args: BitwuzlaTermArray): BitwuzlaTerm = when (args.size) {
        0 -> bitwuzlaCtx.trueTerm
        1 -> args[0]
        else -> Native.bitwuzlaMkTerm(
            bitwuzla, BitwuzlaKind.BITWUZLA_KIND_AND, args
        )
    }

    override fun transform(expr: KAndBinaryExpr) = with(expr) {
        transform(lhs, rhs, BitwuzlaKind.BITWUZLA_KIND_AND)
    }

    override fun transform(expr: KOrExpr) = with(expr) {
        transformList(args, ::mkOrTerm)
    }

    fun mkOrTerm(args: BitwuzlaTermArray): BitwuzlaTerm = when (args.size) {
        0 -> bitwuzlaCtx.falseTerm
        1 -> args[0]
        else -> Native.bitwuzlaMkTerm(
            bitwuzla, BitwuzlaKind.BITWUZLA_KIND_OR, args
        )
    }

    override fun transform(expr: KOrBinaryExpr) = with(expr) {
        transform(lhs, rhs, BitwuzlaKind.BITWUZLA_KIND_OR)
    }

    override fun transform(expr: KNotExpr) = with(expr) {
        transform(arg, ::mkNotTerm)
    }

    fun mkNotTerm(arg: BitwuzlaTerm): BitwuzlaTerm =
        Native.bitwuzlaMkTerm1(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_NOT, arg)

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
            return mkArrayEqTerm(sort, l, lIsArray, r, rIsArray)
        }
        return Native.bitwuzlaMkTerm2(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_EQUAL, l, r)
    }

    private fun mkDistinctTerm(sort: KSort, args: LongArray): BitwuzlaTerm {
        if (sort is KArraySort<*, *>) {
            return blastArrayDistinct(sort, args)
        }

        return Native.bitwuzlaMkTerm(
            bitwuzla, BitwuzlaKind.BITWUZLA_KIND_DISTINCT, args
        )
    }

    private fun mkArrayEqTerm(
        sort: KArraySort<*, *>,
        lhs: BitwuzlaTerm, lhsIsArray: Boolean,
        rhs: BitwuzlaTerm, rhsIsArray: Boolean
    ): BitwuzlaTerm {
        if (lhsIsArray == rhsIsArray) {
            return Native.bitwuzlaMkTerm2(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_EQUAL, lhs, rhs)
        }

        return if (lhsIsArray) {
            mkArrayEqFunctionTerm(sort, rhs, lhs)
        } else {
            mkArrayEqFunctionTerm(sort, lhs, rhs)
        }
    }

    private fun mkArrayEqFunctionTerm(
        sort: KArraySort<*, *>,
        functionTerm: BitwuzlaTerm,
        arrayTerm: BitwuzlaTerm
    ): BitwuzlaTerm {
        val arrayKind = Native.bitwuzlaTermGetBitwuzlaKind(arrayTerm)
        if (arrayKind == BitwuzlaKind.BITWUZLA_KIND_CONST) {
            // It is incorrect to use array and function, but equality should work
            return Native.bitwuzlaMkTerm2(
                bitwuzla,
                BitwuzlaKind.BITWUZLA_KIND_EQUAL,
                arrayTerm,
                functionTerm
            )
        }

        val wrappedArray = mkArrayLambdaTerm(sort.domain) { boundVar ->
            mkArraySelectTerm(isArray = true, array = arrayTerm, idx = boundVar)
        }.also {
            check(!Native.bitwuzlaTermIsArray(it)) { "Array term was not eliminated" }
        }
        return Native.bitwuzlaMkTerm2(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_EQUAL, wrappedArray, functionTerm)
    }

    private fun blastArrayDistinct(sort: KArraySort<*, *>, arrays: LongArray): BitwuzlaTerm {
        val inequalities = mutableListOf<Long>()

        for (i in arrays.indices) {
            val arrayI = arrays[i]
            val isArrayI = Native.bitwuzlaTermIsArray(arrayI)
            for (j in (i + 1) until arrays.size) {
                val arrayJ = arrays[j]
                val isArrayJ = Native.bitwuzlaTermIsArray(arrayJ)

                val equality = mkArrayEqTerm(sort, arrayI, isArrayI, arrayJ, isArrayJ)
                val inequality = Native.bitwuzlaMkTerm1(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_NOT, equality)

                inequalities += inequality
            }
        }

        return mkAndTerm(inequalities.toLongArray())
    }

    override fun transform(expr: KBitVec1Value) = with(expr) {
        transform { if (value) bitwuzlaCtx.trueTerm else bitwuzlaCtx.falseTerm }
    }

    override fun transform(expr: KBitVec8Value): KExpr<KBv8Sort> = transformBv32Number(expr, expr.byteValue.toInt())
    override fun transform(expr: KBitVec16Value): KExpr<KBv16Sort> = transformBv32Number(expr, expr.shortValue.toInt())
    override fun transform(expr: KBitVec32Value): KExpr<KBv32Sort> = transformBv32Number(expr, expr.intValue)
    override fun transform(expr: KBitVec64Value): KExpr<KBv64Sort> = transformBv64Number(expr, expr.longValue)

    fun <T : KBitVecNumberValue<S, *>, S : KBvSort> transformBv32Number(expr: T, value: Int): T = with(expr) {
        transform {
            Native.bitwuzlaMkBvValueUint32(bitwuzla, sort.internalizeSort(), value)
                .also { bitwuzlaCtx.saveInternalizedValue(expr, it) }
        }
    }

    fun <T : KBitVecNumberValue<S, *>, S : KBvSort> transformBv64Number(expr: T, value: Long): T = with(expr) {
        transform {
            transformBvLongNumber(value, sort.sizeBits.toInt())
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
        transform(arg0, arg1) { a0: BitwuzlaTerm, a1: BitwuzlaTerm ->
            if (isSigned) {
                mkBvAddSignedNoOverflowTerm(arg0.sort.sizeBits.toInt(), a0, a1, BvOverflowCheckMode.OVERFLOW)
            } else {
                val overflowCheck = Native.bitwuzlaMkTerm2(
                    bitwuzla, BitwuzlaKind.BITWUZLA_KIND_BV_UADD_OVERFLOW, a0, a1
                )
                mkNotTerm(overflowCheck)
            }
        }
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>) = with(expr) {
        transform(arg0, arg1) { a0: BitwuzlaTerm, a1: BitwuzlaTerm ->
            mkBvAddSignedNoOverflowTerm(arg0.sort.sizeBits.toInt(), a0, a1, BvOverflowCheckMode.UNDERFLOW)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>) = with(expr) {
        transform(arg0, arg1) { a0: BitwuzlaTerm, a1: BitwuzlaTerm ->
            mkBvSubSignedNoOverflowTerm(arg0.sort.sizeBits.toInt(), a0, a1, BvOverflowCheckMode.OVERFLOW)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>) = with(expr) {
        if (isSigned) {
            transform(arg0, arg1) { a0: BitwuzlaTerm, a1: BitwuzlaTerm ->
                mkBvSubSignedNoOverflowTerm(arg0.sort.sizeBits.toInt(), a0, a1, BvOverflowCheckMode.UNDERFLOW)
            }
        } else {
            transform {
                ctx.rewriteBvSubNoUnderflowExpr(arg0, arg1, isSigned = false).internalizeExpr()
            }
        }
    }

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>) = with(expr) {
        transform(arg0, arg1) { a0: BitwuzlaTerm, a1: BitwuzlaTerm ->
            val overflowCheck = Native.bitwuzlaMkTerm2(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_BV_SDIV_OVERFLOW, a0, a1)
            mkNotTerm(overflowCheck)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>) = with(expr) {
        transform {
            ctx.rewriteBvNegNoOverflowExpr(value).internalizeExpr()
        }
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>) = with(expr) {
        transform(arg0, arg1) { a0: BitwuzlaTerm, a1: BitwuzlaTerm ->
            if (isSigned) {
                mkBvMulSignedNoOverflowTerm(arg0.sort.sizeBits.toInt(), a0, a1, BvOverflowCheckMode.OVERFLOW)
            } else {
                val overflowCheck = Native.bitwuzlaMkTerm2(
                    bitwuzla, BitwuzlaKind.BITWUZLA_KIND_BV_UMUL_OVERFLOW, a0, a1
                )
                mkNotTerm(overflowCheck)
            }
        }
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>) = with(expr) {
        transform(arg0, arg1) { a0: BitwuzlaTerm, a1: BitwuzlaTerm ->
            mkBvMulSignedNoOverflowTerm(arg0.sort.sizeBits.toInt(), a0, a1, BvOverflowCheckMode.UNDERFLOW)
        }
    }

    /**
     * Signed Bv overflow check mode.
     * [OVERFLOW] --- the value is greater than max supported value.
     * [UNDERFLOW] --- the value is lower than min supported value.
     * */
    private enum class BvOverflowCheckMode {
        OVERFLOW,
        UNDERFLOW
    }

    private fun mkBvAddSignedNoOverflowTerm(
        sizeBits: Int,
        a0: BitwuzlaTerm,
        a1: BitwuzlaTerm,
        mode: BvOverflowCheckMode
    ): BitwuzlaTerm = mkBvSignedOverflowCheck(
        sizeBits,
        a0,
        a1,
        BitwuzlaKind.BITWUZLA_KIND_BV_SADD_OVERFLOW
    ) { a0Sign, a1Sign ->
        if (mode == BvOverflowCheckMode.OVERFLOW) {
            // Both positive
            mkAndTerm(longArrayOf(mkNotTerm(a0Sign), mkNotTerm(a1Sign)))
        } else {
            // Both negative
            mkAndTerm(longArrayOf(a0Sign, a1Sign))
        }
    }

    private fun mkBvSubSignedNoOverflowTerm(
        sizeBits: Int,
        a0: BitwuzlaTerm,
        a1: BitwuzlaTerm,
        mode: BvOverflowCheckMode
    ): BitwuzlaTerm = mkBvSignedOverflowCheck(
        sizeBits,
        a0,
        a1,
        BitwuzlaKind.BITWUZLA_KIND_BV_SSUB_OVERFLOW
    ) { a0Sign, a1Sign ->
        if (mode == BvOverflowCheckMode.OVERFLOW) {
            // Positive sub negative
            mkAndTerm(longArrayOf(mkNotTerm(a0Sign), a1Sign))
        } else {
            // Negative sub positive
            mkAndTerm(longArrayOf(a0Sign, mkNotTerm(a1Sign)))
        }
    }

    private fun mkBvMulSignedNoOverflowTerm(
        sizeBits: Int,
        a0: BitwuzlaTerm,
        a1: BitwuzlaTerm,
        mode: BvOverflowCheckMode
    ): BitwuzlaTerm = mkBvSignedOverflowCheck(
        sizeBits,
        a0,
        a1,
        BitwuzlaKind.BITWUZLA_KIND_BV_SMUL_OVERFLOW
    ) { a0Sign, a1Sign ->
        if (mode == BvOverflowCheckMode.OVERFLOW) {
            // Overflow is possible when sign bits are equal
            mkEqTerm(bitwuzlaCtx.ctx.boolSort, a0Sign, a1Sign)
        } else {
            // Underflow is possible when sign bits are different
            mkNotTerm(mkEqTerm(bitwuzlaCtx.ctx.boolSort, a0Sign, a1Sign))
        }
    }

    /**
     * Bitwuzla doesn't distinguish between signed overflow and underflow.
     * We perform sign check to cancel out false overflows.
     * */
    private inline fun mkBvSignedOverflowCheck(
        sizeBits: Int,
        a0: BitwuzlaTerm,
        a1: BitwuzlaTerm,
        checkKind: BitwuzlaKind,
        checkSign: (BitwuzlaTerm, BitwuzlaTerm) -> BitwuzlaTerm
    ): BitwuzlaTerm {
        val overflowCheck = Native.bitwuzlaMkTerm2(bitwuzla, checkKind, a0, a1)

        val a0Sign = mkBvSignTerm(sizeBits, a0)
        val a1Sign = mkBvSignTerm(sizeBits, a1)
        val signCheck = checkSign(a0Sign, a1Sign)

        val overflow = mkAndTerm(longArrayOf(overflowCheck, signCheck))
        return mkNotTerm(overflow)
    }

    private fun mkBvSignTerm(sizeBits: Int, bvExpr: BitwuzlaTerm): BitwuzlaTerm {
        return Native.bitwuzlaMkTerm1Indexed2(
            bitwuzla, BitwuzlaKind.BITWUZLA_KIND_BV_EXTRACT,
            bvExpr,
            sizeBits - 1, sizeBits - 1
        )
    }

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
            add(expr.array)
            addAll(expr.indices)
            add(expr.value)
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
            mkArrayConstTerm(sort, value)
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

    private fun <E : KArrayLambdaBase<*, *>> E.transformArrayLambda(): E =
        internalizeQuantifierBody(
            bounds = indexVarDeclarations,
            body = body,
            transformInternalizedWithoutQuantifiedVars = { internalizedBody ->
                mkArrayConstTerm(sort, internalizedBody)
            },
            transformInternalizedWithQuantifiedVars = { internalizedBounds, internalizedBody ->
                mkLambdaTerm(internalizedBounds, internalizedBody)
            }
        )

    private fun mkArrayConstTerm(
        sort: KArraySortBase<*>,
        value: BitwuzlaTerm
    ): BitwuzlaTerm {
        if (sort is KArraySort<*, *>) {
            val internalizedSort = sort.internalizeSort()
            return Native.bitwuzlaMkConstArray(bitwuzla, internalizedSort, value)
        }

        return mkArrayLambdaTerm(sort.domainSorts) { value }
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

    override fun transform(expr: KUninterpretedSortValue): KExpr<KUninterpretedSort> = expr.transform {
        Native.bitwuzlaMkBvValueUint32(
            bitwuzla,
            expr.sort.internalizeSort(),
            expr.valueIdx
        )
    }

    private inline fun <T : KQuantifier> T.internalizeQuantifier(
        crossinline mkQuantifierTerm: (LongArray) -> BitwuzlaTerm
    ): T {
        if (bounds.any { it.hasUninterpretedSorts() }) {
            throw KSolverUnsupportedFeatureException("Bitwuzla doesn't support quantifiers with uninterpreted sorts")
        }

        return internalizeQuantifierBody(
            bounds = bounds,
            body = body,
            transformInternalizedWithoutQuantifiedVars = { internalizedBody -> internalizedBody },
            transformInternalizedWithQuantifiedVars = { internalizedBounds, internalizedBody ->
                if (internalizedBounds.isEmpty()) {
                    internalizedBody
                } else {
                    val args = internalizedBounds.addLast(internalizedBody)
                    mkQuantifierTerm(args)
                }
            }
        )
    }

    private fun pushQuantifiedDeclScope(owner: KExpr<*>, quantifiedDecls: List<KDecl<*>>) {
        if (owner == quantifiedDeclarationsScopeOwner.lastOrNull()) return

        quantifiedDeclarationsScopeOwner.add(owner)
        quantifiedDeclarationsScope.add(quantifiedDeclarations)

        val newQuantifiedDecls = quantifiedDeclarations?.toHashSet() ?: hashSetOf()
        newQuantifiedDecls.addAll(quantifiedDecls)
        quantifiedDeclarations = newQuantifiedDecls
    }

    private fun popQuantifiedDeclScope() {
        quantifiedDeclarationsScopeOwner.removeLast()
        quantifiedDeclarations = quantifiedDeclarationsScope.removeLast()
    }

    private inline fun <T : KExpr<*>> T.internalizeQuantifierBody(
        bounds: List<KDecl<*>>,
        body: KExpr<*>,
        crossinline transformInternalizedWithoutQuantifiedVars: (BitwuzlaTerm) -> BitwuzlaTerm,
        crossinline transformInternalizedWithQuantifiedVars: (LongArray, BitwuzlaTerm) -> BitwuzlaTerm
    ): T {
        pushQuantifiedDeclScope(this, bounds)
        return transform(body) { internalizedBody ->
            popQuantifiedDeclScope()

            val boundConstants = LongArray(bounds.size)
            val boundVars = LongArray(bounds.size)
            for (idx in bounds.indices) {
                val boundDecl = bounds[idx]
                val boundSort = boundDecl.sort.internalizeSort()

                boundConstants[idx] = bitwuzlaCtx.mkConstant(boundDecl, boundSort, isQuantifiedConstant = true)
                boundVars[idx] = bitwuzlaCtx.mkVar(boundDecl, boundSort)
            }

            val internalizedBodyWithVars = Native.bitwuzlaSubstituteTerm(
                bitwuzla, internalizedBody, boundConstants, boundVars
            )

            /**
             * Body has not changed after substitution => quantified vars do not occur in the body
             * */
            if (internalizedBodyWithVars == internalizedBody) {
                transformInternalizedWithoutQuantifiedVars(internalizedBodyWithVars)
            } else {
                transformInternalizedWithQuantifiedVars(boundVars, internalizedBodyWithVars)
            }
        }
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
    } finally {
        resetInternalizer()
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
