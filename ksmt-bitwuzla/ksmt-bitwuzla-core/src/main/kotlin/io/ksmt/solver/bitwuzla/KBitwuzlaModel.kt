package io.ksmt.solver.bitwuzla

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.decl.KUninterpretedSortValueDecl
import io.ksmt.expr.KExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.solver.KModel
import io.ksmt.solver.model.KFuncInterp
import io.ksmt.solver.model.KFuncInterpEntryVarsFree
import io.ksmt.solver.model.KFuncInterpEntryVarsFreeOneAry
import io.ksmt.solver.model.KFuncInterpVarsFree
import io.ksmt.solver.model.KFuncInterpWithVars
import io.ksmt.solver.model.KModelEvaluator
import io.ksmt.solver.model.KModelImpl
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import io.ksmt.utils.mkFreshConst
import io.ksmt.utils.mkFreshConstDecl
import io.ksmt.utils.uncheckedCast
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaKind
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native

open class KBitwuzlaModel(
    private val ctx: KContext,
    private val bitwuzlaCtx: KBitwuzlaContext,
    private val converter: KBitwuzlaExprConverter,
    private val internalizer: KBitwuzlaExprInternalizer,
    assertedDeclarations: Set<KDecl<*>>,
    private val uninterpretedSortDependency: Map<KUninterpretedSort, Set<KDecl<*>>>
) : KModel {

    private val modelDeclarations = assertedDeclarations.toHashSet()

    override val declarations: Set<KDecl<*>>
        get() = modelDeclarations.toHashSet()

    private val evaluatorWithModelCompletion by lazy { KModelEvaluator(ctx, this, isComplete = true) }
    private val evaluatorWithoutModelCompletion by lazy { KModelEvaluator(ctx, this, isComplete = false) }

    private var isValid: Boolean = true

    fun markInvalid() {
        isValid = false
    }

    private fun ensureModelValid() {
        bitwuzlaCtx.ensureActive()
        check(isValid) { "The model is no longer valid" }
    }

    override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> {
        ctx.ensureContextMatch(expr)

        val evaluator = if (isComplete) evaluatorWithModelCompletion else evaluatorWithoutModelCompletion
        return evaluator.apply(expr)
    }

    private val uninterpretedSortValueContext = KBitwuzlaUninterpretedSortValueContext(ctx)

    override val uninterpretedSorts: Set<KUninterpretedSort>
        get() = uninterpretedSortDependency.keys

    private val uninterpretedSortsUniverses = hashMapOf<KUninterpretedSort, Set<KUninterpretedSortValue>>()

    override fun uninterpretedSortUniverse(sort: KUninterpretedSort): Set<KUninterpretedSortValue>? =
        uninterpretedSortsUniverses.getOrPut(sort) {
            ctx.ensureContextMatch(sort)

            val sortDependency = uninterpretedSortDependency[sort]
                ?: return@uninterpretedSortUniverse null

            /**
             * Resolve interpretation for all relevant declarations
             * to ensure that [uninterpretedSortValueContext] contains
             * all possible values for the given sort.
             * */
            sortDependency.forEach {
                if (it is KUninterpretedSortValueDecl) {
                    val value = ctx.mkUninterpretedSortValue(it.sort, it.valueIdx)
                    uninterpretedSortValueContext.registerValue(value)
                } else interpretation(it)
            }

            uninterpretedSortValueContext.currentSortUniverse(sort)
        }

    private val interpretations: MutableMap<KDecl<*>, KFuncInterp<*>> = hashMapOf()

    override fun <T : KSort> interpretation(decl: KDecl<T>): KFuncInterp<T>? {
        ensureModelValid()
        ctx.ensureContextMatch(decl)

        if (decl !in modelDeclarations) return null

        val interpretation = interpretations.getOrPut(decl) {
            // Constant was not internalized --> constant is unknown to solver --> constant is not present in model
            val bitwuzlaConstant = bitwuzlaCtx.findConstant(decl)
                ?: return@interpretation null

            getInterpretation(decl, bitwuzlaConstant)
        }

        return interpretation.uncheckedCast()
    }

    private fun <T : KSort> getInterpretation(
        decl: KDecl<T>,
        term: BitwuzlaTerm
    ): KFuncInterp<T> = converter.withUninterpretedSortValueContext(uninterpretedSortValueContext) {
        when {
            Native.bitwuzlaTermIsArray(term) -> arrayInterpretation(decl, term)
            Native.bitwuzlaTermIsFun(term) -> functionInterpretation(decl, term)
            else -> {
                val value = Native.bitwuzlaGetValue(bitwuzlaCtx.bitwuzla, term)
                val convertedValue = with(converter) { value.convertExpr(decl.sort) }
                KFuncInterpVarsFree(
                    decl = decl,
                    entries = emptyList(),
                    default = convertedValue
                )
            }
        }
    }

    private fun <T : KSort> functionInterpretation(
        decl: KDecl<T>,
        term: BitwuzlaTerm
    ): KFuncInterp<T> = if (decl.argSorts.isEmpty())
        converter.retrieveFunctionValue(decl, term)
    else handleArrayFunctionDecl(decl) { functionDecl ->
        val interp = Native.bitwuzlaGetValue(bitwuzlaCtx.bitwuzla, term)
        functionValueInterpretation(functionDecl, interp)
    }

    private fun <T : KSort> functionValueInterpretation(
        decl: KDecl<T>,
        interp: BitwuzlaTerm
    ): KFuncInterpWithVars<T> {
        val vars = decl.argSorts.map { it.mkFreshConst("x") }

        val (bitwuzlaInterpVars, bitwuzlaInterpValue) = let {
            val varTerms = mutableListOf<BitwuzlaTerm>()
            var (varTerm, lambdaTerm) = Native.bitwuzlaTermGetChildren(interp)
            varTerms += varTerm
            while (Native.bitwuzlaTermGetKind(lambdaTerm) == BitwuzlaKind.BITWUZLA_KIND_LAMBDA) {
                val children = Native.bitwuzlaTermGetChildren(lambdaTerm)
                varTerm = children[0]
                varTerms += varTerm
                lambdaTerm = children[1]
            }

            varTerms.toLongArray() to lambdaTerm
        }

        val bitwuzlaVars = vars.map { with(internalizer) { it.internalizeExpr() } }.toLongArray()
        val bitwuzlaFreshInterp = Native.bitwuzlaSubstituteTerm(bitwuzlaInterpValue, bitwuzlaInterpVars, bitwuzlaVars)

        val varsToDecls = bitwuzlaVars.zip(vars).associate { (varTerm, varApp) -> varTerm to varApp.decl }
        val c = KBitwuzlaExprConverter(ctx, bitwuzlaCtx, varsToDecls)
        val defaultBody = with(c) { bitwuzlaFreshInterp.convertExpr(decl.sort) }

        return KFuncInterpWithVars(decl, vars.map { it.decl }, emptyList(), defaultBody)
    }

    private fun <T : KSort> KBitwuzlaExprConverter.retrieveFunctionValue(
        decl: KDecl<T>,
        functionTerm: BitwuzlaTerm
    ): KFuncInterp<T> = handleArrayFunctionInterpretation(decl) { arraySort ->
        // We expect lambda expression here. Therefore, we convert function interpretation as array.
        val functionValue = Native.bitwuzlaGetValue(bitwuzlaCtx.bitwuzla, functionTerm)
        functionValue.convertExpr(arraySort)
    }

    private fun <T : KSort> arrayInterpretation(
        decl: KDecl<T>,
        term: BitwuzlaTerm
    ): KFuncInterp<T> = handleArrayFunctionDecl(decl) { arrayFunctionDecl ->
        val sort: KArraySort<KSort, KSort> = decl.sort.uncheckedCast()
        val entries = mutableListOf<KFuncInterpEntryVarsFree<KSort>>()

        val interp = Native.bitwuzlaGetValue(bitwuzlaCtx.bitwuzla, term)

        var currentEntry = interp
        while (Native.bitwuzlaTermGetKind(currentEntry) == BitwuzlaKind.BITWUZLA_KIND_ARRAY_STORE) {
            val (array, bzlaIndex, bzlaValue) = Native.bitwuzlaTermGetChildren(currentEntry)
            val index = bzlaIndex.convertExpr(sort.domain)
            val value = bzlaValue.convertExpr(sort.range)
            entries += KFuncInterpEntryVarsFreeOneAry(index, value)
            currentEntry = array
        }
        val defaultValue = Native.bitwuzlaTermGetChildren(currentEntry)[0].convertExpr(sort.range)
        KFuncInterpVarsFree(arrayFunctionDecl, entries, defaultValue)
    }

    private inline fun <T : KSort> handleArrayFunctionDecl(
        decl: KDecl<T>,
        body: KBitwuzlaExprConverter.(KDecl<KSort>) -> KFuncInterp<*>
    ): KFuncInterp<T> = with(ctx) {
        val sort = decl.sort

        if (sort !is KArraySortBase<*>) {
            return converter.body(decl.uncheckedCast()).uncheckedCast()
        }

        check(decl.argSorts.isEmpty()) { "Unexpected function with array range" }

        val arrayInterpDecl = mkFreshFuncDecl("array", sort.range, sort.domainSorts)

        modelDeclarations += arrayInterpDecl
        interpretations[arrayInterpDecl] = converter.body(arrayInterpDecl)

        KFuncInterpVarsFree(
            decl = decl,
            entries = emptyList(),
            default = mkFunctionAsArray(sort.uncheckedCast(), arrayInterpDecl).uncheckedCast()
        )
    }

    private inline fun <T : KSort> KBitwuzlaExprConverter.handleArrayFunctionInterpretation(
        decl: KDecl<T>,
        convertInterpretation: (KArraySortBase<*>) -> KExpr<KArraySortBase<*>>
    ): KFuncInterp<T> {
        val sort = decl.sort

        if (sort is KArraySortBase<*> && decl.argSorts.isEmpty()) {
            val arrayInterpretation = convertInterpretation(sort)
            return KFuncInterpVarsFree(
                decl = decl,
                entries = emptyList(),
                default = arrayInterpretation.uncheckedCast()
            )
        }

        check(decl.argSorts.isEmpty()) { "Unexpected function with array range" }

        val interpretationSort = ctx.mkAnyArraySort(decl.argSorts, decl.sort)
        val arrayInterpretation = convertInterpretation(interpretationSort)

        val functionVars = decl.argSorts.mapIndexed { i, s -> s.mkFreshConstDecl("x!$i") }
        val functionValue = ctx.mkAnyArraySelect(arrayInterpretation, functionVars.map { it.apply() })

        return KFuncInterpWithVars(
            decl = decl,
            vars = functionVars,
            entries = emptyList(),
            default = functionValue.uncheckedCast()
        )
    }

    override fun detach(): KModel {
        uninterpretedSorts.forEach {
            uninterpretedSortUniverse(it) ?: error("missed sort universe for $it")
        }

        declarations.forEach {
            interpretation(it) ?: error("missed interpretation for $it")
        }

        return KModelImpl(ctx, interpretations, uninterpretedSortsUniverses)
    }

    override fun toString(): String = detach().toString()
    override fun hashCode(): Int = detach().hashCode()
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is KModel) return false
        return detach() == other
    }
}
