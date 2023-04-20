package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.expr.KUninterpretedSortValue
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolverUnsupportedFeatureException
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaNativeException
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.FunValue
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.solver.model.KModelEvaluator
import org.ksmt.solver.model.KModelImpl
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.utils.mkFreshConstDecl
import org.ksmt.utils.uncheckedCast

open class KBitwuzlaModel(
    private val ctx: KContext,
    private val bitwuzlaCtx: KBitwuzlaContext,
    private val converter: KBitwuzlaExprConverter,
    assertedDeclarations: Set<KDecl<*>>,
    private val uninterpretedSortDependency: Map<KUninterpretedSort, Set<KDecl<*>>>
) : KModel {

    private val modelDeclarations = assertedDeclarations.toHashSet()

    override val declarations: Set<KDecl<*>>
        get() = modelDeclarations.toHashSet()

    private val evaluatorWithModelCompletion by lazy { KModelEvaluator(ctx, this, isComplete = true) }
    private val evaluatorWithoutModelCompletion by lazy { KModelEvaluator(ctx, this, isComplete = false) }

    override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> {
        bitwuzlaCtx.ensureActive()
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
            sortDependency.forEach { interpretation(it) }

            uninterpretedSortValueContext.currentSortUniverse(sort)
        }

    private val interpretations: MutableMap<KDecl<*>, KModel.KFuncInterp<*>> = hashMapOf()

    override fun <T : KSort> interpretation(decl: KDecl<T>): KModel.KFuncInterp<T>? {
        ctx.ensureContextMatch(decl)
        bitwuzlaCtx.ensureActive()

        if (decl !in modelDeclarations) return null

        val interpretation = interpretations.getOrPut(decl) {
            // Constant was not internalized --> constant is unknown to solver --> constant is not present in model
            val bitwuzlaConstant = bitwuzlaCtx.findConstant(decl)
                ?: return@interpretation null

            getInterpretationSafe(decl, bitwuzlaConstant)
        }

        return interpretation.uncheckedCast()
    }

    private fun <T : KSort> getInterpretationSafe(
        decl: KDecl<T>,
        term: BitwuzlaTerm
    ): KModel.KFuncInterp<T> = bitwuzlaCtx.bitwuzlaTry {
        handleModelIsUnsupportedWithQuantifiers {
            getInterpretation(decl, term)
        }
    }

    private fun <T : KSort> getInterpretation(
        decl: KDecl<T>,
        term: BitwuzlaTerm
    ): KModel.KFuncInterp<T> = converter.withUninterpretedSortValueContext(uninterpretedSortValueContext) {
        when {
            Native.bitwuzlaTermIsArray(term) -> arrayInterpretation(decl, term)
            Native.bitwuzlaTermIsFun(term) -> functionInterpretation(decl, term)
            else -> {
                val value = Native.bitwuzlaGetValue(bitwuzlaCtx.bitwuzla, term)
                val convertedValue = with(converter) { value.convertExpr(decl.sort) }
                KModel.KFuncInterp(
                    decl = decl,
                    vars = emptyList(),
                    entries = emptyList(),
                    default = convertedValue
                )
            }
        }
    }

    private fun <T : KSort> functionInterpretation(
        decl: KDecl<T>,
        term: BitwuzlaTerm
    ): KModel.KFuncInterp<T> {
        val interp = Native.bitwuzlaGetFunValue(bitwuzlaCtx.bitwuzla, term)
        return if (interp.size != 0) {
            handleArrayFunctionDecl(decl) { functionDecl, vars ->
                functionValueInterpretation(functionDecl, vars, interp)
            }
        } else {
            /**
             * Function has default value or bitwuzla can't retrieve its entries.
             * Try parse function interpretation from value term.
             * */
            converter.retrieveFunctionValue(decl, term)
        }
    }

    private fun <T : KSort> KBitwuzlaExprConverter.functionValueInterpretation(
        decl: KDecl<T>,
        vars: List<KConstDecl<*>>,
        interp: FunValue
    ): KModel.KFuncInterp<T> {
        val entries = mutableListOf<KModel.KFuncInterpEntry<T>>()

        for (i in 0 until interp.size) {
            // Don't substitute vars since arguments in Bitwuzla model are always constants
            val args = interp.args!![i].zip(decl.argSorts) { arg, sort -> arg.convertExpr(sort) }
            val value = interp.values!![i].convertExpr(decl.sort)
            entries += KModel.KFuncInterpEntry(args, value)
        }

        return KModel.KFuncInterp(
            decl = decl,
            vars = vars,
            entries = entries,
            default = null
        )
    }

    private fun <T : KSort> KBitwuzlaExprConverter.retrieveFunctionValue(
        decl: KDecl<T>,
        functionTerm: BitwuzlaTerm
    ): KModel.KFuncInterp<T> = handleArrayFunctionInterpretation(decl) { arraySort ->
        // We expect lambda expression here. Therefore, we convert function interpretation as array.
        val functionValue = Native.bitwuzlaGetValue(bitwuzlaCtx.bitwuzla, functionTerm)
        functionValue.convertExpr(arraySort)
    }

    private fun <T : KSort> arrayInterpretation(
        decl: KDecl<T>,
        term: BitwuzlaTerm
    ): KModel.KFuncInterp<T> = handleArrayFunctionDecl(decl) { arrayFunctionDecl, vars ->
        val sort: KArraySort<KSort, KSort> = decl.sort.uncheckedCast()
        val entries = mutableListOf<KModel.KFuncInterpEntry<KSort>>()
        val interp = Native.bitwuzlaGetArrayValue(bitwuzlaCtx.bitwuzla, term)

        for (i in 0 until interp.size) {
            val index = interp.indices!![i].convertExpr(sort.domain)
            val value = interp.values!![i].convertExpr(sort.range)
            entries += KModel.KFuncInterpEntry(listOf(index), value)
        }

        val default = interp.defaultValue.takeIf { it != 0L }?.convertExpr(sort.range)

        KModel.KFuncInterp(
            decl = arrayFunctionDecl,
            vars = vars,
            entries = entries,
            default = default
        )
    }

    private inline fun <T : KSort> handleArrayFunctionDecl(
        decl: KDecl<T>,
        body: KBitwuzlaExprConverter.(KDecl<KSort>, List<KConstDecl<*>>) -> KModel.KFuncInterp<*>
    ): KModel.KFuncInterp<T> = with(ctx) {
        val sort = decl.sort

        if (sort !is KArraySortBase<*>) {
            val vars = decl.argSorts.mapIndexed { i, s -> s.mkFreshConstDecl("x!$i") }
            return converter.body(decl.uncheckedCast(), vars).uncheckedCast()
        }

        check(decl.argSorts.isEmpty()) { "Unexpected function with array range" }

        val arrayInterpDecl = mkFreshFuncDecl("array", sort.range, sort.domainSorts)
        val arrayInterpIndicesDecls = sort.domainSorts.mapIndexed { i, s ->
            s.mkFreshConstDecl("idx!$i")
        }

        modelDeclarations += arrayInterpDecl
        interpretations[arrayInterpDecl] = converter.body(arrayInterpDecl, arrayInterpIndicesDecls)

        KModel.KFuncInterp(
            decl = decl,
            vars = emptyList(),
            entries = emptyList(),
            default = mkFunctionAsArray(sort.uncheckedCast(), arrayInterpDecl).uncheckedCast()
        )
    }

    private inline fun <T : KSort> KBitwuzlaExprConverter.handleArrayFunctionInterpretation(
        decl: KDecl<T>,
        convertInterpretation: (KArraySortBase<*>) -> KExpr<KArraySortBase<*>>
    ): KModel.KFuncInterp<T> {
        val sort = decl.sort

        if (sort is KArraySortBase<*> && decl.argSorts.isEmpty()) {
            val arrayInterpretation = convertInterpretation(sort)
            return KModel.KFuncInterp(
                decl = decl,
                vars = emptyList(),
                entries = emptyList(),
                default = arrayInterpretation.uncheckedCast()
            )
        }

        check(decl.argSorts.isEmpty()) { "Unexpected function with array range" }

        val interpretationSort = ctx.mkAnyArraySort(decl.argSorts, decl.sort)
        val arrayInterpretation = convertInterpretation(interpretationSort)

        val functionVars = decl.argSorts.mapIndexed { i, s -> s.mkFreshConstDecl("x!$i") }
        val functionValue = ctx.mkAnyArraySelect(arrayInterpretation, functionVars.map { it.apply() })

        return KModel.KFuncInterp(
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

    /**
     * Models are currently not supported for formulas with quantifiers.
     * Handle this case as unsupported feature rather than general solver failure.
     * */
    private inline fun <T> handleModelIsUnsupportedWithQuantifiers(body: () -> T): T = try {
        body()
    } catch (ex: BitwuzlaNativeException) {
        if (isModelUnsupportedWithQuantifiers(ex)) {
            throw KSolverUnsupportedFeatureException("Model are not supported for formulas with quantifiers")
        }
        throw ex
    }

    companion object {
        private const val MODEL_UNSUPPORTED_WITH_QUANTIFIERS =
            "'get-value' is currently not supported with quantifiers\n"

        private fun isModelUnsupportedWithQuantifiers(ex: BitwuzlaNativeException): Boolean {
            val message = ex.message ?: return false
            /**
             * Bitwuzla exception message has the following format:
             * `[bitwuzla] <api method name> 'get-value' is currently ....`.
             *
             * Since we don't know the actual api method name
             * (we have multiple ways to trigger get-value),
             * we use `endsWith`.
             * */
            return message.endsWith(MODEL_UNSUPPORTED_WITH_QUANTIFIERS)
        }
    }
}
