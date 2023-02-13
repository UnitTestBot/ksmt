package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolverUnsupportedFeatureException
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaNativeException
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.solver.model.KModelEvaluator
import org.ksmt.solver.model.KModelImpl
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KSortVisitor
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.utils.mkFreshConstDecl
import org.ksmt.utils.uncheckedCast

open class KBitwuzlaModel(
    private val ctx: KContext,
    private val bitwuzlaCtx: KBitwuzlaContext,
    private val converter: KBitwuzlaExprConverter,
    override val declarations: Set<KDecl<*>>
) : KModel {

    override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> {
        ctx.ensureContextMatch(expr)
        bitwuzlaCtx.ensureActive()

        return KModelEvaluator(ctx, this, isComplete).apply(expr)
    }

    private val uninterpretedSortValueContext = KBitwuzlaUninterpretedSortValueContext(ctx)

    private val uninterpretedSortDependency by lazy {
        val dependencies = hashMapOf<KUninterpretedSort, MutableSet<KDecl<*>>>()

        // Collect relevant declarations for each uninterpreted sort
        val uninterpretedSortDependencyRegisterer = UninterpretedSortRegisterer(dependencies)
        declarations.forEach {
            uninterpretedSortDependencyRegisterer.element = it
            it.sort.accept(uninterpretedSortDependencyRegisterer)
        }

        dependencies
    }

    override val uninterpretedSorts: Set<KUninterpretedSort>
        get() = uninterpretedSortDependency.keys

    private val uninterpretedSortsUniverses = hashMapOf<KUninterpretedSort, Set<KExpr<KUninterpretedSort>>>()

    override fun uninterpretedSortUniverse(sort: KUninterpretedSort): Set<KExpr<KUninterpretedSort>>? =
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

        // Constant was not internalized --> constant is unknown to solver --> constant is not present in model
        val bitwuzlaConstant = bitwuzlaCtx.findConstant(decl)
            ?: return null

        val interpretation = interpretations.getOrPut(decl) {
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
            Native.bitwuzlaTermIsArray(term) -> ctx.arrayInterpretation(decl, term)
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
    ): KModel.KFuncInterp<T> = with(converter) {
        check(decl is KFuncDecl<T>) { "function expected but actual is $decl" }

        val entries = mutableListOf<KModel.KFuncInterpEntry<T>>()
        val interp = Native.bitwuzlaGetFunValue(bitwuzlaCtx.bitwuzla, term)

        check(interp.arity == decl.argSorts.size) {
            "function arity mismatch: ${interp.arity} and ${decl.argSorts.size}"
        }

        val vars = decl.argSorts.map { it.mkFreshConstDecl("x") }
        for (i in 0 until interp.size) {
            // Don't substitute vars since arguments in Bitwuzla model are always constants
            val args = interp.args!![i].zip(decl.argSorts) { arg, sort -> arg.convertExpr(sort) }
            val value = interp.values!![i].convertExpr(decl.sort)
            entries += KModel.KFuncInterpEntry(args, value)
        }

        KModel.KFuncInterp(
            decl = decl,
            vars = vars,
            entries = entries,
            default = null
        )
    }

    private fun <T : KSort> KContext.arrayInterpretation(
        decl: KDecl<T>,
        term: BitwuzlaTerm
    ): KModel.KFuncInterp<T> = with(converter) {
        val sort = decl.sort as KArraySort<*, *>
        val entries = mutableListOf<KModel.KFuncInterpEntry<KSort>>()
        val interp = Native.bitwuzlaGetArrayValue(bitwuzlaCtx.bitwuzla, term)

        for (i in 0 until interp.size) {
            val index = interp.indices!![i].convertExpr(sort.domain)
            val value = interp.values!![i].convertExpr(sort.range)
            entries += KModel.KFuncInterpEntry(listOf(index), value)
        }

        val default = interp.defaultValue.takeIf { it != 0L }?.convertExpr(sort.range)
        val arrayInterpDecl = mkFreshFuncDecl("array", sort.range, listOf(sort.domain))
        val arrayInterpIndexDecl = mkFreshConstDecl("idx", sort.domain)

        interpretations[arrayInterpDecl] = KModel.KFuncInterp(
            decl = arrayInterpDecl,
            vars = listOf(arrayInterpIndexDecl),
            entries = entries,
            default = default
        )

        KModel.KFuncInterp(
            decl = decl,
            vars = emptyList(),
            entries = emptyList(),
            default = mkFunctionAsArray<KSort, KSort>(arrayInterpDecl).uncheckedCast()
        )
    }

    override fun detach(): KModel {
        val interpretations = declarations.associateWith {
            interpretation(it) ?: error("missed interpretation for $it")
        }

        val uninterpretedSortsUniverses = uninterpretedSorts.associateWith {
            uninterpretedSortUniverse(it) ?: error("missed sort universe for $it")
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
        val modelUnsupportedMessage = "'get-value' is currently not supported with quantifiers"
        if (modelUnsupportedMessage in (ex.message ?: "")) {
            throw KSolverUnsupportedFeatureException(modelUnsupportedMessage)
        }
        throw ex
    }

    private class UninterpretedSortRegisterer<T : Any>(
        private val register: MutableMap<KUninterpretedSort, MutableSet<T>>
    ) : KSortVisitor<Unit> {
        lateinit var element: T

        override fun visit(sort: KUninterpretedSort) {
            val sortElements = register.getOrPut(sort) { hashSetOf() }
            sortElements += element
        }

        override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>) {
            sort.domain.accept(this)
            sort.range.accept(this)
        }

        override fun visit(sort: KBoolSort) {
        }

        override fun visit(sort: KIntSort) {
        }

        override fun visit(sort: KRealSort) {
        }

        override fun <S : KBvSort> visit(sort: S) {
        }

        override fun <S : KFpSort> visit(sort: S) {
        }

        override fun visit(sort: KFpRoundingModeSort) {
        }
    }
}
