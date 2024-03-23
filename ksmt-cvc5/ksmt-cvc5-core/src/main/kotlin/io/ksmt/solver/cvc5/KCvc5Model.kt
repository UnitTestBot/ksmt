package io.ksmt.solver.cvc5

import io.github.cvc5.Term
import io.ksmt.KContext
import io.ksmt.decl.KConstDecl
import io.ksmt.decl.KDecl
import io.ksmt.decl.KFuncDecl
import io.ksmt.expr.KExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.solver.model.KFuncInterp
import io.ksmt.solver.model.KFuncInterpVarsFree
import io.ksmt.solver.model.KFuncInterpWithVars
import io.ksmt.solver.KModel
import io.ksmt.solver.model.KModelEvaluator
import io.ksmt.solver.model.KModelImpl
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import io.ksmt.utils.mkFreshConst
import java.util.TreeMap

open class KCvc5Model(
    private val ctx: KContext,
    private val cvc5Ctx: KCvc5Context,
    private val internalizer: KCvc5ExprInternalizer,
    override val declarations: Set<KDecl<*>>,
    override val uninterpretedSorts: Set<KUninterpretedSort>
) : KModel {
    private val converter: KCvc5ExprConverter by lazy { KCvc5ExprConverter(ctx, cvc5Ctx, this) }

    private val interpretations = hashMapOf<KDecl<*>, KFuncInterp<*>?>()
    private val uninterpretedSortValues = hashMapOf<KUninterpretedSort, UninterpretedSortValueContext>()

    private val evaluatorWithCompletion by lazy { KModelEvaluator(ctx, this, isComplete = true) }
    private val evaluatorWithoutCompletion by lazy { KModelEvaluator(ctx, this, isComplete = false) }

    private var isValid: Boolean = true

    fun markInvalid() {
        isValid = false
    }

    private fun ensureModelValid() {
        ensureContextActive()
        check(isValid) { "The model is no longer valid" }
    }

    override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> {
        ctx.ensureContextMatch(expr)

        val evaluator = if (isComplete) evaluatorWithCompletion else evaluatorWithoutCompletion
        return evaluator.apply(expr)
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : KSort> interpretation(decl: KDecl<T>): KFuncInterp<T>? = interpretations.getOrPut(decl) {
        ensureModelValid()
        ctx.ensureContextMatch(decl)

        if (decl !in declarations) return@getOrPut null

        val cvc5Decl = with(internalizer) { decl.internalizeDecl() }

        when (decl) {
            is KConstDecl<T> -> constInterp(decl, cvc5Decl)
            is KFuncDecl<T> -> funcInterp(decl, cvc5Decl)
            else -> error("Unknown declaration")
        }
    } as? KFuncInterp<T>


    // cvc5 function interpretation - declaration is Term of kind Lambda
    private fun <T : KSort> funcInterp(
        decl: KDecl<T>,
        internalizedDecl: Term
    ): KFuncInterp<T> = with(converter) {
        val cvc5Interp = cvc5Ctx.nativeSolver.getValue(internalizedDecl)

        val vars = decl.argSorts.map { it.mkFreshConst("x") }
        val cvc5Vars = vars.map { with(internalizer) { it.internalizeExpr() } }.toTypedArray()

        val cvc5InterpArgs = cvc5Interp.getChild(0).getChildren()
        val cvc5FreshVarsInterp = cvc5Interp.substitute(cvc5InterpArgs, cvc5Vars)

        val defaultBody = cvc5FreshVarsInterp.getChild(1).convertExpr<T>()

        KFuncInterpWithVars(decl, vars.map { it.decl }, emptyList(), defaultBody)
    }

    private fun <T : KSort> constInterp(decl: KDecl<T>, const: Term): KFuncInterp<T> = with(converter) {
        val cvc5Interp = cvc5Ctx.nativeSolver.getValue(const)
        val interp = cvc5Interp.convertExpr<T>()

        KFuncInterpVarsFree(decl = decl, entries = emptyList(), default = interp)
    }

    override fun uninterpretedSortUniverse(sort: KUninterpretedSort): Set<KUninterpretedSortValue>? =
        getUninterpretedSortContext(sort).getSortUniverse()

    internal fun resolveUninterpretedSortValue(sort: KUninterpretedSort, value: Term): KUninterpretedSortValue =
        getUninterpretedSortContext(sort).getValue(value)

    override fun detach(): KModel {
        val interpretations = declarations.associateWith {
            interpretation(it) ?: error("missed interpretation for $it")
        }

        val uninterpretedSortsUniverses = uninterpretedSorts.associateWith {
            uninterpretedSortUniverse(it) ?: error("missed sort universe for $it")
        }

        // The model is detached from the solver and therefore invalid
        markInvalid()

        return KModelImpl(ctx, interpretations, uninterpretedSortsUniverses)
    }

    override fun close() {
        markInvalid()
    }

    private fun ensureContextActive() = check(cvc5Ctx.isActive) { "Context already closed" }

    private fun getUninterpretedSortContext(sort: KUninterpretedSort): UninterpretedSortValueContext =
        uninterpretedSortValues.getOrPut(sort) { UninterpretedSortValueContext(sort) }

    private inner class UninterpretedSortValueContext(val sort: KUninterpretedSort) {
        private var initialized = false
        private var currentValueIdx = 0
        private val modelValues = TreeMap<Term, KUninterpretedSortValue>()
        private val sortUniverse = hashSetOf<KUninterpretedSortValue>()

        fun getSortUniverse(): Set<KUninterpretedSortValue> {
            ensureInitialized()
            return sortUniverse
        }

        fun getValue(modelValue: Term): KUninterpretedSortValue {
            ensureInitialized()
            return mkValue(modelValue)
        }

        private fun ensureInitialized() {
            if (initialized) return
            initialize()
            initialized = true
        }

        private fun initialize() {
            if (sort !in uninterpretedSorts) {
                return
            }

            ensureModelValid()

            initializeModelValues()

            val cvc5Sort = with(internalizer) { sort.internalizeSort() }
            val cvc5SortUniverse = cvc5Ctx.nativeSolver.getModelDomainElements(cvc5Sort)

            initializeSortUniverse(cvc5SortUniverse)
        }

        private fun initializeModelValues() {
            val registeredValues = cvc5Ctx.getRegisteredSortValues(sort)
            registeredValues.forEach { (nativeValue, value) ->
                val modelValue = cvc5Ctx.nativeSolver.getValue(nativeValue)
                modelValues[modelValue] = value
                currentValueIdx = maxOf(currentValueIdx, value.valueIdx + 1)
            }
        }

        private fun initializeSortUniverse(universe: Array<Term>) {
            universe.forEach {
                sortUniverse.add(mkValue(it))
            }
        }

        private fun mkValue(modelValue: Term): KUninterpretedSortValue = modelValues.getOrPut(modelValue) {
            mkFreshValue()
        }

        private fun mkFreshValue() = ctx.mkUninterpretedSortValue(sort, currentValueIdx++)
    }
}
