package io.ksmt.solver.z3

import com.microsoft.z3.Model
import com.microsoft.z3.Native
import com.microsoft.z3.evalNative
import com.microsoft.z3.getConstInterp
import com.microsoft.z3.getFuncInterp
import com.microsoft.z3.getNativeConstDecls
import com.microsoft.z3.getNativeFuncDecls
import com.microsoft.z3.getNativeSorts
import com.microsoft.z3.getSortUniverse
import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.solver.KModel
import io.ksmt.solver.model.KFuncInterp
import io.ksmt.solver.model.KFuncInterpEntryVarsFree
import io.ksmt.solver.model.KFuncInterpEntryWithVars
import io.ksmt.solver.model.KFuncInterpVarsFree
import io.ksmt.solver.model.KFuncInterpWithVars
import io.ksmt.solver.model.KModelImpl
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import io.ksmt.utils.mkFreshConst
import io.ksmt.utils.uncheckedCast

open class KZ3Model(
    nativeModel: Model,
    private val ctx: KContext,
    private val z3Ctx: KZ3Context,
    private val internalizer: KZ3ExprInternalizer
) : KModel {
    private var nativeModel: Model? = nativeModel
    private val model: Model
        get() = nativeModel ?: error("Native model released")

    private val converter by lazy { KZ3ExprConverter(ctx, z3Ctx, this) }

    private val constantDeclarations: Set<KDecl<*>> by lazy {
        loadConstDeclarations()
    }

    private val functionDeclarations: Set<KDecl<*>> by lazy {
        model.getNativeFuncDecls().convertToSet {
            /**
             * In a case of uninterpreted sort values, we introduce
             * special `interpreter` functions which are internal and
             * must not appear in a user models.
             * */
            if (z3Ctx.isInternalFuncDecl(it)) null else it.convertDecl<KSort>()
        }
    }

    override val uninterpretedSorts: Set<KUninterpretedSort> by lazy {
        model.getNativeSorts().convertToSet { it.convertSort() as KUninterpretedSort }
    }

    override val declarations: Set<KDecl<*>> by lazy {
        constantDeclarations + functionDeclarations
    }

    private val interpretations = hashMapOf<KDecl<*>, KFuncInterp<*>?>()
    private val uninterpretedSortValues = hashMapOf<KUninterpretedSort, UninterpretedSortValueContext>()

    private fun loadConstDeclarations(): Set<KDecl<KSort>> {
        val nativeDecls = model.getNativeConstDecls()
        return nativeDecls.convertToSet {
            val uninterpretedSortValue = z3Ctx.findInternalConstDeclAssociatedUninterpretedSortValue(it)
            if (uninterpretedSortValue == null) {
                // normal decl
                it.convertDecl<KSort>()
            } else {
                // internal decl
                val valueContext = getUninterpretedSortContext(uninterpretedSortValue.sort)
                valueContext.registerValueDecl(it, uninterpretedSortValue)

                // hide declaration in model
                null
            }
        }
    }

    override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> {
        ctx.ensureContextMatch(expr)
        ensureContextActive()

        val z3Expr = with(internalizer) { expr.internalizeExpr() }
        val z3Result = z3Ctx.temporaryAst(model.evalNative(z3Expr, isComplete))

        val result: KExpr<T> = with(converter) { z3Result.convertExpr() }

        z3Ctx.releaseTemporaryAst(z3Result)

        return result
    }

    override fun <T : KSort> interpretation(decl: KDecl<T>): KFuncInterp<T>? =
        interpretations.getOrPut(decl) {
            ctx.ensureContextMatch(decl)
            ensureContextActive()

            if (decl !in declarations) return@getOrPut null

            val z3Decl = with(internalizer) { decl.internalizeDecl() }

            when (decl) {
                in constantDeclarations -> constInterp(decl, z3Decl)
                in functionDeclarations -> funcInterp(decl, z3Decl)
                else -> error("decl $decl is in model declarations but not present in model")
            }
        }?.uncheckedCast()

    override fun uninterpretedSortUniverse(sort: KUninterpretedSort): Set<KUninterpretedSortValue>? =
        getUninterpretedSortContext(sort).getSortUniverse()

    internal fun resolveUninterpretedSortValue(sort: KUninterpretedSort, decl: Long): KUninterpretedSortValue =
        getUninterpretedSortContext(sort).getValue(decl)

    private fun <T : KSort> constInterp(decl: KDecl<T>, z3Decl: Long): KFuncInterp<T>? {
        val z3Interp = model.getConstInterp(z3Decl) ?: return null
        val expr = with(converter) { z3Interp.convertExpr<T>() }
        return KFuncInterpVarsFree(decl = decl, entries = emptyList(), default = expr)
    }

    private fun <T : KSort> funcInterp(decl: KDecl<T>, z3Decl: Long): KFuncInterp<T>? = with(converter) {
        val z3Interp = model.getFuncInterp(z3Decl) ?: return null

        val vars = decl.argSorts.map { it.mkFreshConst("x") }
        val z3Vars = LongArray(vars.size) {
            with(internalizer) { vars[it].internalizeExpr() }
        }

        var interpretationVarsFree = true

        val entries = z3Interp.entries.map { entry ->
            var entryVarsFree = true

            val args = entry.args.map {
                it.substituteVarsAndConvert<KSort>(z3Vars) { varsFree ->
                    entryVarsFree = entryVarsFree && varsFree
                }
            }

            val value = entry.value.substituteVarsAndConvert<T>(z3Vars) { varsFree ->
                entryVarsFree = entryVarsFree && varsFree
            }

            interpretationVarsFree = interpretationVarsFree && entryVarsFree

            if (entryVarsFree) {
                KFuncInterpEntryVarsFree.create(args, value)
            } else {
                KFuncInterpEntryWithVars.create(args, value)
            }
        }

        val default = z3Interp.elseExpr.substituteVarsAndConvert<T>(z3Vars) { varsFree ->
            interpretationVarsFree = interpretationVarsFree && varsFree
        }

        return if (interpretationVarsFree) {
            KFuncInterpVarsFree(decl, entries.uncheckedCast(), default)
        } else {
            val varDecls = vars.map { it.decl }
            KFuncInterpWithVars(decl, varDecls, entries, default)
        }
    }

    override fun detach(): KModel {
        val interpretations = declarations.associateWith {
            interpretation(it) ?: error("missed interpretation for $it")
        }

        val uninterpretedSortsUniverses = uninterpretedSorts.associateWith {
            uninterpretedSortUniverse(it) ?: error("missed sort universe for $it")
        }

        // The model is detached from the solver and therefore can be released
        releaseNativeModel()

        return KModelImpl(ctx, interpretations, uninterpretedSortsUniverses)
    }

    override fun close() {
        releaseNativeModel()
    }

    private fun ensureContextActive() = check(z3Ctx.isActive) { "Context already closed" }

    private fun releaseNativeModel() {
        // Released by GC
        nativeModel = null
    }

    private fun <T> LongArray.convertToSet(convert: KZ3ExprConverter.(Long) -> T?): Set<T> {
        val result = HashSet<T>(size)
        for (it in this) {
            val converted = converter.convert(it)
            if (converted != null) {
                result.add(converted)
            }
        }
        return result
    }

    private inline fun <T : KSort> Long.substituteVarsAndConvert(
        vars: LongArray,
        expressionHasNoVars: (Boolean) -> Unit
    ): KExpr<T> {
        val preparedExpr = z3Ctx.temporaryAst(
            Native.substituteVars(z3Ctx.nCtx, this, vars.size, vars)
        )

        // Expression remain unchanged -> no vars were substituted
        expressionHasNoVars(this == preparedExpr)

        val convertedExpr = with(converter) {
            preparedExpr.convertExpr<T>()
        }
        z3Ctx.releaseTemporaryAst(preparedExpr)
        return convertedExpr
    }

    private fun getUninterpretedSortContext(sort: KUninterpretedSort): UninterpretedSortValueContext =
        uninterpretedSortValues.getOrPut(sort) { UninterpretedSortValueContext(sort) }

    private inner class UninterpretedSortValueContext(val sort: KUninterpretedSort) {
        private var initialized = false
        private var currentValueIdx = 0
        private val declValues = hashMapOf<Long, KUninterpretedSortValue>()
        private val modelValues = hashMapOf<Long, KUninterpretedSortValue>()
        private val sortUniverse = hashSetOf<KUninterpretedSortValue>()

        fun registerValueDecl(decl: Long, value: KUninterpretedSortValue) {
            declValues[decl] = value
        }

        fun getSortUniverse(): Set<KUninterpretedSortValue> {
            ensureInitialized()
            return sortUniverse
        }

        fun getValue(decl: Long): KUninterpretedSortValue {
            ensureInitialized()
            return mkValue(decl)
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

            /**
             * Force model constants initialization to register all value decls.
             * See [registerValueDecl] usages.
             * */
            constantDeclarations

            initializeModelValues(model)

            val z3Sort = with(internalizer) { sort.internalizeSort() }
            val z3SortUniverse = model.getSortUniverse(z3Sort)

            initializeSortUniverse(z3SortUniverse)
        }

        private fun initializeModelValues(model: Model) {
            declValues.forEach { (modelDecl, value) ->
                val modelValue = model.getConstInterp(modelDecl)
                    ?: error("Const decl is in model decls but not in the model")

                val modelValueDecl = Native.getAppDecl(z3Ctx.nCtx, modelValue)

                modelValues[modelValueDecl] = value
                currentValueIdx = maxOf(currentValueIdx, value.valueIdx + 1)
            }
        }

        private fun initializeSortUniverse(universe: LongArray) {
            universe.forEach {
                val modelValueDecl = Native.getAppDecl(z3Ctx.nCtx, it)
                sortUniverse.add(mkValue(modelValueDecl))
            }
        }

        private fun mkValue(decl: Long): KUninterpretedSortValue = modelValues.getOrPut(decl) {
            mkFreshValue()
        }

        private fun mkFreshValue() = ctx.mkUninterpretedSortValue(sort, currentValueIdx++)
    }
}
