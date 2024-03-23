package io.ksmt.solver.yices

import com.sri.yices.Model
import com.sri.yices.YVal
import com.sri.yices.YValTag
import io.ksmt.KContext
import io.ksmt.decl.KConstDecl
import io.ksmt.decl.KDecl
import io.ksmt.decl.KFuncDecl
import io.ksmt.expr.KExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.solver.model.KFuncInterp
import io.ksmt.solver.model.KFuncInterpEntryVarsFree
import io.ksmt.solver.model.KFuncInterpVarsFree
import io.ksmt.solver.KModel
import io.ksmt.solver.model.KModelEvaluator
import io.ksmt.solver.model.KModelImpl
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KSortVisitor
import io.ksmt.sort.KUninterpretedSort
import io.ksmt.utils.uncheckedCast

class KYicesModel(
    nativeModel: Model,
    private val ctx: KContext,
    private val yicesCtx: KYicesContext,
    private val internalizer: KYicesExprInternalizer,
    private val converter: KYicesExprConverter
) : KModel {
    private var nativeModel: Model? = nativeModel
    private val model: Model
        get() = nativeModel ?: error("Native model released")

    override val declarations: Set<KDecl<*>> by lazy {
        model.collectDefinedTerms().mapTo(hashSetOf()) { converter.convertDecl(it) }
    }

    override val uninterpretedSorts: Set<KUninterpretedSort> by lazy {
        uninterpretedSortDependencies.keys
    }

    private val uninterpretedSortDependencies: Map<KUninterpretedSort, Set<KDecl<*>>> by lazy {
        val sortsWithDependencies = hashMapOf<KUninterpretedSort, MutableSet<KDecl<*>>>()
        val sortCollector = UninterpretedSortCollector(sortsWithDependencies)
        declarations.forEach { sortCollector.collect(it) }
        sortsWithDependencies
    }

    private val uninterpretedSortUniverse =
        hashMapOf<KUninterpretedSort, Set<KUninterpretedSortValue>>()

    private val knownUninterpretedSortValues =
        hashMapOf<KUninterpretedSort, MutableMap<Int, KUninterpretedSortValue>>()

    private val interpretations = hashMapOf<KDecl<*>, KFuncInterp<*>>()
    private val funcInterpretationsToDo = arrayListOf<Pair<YVal, KFuncDecl<*>>>()

    override fun uninterpretedSortUniverse(
        sort: KUninterpretedSort
    ): Set<KUninterpretedSortValue>? = uninterpretedSortUniverse.getOrPut(sort) {
        val sortDependencies = uninterpretedSortDependencies[sort] ?: return null

        sortDependencies.forEach { interpretation(it) }

        knownUninterpretedSortValues[sort]?.values?.toHashSet() ?: hashSetOf()
    }

    private val evaluatorWithModelCompletion by lazy { KModelEvaluator(ctx, this, isComplete = true) }
    private val evaluatorWithoutModelCompletion by lazy { KModelEvaluator(ctx, this, isComplete = false) }

    override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> {
        ctx.ensureContextMatch(expr)

        val evaluator = if (isComplete) evaluatorWithModelCompletion else evaluatorWithoutModelCompletion
        return evaluator.apply(expr)
    }

    private fun getValue(yval: YVal, sort: KSort): KExpr<*> = with(ctx) {
        return when (sort) {
            is KBoolSort -> model.boolValue(yval).expr
            is KBvSort -> mkBv(model.bvValue(yval), sort.sizeBits)
            is KRealSort -> mkRealNum(model.bigRationalValue(yval))
            is KIntSort -> mkIntNum(model.bigRationalValue(yval))
            is KUninterpretedSort -> {
                val uninterpretedSortValueId = model.scalarValue(yval)[0]
                val sortValues = knownUninterpretedSortValues.getOrPut(sort) { hashMapOf() }
                sortValues.getOrPut(uninterpretedSortValueId) {
                    val valueIndex = yicesCtx.convertUninterpretedSortValueIndex(uninterpretedSortValueId)
                    mkUninterpretedSortValue(sort, valueIndex)
                }
            }
            is KArraySortBase<*> -> {
                val funcDecl = ctx.mkFreshFuncDecl("array", sort.range, sort.domainSorts)

                funcInterpretationsToDo.add(Pair(yval, funcDecl))

                mkFunctionAsArray(sort.uncheckedCast(), funcDecl)
            }
            else -> error("Unsupported sort $sort")
        }
    }

    private fun <T: KSort> functionInterpretation(yval: YVal, decl: KFuncDecl<T>): KFuncInterp<T> {
        val functionChildren = model.expandFunction(yval)
        val default = if (yval.tag != YValTag.UNKNOWN) {
            getValue(functionChildren.value, decl.sort).uncheckedCast<_, KExpr<T>>()
        } else {
            null
        }

        val entries = functionChildren.vector.map { mapping ->
            val entry = model.expandMapping(mapping)
            val args = entry.vector.zip(decl.argSorts).map { (arg, sort) ->
                getValue(arg, sort)
            }
            val res = getValue(entry.value, decl.sort).uncheckedCast<_, KExpr<T>>()

            KFuncInterpEntryVarsFree.create(args, res)
        }

        return KFuncInterpVarsFree(
            decl = decl,
            entries = entries,
            default = default
        )
    }

    override fun <T : KSort> interpretation(decl: KDecl<T>): KFuncInterp<T>? = with(ctx) {
        interpretations.getOrPut(decl) {
            if (decl !in declarations) return@with null

            val yicesDecl = with(internalizer) { decl.internalizeDecl() }
            val yval = model.getValue(yicesDecl)

            val result = when (decl) {
                is KConstDecl<T> -> KFuncInterpVarsFree(
                    decl = decl,
                    entries = emptyList(),
                    default = getValue(yval, decl.sort).uncheckedCast()
                )
                is KFuncDecl<T> -> functionInterpretation(yval, decl)
                else -> error("Unexpected declaration $decl")
            }

            while (funcInterpretationsToDo.isNotEmpty()) {
                val (yvalT, declT) = funcInterpretationsToDo.removeLast()

                interpretations[declT] = functionInterpretation(yvalT, declT)
            }

            result
        }.uncheckedCast<_, KFuncInterp<T>?>()
    }

    override fun detach(): KModel {
        declarations.forEach { interpretation(it) }

        val uninterpretedSortsUniverses = uninterpretedSorts.associateWith {
            uninterpretedSortUniverse(it) ?: error("missed sort universe for $it")
        }

        // The model is detached from the solver and therefore can be released
        releaseNativeModel()

        return KModelImpl(ctx, interpretations.toMap(), uninterpretedSortsUniverses)
    }

    override fun close() {
        releaseNativeModel()
    }

    private fun releaseNativeModel() {
        nativeModel?.close()
        nativeModel = null
    }

    private class UninterpretedSortCollector(
        private val sorts: MutableMap<KUninterpretedSort, MutableSet<KDecl<*>>>
    ) : KSortVisitor<Unit> {
        private lateinit var currentDecl: KDecl<*>

        fun collect(decl: KDecl<*>) {
            currentDecl = decl

            decl.sort.accept(this)
            decl.argSorts.forEach { it.accept(this) }
        }

        override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>) {
            sort.range.accept(this)
            sort.domain.accept(this)
        }

        override fun <D0 : KSort, D1 : KSort, R : KSort> visit(sort: KArray2Sort<D0, D1, R>) {
            sort.range.accept(this)
            sort.domain0.accept(this)
            sort.domain1.accept(this)
        }

        override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(sort: KArray3Sort<D0, D1, D2, R>) {
            sort.range.accept(this)
            sort.domain0.accept(this)
            sort.domain1.accept(this)
            sort.domain2.accept(this)
        }

        override fun <R : KSort> visit(sort: KArrayNSort<R>) {
            sort.range.accept(this)
            sort.domainSorts.forEach { it.accept(this) }
        }

        override fun visit(sort: KUninterpretedSort) {
            val sortDependencies = sorts.getOrPut(sort) { hashSetOf() }
            sortDependencies.add(currentDecl)
        }

        override fun visit(sort: KBoolSort) = Unit
        override fun visit(sort: KIntSort) = Unit
        override fun visit(sort: KRealSort) = Unit
        override fun <S : KBvSort> visit(sort: S) = Unit
        override fun <S : KFpSort> visit(sort: S) = Unit
        override fun visit(sort: KFpRoundingModeSort) = Unit
    }
}
