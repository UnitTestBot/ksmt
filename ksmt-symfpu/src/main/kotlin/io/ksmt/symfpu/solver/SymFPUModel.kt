package io.ksmt.symfpu.solver

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KApp
import io.ksmt.expr.KArrayConst
import io.ksmt.expr.KArrayLambdaBase
import io.ksmt.expr.KArrayStoreBase
import io.ksmt.expr.KConst
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFunctionAsArray
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.expr.transformer.KTransformer
import io.ksmt.solver.KModel
import io.ksmt.solver.model.KFuncInterp
import io.ksmt.solver.model.KFuncInterpEntry
import io.ksmt.solver.model.KFuncInterpEntryWithVars
import io.ksmt.solver.model.KFuncInterpWithVars
import io.ksmt.solver.model.KModelEvaluator
import io.ksmt.solver.model.KModelImpl
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import io.ksmt.symfpu.operations.pack
import io.ksmt.symfpu.solver.ArraysTransform.Companion.mkAnyArrayLambda
import io.ksmt.symfpu.solver.ArraysTransform.Companion.mkAnyArrayStore
import io.ksmt.utils.uncheckedCast

class SymFPUModel(private val kModel: KModel, val ctx: KContext, val transformer: FpToBvTransformer) : KModel {
    private val mapBvToFpDecls by lazy {
        transformer.mapFpToBvDecl.entries.associateBy({ it.value.decl }) { it.key }
    }
    override val declarations: Set<KDecl<*>>
        get() = kModel.declarations.mapTo(hashSetOf()) { mapBvToFpDecls[it] ?: it }


    override val uninterpretedSorts
        get() = kModel.uninterpretedSorts


    private val evaluatorWithModelCompletion by lazy { KModelEvaluator(ctx, this, isComplete = true) }
    private val evaluatorWithoutModelCompletion by lazy { KModelEvaluator(ctx, this, isComplete = false) }
    private val interpretations: MutableMap<KDecl<*>, KFuncInterp<*>> = hashMapOf()

    override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> {
        ctx.ensureContextMatch(expr)

        val evaluator = if (isComplete) evaluatorWithModelCompletion else evaluatorWithoutModelCompletion
        return evaluator.apply(expr)
    }

    override fun <T : KSort> interpretation(decl: KDecl<T>): KFuncInterp<T>? = with(ctx) {
        ensureContextMatch(decl)
        return interpretations.getOrPut(decl) {
            if (!declContainsFp(decl)) {
                return@getOrPut kModel.interpretation<T>(decl) ?: return@with null
            }

            val const: KConst<*> = transformer.mapFpToBvDecl[decl] ?: return@with null
            val interpretation = kModel.interpretation(const.decl) ?: return null
            transformInterpretation<T>(interpretation, decl)
        }.uncheckedCast()
    }

    private fun <T : KSort> transformInterpretation(
        interpretation: KFuncInterp<out KSort>, decl: KDecl<T>,
    ): KFuncInterp<T> = with(ctx) {
        val vars: Map<KDecl<*>, KConst<*>> =
            interpretation.vars.zip(decl.argSorts) { v, sort: KSort ->
                val newConst: KConst<KSort> = mkFreshConst("var", sort).uncheckedCast()
                v to newConst
            }.toMap()

        val default = interpretation.default?.let { transformToFpSort(decl.sort, it.uncheckedCast(), vars) }
        val entries: List<KFuncInterpEntry<T>> = interpretation.entries.map {
            val args: List<KExpr<*>> = it.args.zip(decl.argSorts) { arg, sort ->
                transformToFpSort(sort, arg.uncheckedCast(), vars)
            }
            val newValue: KExpr<T> = transformToFpSort(decl.sort, it.value.uncheckedCast(), vars)
            val entry = KFuncInterpEntryWithVars.create(args, newValue)
            entry
        }

        return KFuncInterpWithVars(decl, vars.values.map { it.decl }, entries, default)
    }


    private fun transformArrayLambda(
        bvLambda: KArrayLambdaBase<*, *>, toSort: KArraySortBase<*>, vars: Map<KDecl<*>, KConst<*>>,
    ): KExpr<*> = with(ctx) {
        val fromSort = bvLambda.sort

        if (fromSort == toSort) {
            return@with bvLambda.uncheckedCast()
        }

        val indices: List<KConst<KSort>> = toSort.domainSorts.map {
            mkFreshConst("i", it).uncheckedCast()
        }

        val targetFpSort = toSort.range
        val fpValue = transformToFpSort(targetFpSort, bvLambda.body.uncheckedCast(), vars)

        mkAnyArrayLambda(
            indices.map { it.decl }, fpValue
        )
    }


    private fun <T : KSort> transformToFpSort(
        targetFpSort: T, bvExpr: KExpr<T>, vars: Map<KDecl<*>, KConst<*>>,
    ): KExpr<T> {
        if (bvExpr is KApp<*, *>) {
            vars[bvExpr.decl]?.let { return it.uncheckedCast() }
        }
        return when {
            !sortContainsFP(targetFpSort) -> bvExpr

            targetFpSort is KFpSort -> {
                ctx.pack(bvExpr.uncheckedCast(), targetFpSort.uncheckedCast()).uncheckedCast()
            }

            targetFpSort is KArraySortBase<*> -> {
                transformArray(bvExpr, targetFpSort, vars).uncheckedCast()
            }

            else -> throw IllegalArgumentException("Unsupported sort: $targetFpSort")
        }
    }

    private fun <T : KSort> SymFPUModel.transformArray(
        bvExpr: KExpr<T>, targetFpSort: KArraySortBase<*>, vars: Map<KDecl<*>, KConst<*>>,
    ) =
        when (val array: KExpr<KArraySortBase<*>> = bvExpr.uncheckedCast()) {
            is KArrayConst<*, *> -> {
                val transformedValue = transformToFpSort(targetFpSort.range, array.value.uncheckedCast(), vars)
                ctx.mkArrayConst(targetFpSort.uncheckedCast(), transformedValue)
            }

            is KArrayStoreBase<*, *> -> {
                val indices = array.indices.zip(targetFpSort.domainSorts) { bvIndex, fpSort ->
                    transformToFpSort(fpSort, bvIndex, vars)
                }
                val value = transformToFpSort(targetFpSort.range, array.value.uncheckedCast(), vars)
                val arrayInterpretation = transformToFpSort(targetFpSort, array.array.uncheckedCast(), vars)
                ctx.mkAnyArrayStore(arrayInterpretation, indices, value)
            }

            is KArrayLambdaBase<*, *> -> transformArrayLambda(array, targetFpSort, vars)

            is KFunctionAsArray<*, *> -> {
                val interpretation = interpretation(array.function) ?: throw IllegalStateException(
                    "No interpretation for ${array.function}"
                )
                val funcDecl = ctx.mkFreshFuncDecl("f", targetFpSort.range, targetFpSort.domainSorts)
                interpretations.computeIfAbsent(funcDecl) { transformInterpretation(interpretation, funcDecl) }
                ctx.mkFunctionAsArray(targetFpSort.uncheckedCast(), funcDecl)
            }

            else -> throw IllegalArgumentException(
                "Unsupported array. " +
                    "targetSort: $targetFpSort class: ${array.javaClass} array.sort ${array.sort}")
        }

    override fun uninterpretedSortUniverse(sort: KUninterpretedSort): Set<KUninterpretedSortValue>? {
        return kModel.uninterpretedSortUniverse(sort)
    }

    class AsArrayDeclInterpreter(override val ctx: KContext, private val model: KModel) : KTransformer {
        override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KFunctionAsArray<A, R>): KExpr<A> {
            model.interpretation(expr.function)?.apply {
                entries.forEach { it.value.accept(this@AsArrayDeclInterpreter) }
                default?.accept(this@AsArrayDeclInterpreter)
            }
            return expr
        }
    }

    override fun detach(): KModel {
        val asArrayDeclInterpreter = AsArrayDeclInterpreter(ctx, this)
        declarations.forEach { decl ->
            interpretation(decl)?.apply {
                entries.forEach { it.value.accept(asArrayDeclInterpreter) }
                default?.accept(asArrayDeclInterpreter)
            }
        }

        val uninterpretedSortsUniverses = uninterpretedSorts.associateWith {
            uninterpretedSortUniverse(it) ?: error("missed sort universe for $it")
        }

        return KModelImpl(ctx, interpretations.toMap(), uninterpretedSortsUniverses)
    }

    companion object {
        fun sortContainsFP(curSort: KSort): Boolean {
            return when (curSort) {
                is KFpSort -> true
                is KArraySortBase<*> -> curSort.domainSorts.any { sortContainsFP(it) } || sortContainsFP(curSort.range)
                else -> false
            }
        }

        fun <T : KSort> declContainsFp(decl: KDecl<T>) =
            sortContainsFP(decl.sort) || decl.argSorts.any { sortContainsFP(it) }
    }
}
