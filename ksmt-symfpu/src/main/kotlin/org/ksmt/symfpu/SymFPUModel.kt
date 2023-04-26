package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KArrayConst
import org.ksmt.expr.KArrayLambdaBase
import org.ksmt.expr.KArrayStoreBase
import org.ksmt.expr.KConst
import org.ksmt.expr.KExpr
import org.ksmt.expr.KUninterpretedSortValue
import org.ksmt.solver.KModel
import org.ksmt.solver.model.KModelEvaluator
import org.ksmt.solver.model.KModelImpl
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.symfpu.ArraysTransform.Companion.mkAnyArrayLambda
import org.ksmt.symfpu.ArraysTransform.Companion.mkAnyArraySelect
import org.ksmt.symfpu.ArraysTransform.Companion.mkAnyArrayStore
import org.ksmt.utils.cast
import org.ksmt.utils.uncheckedCast

class SymFPUModel(private val kModel: KModel, val ctx: KContext, val transformer: FpToBvTransformer) : KModel {
    override val declarations: Set<KDecl<*>>
        get() = kModel.declarations +
            transformer.arraysTransform.mapFpToBvDeclImpl.keys -
            transformer.arraysTransform.mapFpToBvDeclImpl.values.map { it.decl }.toSet()


    override val uninterpretedSorts: Set<KUninterpretedSort>
        get() = kModel.uninterpretedSorts


    private val evaluatorWithModelCompletion by lazy { KModelEvaluator(ctx, this, isComplete = true) }
    private val evaluatorWithoutModelCompletion by lazy { KModelEvaluator(ctx, this, isComplete = false) }
    private val interpretations: MutableMap<KDecl<*>, KModel.KFuncInterp<*>> = hashMapOf()

    override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> {
        ctx.ensureContextMatch(expr)

        val evaluator = if (isComplete) evaluatorWithModelCompletion else evaluatorWithoutModelCompletion
        return evaluator.apply(expr)
    }

    private fun <T : KSort> getConst(decl: KDecl<T>): KExpr<*>? =
        transformer.arraysTransform.mapFpToBvDeclImpl[decl]

    override fun <T : KSort> interpretation(decl: KDecl<T>): KModel.KFuncInterp<T>? = with(ctx) {
        ensureContextMatch(decl)
        return interpretations.getOrPut(decl) {
            if (!declContainsFp<T>(decl)) {
                val interpretation = kModel.interpretation(decl)
                if (interpretation != null) {
                    return@getOrPut interpretation
                } else return@with null
            }

            val const = getConst(decl) ?: return@with null
            val eval = kModel.eval(const)
            if (eval.sort is KArraySortBase<*> && eval is KConst<*>) return null
            getInterpretation(decl, eval)
        }.cast()
    }


    private fun transformArrayLambda(
        bvLambda: KExpr<KArraySortBase<*>>, toSort: KArraySortBase<*>,
    ): KExpr<*> = with(ctx) {
        val fromSort = bvLambda.sort

        if (fromSort == toSort) {
            return@with bvLambda.uncheckedCast()
        }


        val (indices: List<KConst<KSort>>, fromIndices: List<KConst<KSort>>) = toSort.domainSorts.zip(
            transformer.lambdasVariables[toSort.domainSorts]!!) { it: KSort, bvConst: KConst<KSort>? ->
            val fpConst: KConst<KSort> = mkFreshConst("i", it).cast()
            Pair(fpConst, (bvConst ?: fpConst))
        }.unzip()

        val bvValue = mkAnyArraySelect(bvLambda, fromIndices)

        val targetFpSort = toSort.range
        val fpValue = transformToFpSort(targetFpSort, bvValue)

        val replacement: KExpr<KArraySortBase<*>> = mkAnyArrayLambda(
            indices.map { it.decl }, fpValue
        ).uncheckedCast()
        replacement
    }

    private fun <T : KSort> getInterpretation(
        decl: KDecl<T>, const: KExpr<*>,
    ): KModel.KFuncInterp<*> = when (val sort = decl.sort) {
        is KFpSort -> {
            KModel.KFuncInterp(decl = decl,
                vars = emptyList(),
                entries = emptyList(),
                default = ctx.pack(const.cast(), sort).cast())
        }

        is KArraySortBase<*> -> {
            val array: KExpr<KArraySortBase<*>> = const.cast()
            val transformed: KExpr<*> = transformToFpSort(sort, array)
            KModel.KFuncInterp(decl = decl,
                vars = emptyList(),
                entries = emptyList(),
                default = transformed.cast())
        }

        else -> throw IllegalArgumentException("Unsupported sort: $sort")
    }


    private fun <T : KSort> transformToFpSort(targetFpSort: T, bvExpr: KExpr<T>): KExpr<T> =
        when {
            !sortContainsFP(targetFpSort) -> bvExpr

            targetFpSort is KFpSort -> {
                ctx.pack(bvExpr.cast(), targetFpSort.cast()).cast()
            }

            targetFpSort is KArraySortBase<*> -> {
                when (val array: KExpr<KArraySortBase<*>> = bvExpr.cast()) {
                    is KArrayConst<*, *> -> {
                        val transformedValue: KExpr<KSort> = transformToFpSort(targetFpSort.range, array.value.cast())
                        ctx.mkArrayConst(targetFpSort.cast(), transformedValue)
                    }

                    is KArrayStoreBase<*, *> -> {
                        val indices = array.indices.zip(targetFpSort.domainSorts) { bvIndex, fpSort ->
                            transformToFpSort(fpSort, bvIndex)
                        }
                        val value = transformToFpSort(targetFpSort.range, array.value.cast())
                        val arrayInterpretation = transformToFpSort(targetFpSort, array.array.cast())
                        ctx.mkAnyArrayStore(arrayInterpretation.cast(), indices, value)
                    }

                    is KArrayLambdaBase<*, *> -> transformArrayLambda(array, targetFpSort)

                    else -> throw IllegalArgumentException(
                        "Unsupported array:  class: ${array.javaClass} array.sort ${array.sort}")
                }.cast()
            }

            else -> throw IllegalArgumentException("Unsupported sort: $targetFpSort")
        }

    override fun uninterpretedSortUniverse(sort: KUninterpretedSort): Set<KUninterpretedSortValue>? {
        return kModel.uninterpretedSortUniverse(sort)
    }

    override fun detach(): KModel {
        declarations.forEach {
            interpretation(it)
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
