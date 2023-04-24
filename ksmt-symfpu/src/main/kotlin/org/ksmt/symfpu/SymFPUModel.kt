package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KConst
import org.ksmt.expr.KExpr
import org.ksmt.expr.KUninterpretedSortValue
import org.ksmt.solver.KModel
import org.ksmt.solver.model.KModelEvaluator
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.symfpu.ArraysTransform.Companion.mkAnyArrayLambda
import org.ksmt.symfpu.ArraysTransform.Companion.mkAnyArraySelect
import org.ksmt.symfpu.ArraysTransform.Companion.transformSortRemoveFP
import org.ksmt.utils.cast
import org.ksmt.utils.uncheckedCast

class SymFPUModel(private val kModel: KModel, val ctx: KContext, val transformer: FpToBvTransformer) : KModel {
    override val declarations: Set<KDecl<*>>
        get() = kModel.declarations

    override val uninterpretedSorts: Set<KUninterpretedSort>
        get() = kModel.uninterpretedSorts

    private val evaluatorWithModelCompletion by lazy { KModelEvaluator(ctx, this, isComplete = true) }
    private val evaluatorWithoutModelCompletion by lazy { KModelEvaluator(ctx, this, isComplete = false) }
    private val interpretations: MutableMap<KDecl<*>, KModel.KFuncInterp<*>> = hashMapOf()

    override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> {
        ctx.ensureContextMatch(expr)

        val evaluator = if (isComplete) evaluatorWithModelCompletion else evaluatorWithoutModelCompletion
        println("eval $expr")
        val res = evaluator.apply(expr)
        println("eval $expr done: $res")
        return res
    }

    private fun <T : KSort> getConst(decl: KDecl<T>): KExpr<*>? = with(ctx) {
        if (sortContainsFP(decl.sort)) {
            transformer.arraysTransform.mapFpToBvDeclImpl.getOrPut(decl) {
                mkFreshConst(decl.name + "!tobv_transform!", transformSortRemoveFP(decl.sort)).cast()
            }.cast()
        } else null
    }

    override fun <T : KSort> interpretation(decl: KDecl<T>): KModel.KFuncInterp<T>? {
        println("DECL::$decl")
        ctx.ensureContextMatch(decl)
        if (!sortContainsFP(decl.sort)) return kModel.interpretation(decl).also { println("INTERP::$decl ==>> $it") }
        // todo (it is KFuncDecl) works ??, quantifier, lambda
//        0_00000001_10000000000000000000000
        val const = kModel.eval(getConst(decl)!!) //?: return kModel.interpretation(decl)
        val interp = getInterpretation(decl, const)
        println("INTERP::$decl ==>> $interp")
        return interp.cast()
    }

    private fun transformArray(
        bvArray: KExpr<KArraySortBase<*>>, toSort: KArraySortBase<*>,
    ): KExpr<*> = with(ctx) {
        val fromSort = bvArray.sort

        if (fromSort == toSort) {
            return@with bvArray.uncheckedCast()
        }

        // fp indices
        val indices: List<KConst<KSort>> = toSort.domainSorts.map {
            mkFreshConst("i", it).cast()
        }
        //bv indices
        val fromIndices: List<KConst<KSort>> =
            indices.zip(fromSort.domainSorts) { idx: KConst<KSort>, bvSort: KSort ->
                val curSort = idx.sort
                getConstOfNonFPSort(curSort, idx, bvSort)
            }
        // bv value
        val value = mkAnyArraySelect(bvArray, fromIndices)
        //fp value
        val targetFpSort = toSort.range
        val toValue = transformToFpSort(targetFpSort, value)

        val replacement: KExpr<KArraySortBase<*>> = mkAnyArrayLambda(
            indices.map { it.decl }, toValue
        ).uncheckedCast()
        replacement
    }

    fun KContext.getConstOfNonFPSort(curSort: KSort, idx: KConst<KSort>, bvSort: KSort): KConst<KSort> =
        if (sortContainsFP(curSort)) {
            transformer.arraysTransform.mapFpToBvDeclImpl.getOrPut(idx.decl.cast()) {
                mkFreshConst(idx.decl.name + "!tobv_transform!", bvSort).cast()
            }.cast()
        } else idx

    private fun <T : KSort> getInterpretation(
        decl: KDecl<T>, const: KExpr<*>,
    ): KModel.KFuncInterp<*> = interpretations.getOrPut(decl) {
        return when (val sort = decl.sort) {
            is KFpSort -> {
                KModel.KFuncInterp(decl = decl,
                    vars = emptyList(),
                    entries = emptyList(),
                    default = ctx.packUnbiased(const.cast(), sort).cast())
            }

            is KArraySortBase<*> -> {
                val array: KExpr<KArraySortBase<*>> = const.cast()
                val origDecl: KDecl<KArraySortBase<*>> = decl.cast()
                val transformed = transformArray(array, origDecl.sort)
                KModel.KFuncInterp(decl = decl,
                    vars = emptyList(),
                    entries = emptyList(),
                    default = transformed.cast())
            }

            else -> throw IllegalArgumentException("Unsupported sort: $sort")
        }
    }

    private fun transformToFpSort(targetFpSort: KSort, value: KExpr<KSort>) =
        when (targetFpSort) {
            is KFpSort -> {
                ctx.pack(value.cast(), targetFpSort.cast())
            }

            is KArraySortBase<*> -> {
                transformArray(value.cast(), targetFpSort)
            }

            else -> value
        }

    override fun uninterpretedSortUniverse(sort: KUninterpretedSort): Set<KUninterpretedSortValue>? {
        return kModel.uninterpretedSortUniverse(sort)
    }

    override fun detach(): KModel {
        return SymFPUModel(kModel.detach(), ctx, transformer)
    }

    companion object {
        fun sortContainsFP(curSort: KSort): Boolean {
            return when (curSort) {
                is KFpSort -> true
                is KArraySortBase<*> -> curSort.domainSorts.any { sortContainsFP(it) } || sortContainsFP(curSort.range)
                else -> false
            }
        }
    }
}
