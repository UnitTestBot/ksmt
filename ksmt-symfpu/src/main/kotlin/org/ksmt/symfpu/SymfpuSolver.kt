package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.expr.KConst
import org.ksmt.expr.KExpr
import org.ksmt.expr.KUninterpretedSortValue
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.model.KModelEvaluator
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.symfpu.ArraysTransform.Companion.mkAnyArrayLambda
import org.ksmt.symfpu.ArraysTransform.Companion.mkAnyArraySelect
import org.ksmt.utils.cast
import org.ksmt.utils.uncheckedCast
import kotlin.time.Duration

open class SymfpuSolver<Config : KSolverConfiguration>(
    val solver: KSolver<Config>,
    val ctx: KContext,
) : KSolver<Config> {

    private val transformer = FpToBvTransformer(ctx)

    override fun configure(configurator: Config.() -> Unit) {
        solver.configure(configurator)
    }

    override fun assert(expr: KExpr<KBoolSort>) = solver.assert(transformer.applyAndGetExpr(expr)) // AndGetExpr

    override fun assertAndTrack(expr: KExpr<KBoolSort>, trackVar: KConstDecl<KBoolSort>) =
        solver.assertAndTrack(transformer.applyAndGetExpr(expr), trackVar)

    override fun push() = solver.push()

    override fun pop(n: UInt) = solver.pop(n)


    override fun check(timeout: Duration): KSolverStatus = solver.check(timeout)

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus =
        solver.checkWithAssumptions(assumptions.map(transformer::applyAndGetExpr), timeout)

    override fun model(): KModel = Model(solver.model())

    override fun unsatCore(): List<KExpr<KBoolSort>> = solver.unsatCore()

    override fun reasonOfUnknown(): String = solver.reasonOfUnknown()

    override fun interrupt() = solver.interrupt()

    override fun close() = solver.close()

    inner class Model(private val kModel: KModel) : KModel {
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
            return evaluator.apply(expr)
        }

        private fun <T : KSort> getConst(decl: KDecl<T>): KExpr<*>? = when (decl.sort) {
            is KFpSort -> {
                transformer.arraysTransform.mapFpToBvDeclImpl[decl.cast()]
            }

            is KArraySortBase<*> -> {
                transformer.arraysTransform.mapFpArrayToBvImpl[decl.cast()]
            }

            else -> null
        }

        override fun <T : KSort> interpretation(decl: KDecl<T>): KModel.KFuncInterp<T>? {
            ctx.ensureContextMatch(decl)

            val const = getConst(decl) ?: return kModel.interpretation(decl)
            return getInterpretation(decl, const).cast()
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

        private fun <T : KSort> getInterpretation(
            decl: KDecl<T>, const: KExpr<*>,
        ): KModel.KFuncInterp<*> = interpretations.getOrPut(decl) {
            return when (val sort = decl.sort) {
                is KFpSort -> {
                    KModel.KFuncInterp(decl = decl,
                        vars = emptyList(),
                        entries = emptyList(),
                        default = ctx.pack(const.cast(), sort).cast())
                }

                is KArraySortBase<*> -> {
                    val array: KConst<KArraySortBase<*>> = const.cast()
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
            return Model(kModel.detach())
        }
    }



    private fun KContext.getConstOfNonFPSort(curSort: KSort, idx: KConst<KSort>, bvSort: KSort): KConst<KSort> =
        if (sortContainsFP(curSort)) {
            transformer.arraysTransform.mapFpToBvDeclImpl.getOrPut(idx.decl.cast()) {
                mkFreshConst(idx.decl.name + "!tobv_transform!", bvSort).cast()
            }.cast()
        } else idx

    private fun KContext.sortContainsFP(curSort: KSort): Boolean {
        return when (curSort) {
            is KFpSort -> true
            is KArraySortBase<*> -> curSort.domainSorts.any { sortContainsFP(it) } || sortContainsFP(curSort.range)
            else -> false
        }
    }

}
