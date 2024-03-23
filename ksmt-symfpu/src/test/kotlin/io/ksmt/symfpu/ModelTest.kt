package io.ksmt.symfpu

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFunctionAsArray
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.solver.KModel
import io.ksmt.solver.model.KFuncInterp
import io.ksmt.solver.model.KFuncInterpVarsFree
import io.ksmt.solver.model.KModelImpl
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import io.ksmt.symfpu.solver.FpToBvTransformer
import io.ksmt.symfpu.solver.KSymFpuModel
import io.ksmt.utils.uncheckedCast
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class ModelTest {

    @Test
    fun testModelInterpretationPropagation() = with(KContext()) {
        val baseModel = ModelStub(this)
        val symFpuModel = KSymFpuModel(baseModel, this, FpToBvTransformer(this, packedBvOptimization = true))

        val aInterp = symFpuModel.interpretation(baseModel.aDecl)
        assertEquals(aInterp, baseModel.interpretation(baseModel.aDecl))

        val aFunction = aInterp?.default as? KFunctionAsArray<*, *>
        assertNotNull(aFunction)
        assertEquals(baseModel.fDecl, aFunction.function)

        val fInterp = symFpuModel.interpretation(aFunction.function)
        assertEquals(baseModel.interpretation(aFunction.function), fInterp)
    }

    @Test
    fun testModelDetachPropagation() = with(KContext()) {
        val baseModel = ModelStub(this)
        val symFpuModel = KSymFpuModel(baseModel, this, FpToBvTransformer(this, packedBvOptimization = true))
        assertEquals(baseModel.detach(), symFpuModel)
    }

    class ModelStub(val ctx: KContext) : KModel {
        val aDecl = ctx.mkFreshConstDecl("a", ctx.mkArraySort(ctx.intSort, ctx.intSort))
        val fDecl = ctx.mkFreshFuncDecl("f", ctx.intSort, listOf(ctx.intSort))

        override val declarations: Set<KDecl<*>>
            get() = setOf(aDecl)

        private val interpretations = mutableMapOf<KDecl<*>, KFuncInterp<*>>()

        override fun <T : KSort> interpretation(decl: KDecl<T>): KFuncInterp<T>? =
            interpretations.getOrPut(decl) {
                when (decl) {
                    aDecl -> {
                        interpretations[fDecl] = KFuncInterpVarsFree(decl, emptyList(), ctx.mkIntNum(0).uncheckedCast())
                        KFuncInterpVarsFree(decl, emptyList(), ctx.mkFunctionAsArray(aDecl.sort, fDecl).uncheckedCast())
                    }

                    else -> {
                        error("Unexpected decl: $decl")
                    }
                }
            }.uncheckedCast()

        override fun detach(): KModel {
            declarations.forEach { interpretation(it) }
            return KModelImpl(ctx, interpretations.toMap(), emptyMap())
        }

        override fun close() {
            // ignored
        }

        override val uninterpretedSorts: Set<KUninterpretedSort> = emptySet()

        override fun uninterpretedSortUniverse(sort: KUninterpretedSort): Set<KUninterpretedSortValue>? = null

        override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> {
            error("Unused")
        }
    }
}
