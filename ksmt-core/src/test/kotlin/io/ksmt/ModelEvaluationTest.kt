package io.ksmt

import io.ksmt.solver.model.KFuncInterpEntryVarsFree
import io.ksmt.solver.model.KFuncInterpVarsFree
import io.ksmt.solver.model.KModelImpl
import io.ksmt.utils.getValue
import io.ksmt.utils.sampleValue
import kotlin.test.Test
import kotlin.test.assertEquals

class ModelEvaluationTest {

    @Test
    fun testModelEvalSimple(): Unit = with(KContext()) {
        val arraySort = mkArraySort(bv32Sort, bv32Sort)
        val array by arraySort
        val idx = 17.toBv(bv32Sort)
        val value = 42.toBv(bv32Sort)
        val baseValue = 137.toBv(bv32Sort)

        val arrayInterp = mkArrayConst(arraySort, baseValue).store(idx, value)
        val exprIdx = array.select(idx)
        val exprBase = array.select(99.toBv(bv32Sort))

        val model = KModelImpl(
            this,
            interpretations = mapOf(
                array.decl to KFuncInterpVarsFree(
                    decl = array.decl,
                    entries = emptyList(),
                    default = arrayInterp
                ),
            ),
            uninterpretedSortsUniverses = emptyMap()
        )

        assertEquals(value, model.eval(exprIdx))
        assertEquals(baseValue, model.eval(exprBase))
    }

    @Test
    fun testModelEvalPartialArray(): Unit = with(KContext()) {
        val arraySort = mkArraySort(bv32Sort, bv32Sort)
        val array by arraySort
        val idx = 17.toBv(bv32Sort)
        val value = 42.toBv(bv32Sort)

        val exprIdx = array.select(idx)
        val exprBase = array.select(99.toBv(bv32Sort))

        val tmpDecl = mkFreshFuncDecl("array", bv32Sort, listOf(bv32Sort))
        val tmpInterp = KFuncInterpVarsFree(
            decl = tmpDecl,
            entries = listOf(
                KFuncInterpEntryVarsFree.create(
                    args = listOf(idx),
                    value = value
                )
            ),
            default = null
        )

        val arrayInterp = mkFunctionAsArray(arraySort, tmpDecl)


        val model = KModelImpl(
            this,
            interpretations = mapOf(
                array.decl to KFuncInterpVarsFree(
                    decl = array.decl,
                    entries = listOf(),
                    default = arrayInterp
                ),
                tmpDecl to tmpInterp
            ),
            uninterpretedSortsUniverses = emptyMap()
        )

        assertEquals(value, model.eval(exprIdx))

        val defaultBaseValue = bv32Sort.sampleValue()
        assertEquals(defaultBaseValue, model.eval(exprBase))
    }

    @Test
    fun testModelEvalPartialArrayEquality(): Unit = with(KContext()) {
        val arraySort = mkArraySort(bv32Sort, bv32Sort)
        val array1 by arraySort
        val array2 by arraySort
        val idx = 17.toBv(bv32Sort)
        val value = 42.toBv(bv32Sort)

        val arrayEquality = array1 eq array2

        val tmpDecl1 = mkFreshFuncDecl("array1", bv32Sort, listOf(bv32Sort))
        val tmpInterp1 = KFuncInterpVarsFree(
            decl = tmpDecl1,
            entries = listOf(
                KFuncInterpEntryVarsFree.create(
                    args = listOf(idx),
                    value = value
                )
            ),
            default = null
        )

        val tmpDecl2 = mkFreshFuncDecl("array2", bv32Sort, listOf(bv32Sort))
        val tmpInterp2 = KFuncInterpVarsFree(
            decl = tmpDecl2,
            entries = listOf(
                KFuncInterpEntryVarsFree.create(
                    args = listOf(idx),
                    value = value
                )
            ),
            default = null
        )

        val array1Interp = mkFunctionAsArray(arraySort, tmpDecl1)
        val array2Interp = mkFunctionAsArray(arraySort, tmpDecl2)


        val model = KModelImpl(
            this,
            interpretations = mapOf(
                array1.decl to KFuncInterpVarsFree(
                    decl = array1.decl,
                    entries = listOf(),
                    default = array1Interp
                ),
                array2.decl to KFuncInterpVarsFree(
                    decl = array2.decl,
                    entries = listOf(),
                    default = array2Interp
                ),
                tmpDecl1 to tmpInterp1,
                tmpDecl2 to tmpInterp2,
            ),
            uninterpretedSortsUniverses = emptyMap()
        )

        assertEquals(trueExpr, model.eval(arrayEquality))
    }

}
