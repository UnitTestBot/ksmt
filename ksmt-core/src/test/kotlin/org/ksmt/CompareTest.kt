package org.ksmt

import org.junit.jupiter.api.Test
import org.ksmt.solver.model.KModelImpl
import org.ksmt.symfpu.Compare.Companion.fpToBvExpr
import org.ksmt.utils.getValue
import kotlin.test.assertEquals

class CompareTest {
    @Test
    fun testDeepExpressionSort(): Unit = with(KContext()) {
        val a = mkFp32(1.6f)
        val model = KModelImpl(
            this,
            interpretations = mapOf(),
            uninterpretedSortsUniverses = emptyMap()
        )
        assertEquals(trueExpr, model.eval(fpToBvExpr(a eq mkFp32(1.6f))))
        assertEquals(trueExpr, model.eval(fpToBvExpr(a eq a)))
        assertEquals(falseExpr, model.eval(fpToBvExpr(a eq mkFp32(1.7f))))
        assertEquals(falseExpr, model.eval(fpToBvExpr(a eq mkFpNan(a.sort))))
        assertEquals(falseExpr, model.eval(fpToBvExpr(a eq mkFpNan(a.sort))))

        val positiveInf = mkFpInf(signBit = false, mkFp32Sort())
        val negativeInf = mkFpInf(signBit = true, mkFp32Sort())
        assertEquals(trueExpr, model.eval(fpToBvExpr(positiveInf eq positiveInf)))
        assertEquals(trueExpr, model.eval(fpToBvExpr(negativeInf eq negativeInf)))
        assertEquals(falseExpr, model.eval(fpToBvExpr(positiveInf eq negativeInf)))
        assertEquals(falseExpr, model.eval(fpToBvExpr(positiveInf eq a)))

        val x by mkFp32Sort()
        assertEquals(mkFpIsNaNExpr(x).not(), model.eval(fpToBvExpr(x eq x)))
    }
}