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
        assertEquals(trueExpr, model.eval(fpToBvExpr(mkFpEqualExpr(a, mkFp32(1.6f)))))
        assertEquals(trueExpr, model.eval(fpToBvExpr(mkFpEqualExpr(a, a))))
        assertEquals(falseExpr, model.eval(fpToBvExpr(mkFpEqualExpr(a, mkFp32(1.7f)))))
        assertEquals(falseExpr, model.eval(fpToBvExpr(mkFpEqualExpr(a, mkFpNan(a.sort)))))
        assertEquals(falseExpr, model.eval(fpToBvExpr(mkFpEqualExpr(a, mkFpNan(a.sort)))))

        val positiveInf = mkFpInf(signBit = false, mkFp32Sort())
        val negativeInf = mkFpInf(signBit = true, mkFp32Sort())
        assertEquals(trueExpr, model.eval(fpToBvExpr(mkFpEqualExpr(positiveInf, positiveInf))))
        assertEquals(trueExpr, model.eval(fpToBvExpr(mkFpEqualExpr(negativeInf, negativeInf))))
        assertEquals(falseExpr, model.eval(fpToBvExpr(mkFpEqualExpr(positiveInf, negativeInf))))
        assertEquals(falseExpr, model.eval(fpToBvExpr(mkFpEqualExpr(positiveInf, a))))

        val x by mkFp32Sort()
        assertEquals(mkFpIsNaNExpr(x).not(), model.eval(fpToBvExpr(mkFpEqualExpr(x, x))))
    }
}