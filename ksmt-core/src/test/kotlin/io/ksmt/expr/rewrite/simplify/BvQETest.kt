package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KExistentialQuantifier
import io.ksmt.expr.KExpr
import io.ksmt.solver.z3.KZ3SMTLibParser
import io.ksmt.sort.KBoolSort
import kotlin.test.Test

class BvQETest {
    private val ctx = KContext()

    @Test
    fun mainTest() {

        val formula =
            """
            (declare-fun y () (_ BitVec 4))
            (assert  (exists ((x (_ BitVec 4))) (not (not (bvult (bvmul x #b0111) y)))))
            (assert  (bvult (bvnot y) #b1000))
            """

        val assertions = KZ3SMTLibParser(ctx).parse(formula)

        val boundAssertions = arrayListOf<KExistentialQuantifier>()
        val notAssertions = arrayListOf<Boolean>()
        val freeAssertions = arrayListOf<KExpr<KBoolSort>>()

        qePreprocess(ctx, assertions, boundAssertions, notAssertions, freeAssertions)
        var qfAssertions = arrayListOf<KExpr<KBoolSort>>()
        for (assert in boundAssertions)
        {
            val qfAssertion = qeProcess(ctx, assert)
            qfAssertions.add(qfAssertion)
        }
    }
}