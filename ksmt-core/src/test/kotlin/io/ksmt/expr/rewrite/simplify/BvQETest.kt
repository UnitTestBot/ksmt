package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.solver.z3.KZ3SMTLibParser
import kotlin.test.Test

class BvQETest {
    private val ctx = KContext()

    @Test
    fun simplestTest() {
        val formula =
            """
            (declare-fun y () (_ BitVec 4))
            (assert  (exists ((x (_ BitVec 4))) (bvult x y)))
            """
        val assertions = KZ3SMTLibParser(ctx).parse(formula)
        val qfAssertions = quantifierElimination(ctx, assertions)
        println(qfAssertions)
    }

    @Test
    fun linTest() {
        val formula =
            """
            (declare-fun y () (_ BitVec 4))
            (assert  (exists ((x (_ BitVec 4))) (not (not (bvult (bvmul x #b0111) y)))))
            (assert  (bvult (bvnot y) #b1000))
            """
        val assertions = KZ3SMTLibParser(ctx).parse(formula)
        val qfAssertions = quantifierElimination(ctx, assertions)
        println(qfAssertions)
    }

    @Test
    fun expTest() {
        val formula =
            """
            (declare-fun y () (_ BitVec 4))
            (assert (exists ((x (_ BitVec 4))) (or (bvult (bvshl #b0001 x) y) (= (bvshl #b0001 x) y))))
            """
        val assertions = KZ3SMTLibParser(ctx).parse(formula)
        quantifierElimination(ctx, assertions)
    }
}