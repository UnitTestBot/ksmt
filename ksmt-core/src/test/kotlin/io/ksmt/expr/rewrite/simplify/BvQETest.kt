package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.solver.z3.KZ3SMTLibParser
import kotlin.test.Test

class BvQETest {
    private val ctx = KContext()

    @Test
    fun linTestWithoutLinCoef0() {
        val formula =
            """
            (declare-fun y () (_ BitVec 4))
            (assert  (exists ((x (_ BitVec 4))) (bvule x y)))
            """
        val assertions = KZ3SMTLibParser(ctx).parse(formula)
        println(assertions)
        val qfAssertions = quantifierElimination(ctx, assertions)
        println(qfAssertions)
    }

    @Test
    fun linTestWithoutLinCoef1() {
        val formula =
            """
            (assert  (exists ((x (_ BitVec 4))) (bvult x #b0000)))
            """
        val assertions = KZ3SMTLibParser(ctx).parse(formula)
        println(assertions)
        val qfAssertions = quantifierElimination(ctx, assertions)
        println(qfAssertions)
    }

    @Test
    fun linTestWithoutLinCoef2() {
        val formula =
            """
            (assert  (exists ((x (_ BitVec 4))) (bvult x #b0001)))
            """
        val assertions = KZ3SMTLibParser(ctx).parse(formula)
        println(assertions)
        val qfAssertions = quantifierElimination(ctx, assertions)
        println(qfAssertions)
    }

    @Test
    fun linTestWithoutLinCoef3() {
        val formula =
            """
            (declare-fun y () (_ BitVec 4))
            (assert  (exists ((x (_ BitVec 4))) (and (bvule x y) (bvuge x #b0010))))
            (assert  (bvule y #b1000))
            """
        val assertions = KZ3SMTLibParser(ctx).parse(formula)
        println(assertions)
        val qfAssertions = quantifierElimination(ctx, assertions)
        println(qfAssertions)
    }

    @Test
    fun linTestWithoutLinCoef4() {
        val formula =
            """
            (declare-fun y () (_ BitVec 4))
            (assert  (exists ((x (_ BitVec 4))) (and (bvult x y) (bvule y #b0000))))
            """
        val assertions = KZ3SMTLibParser(ctx).parse(formula)
        println(assertions)
        val qfAssertions = quantifierElimination(ctx, assertions)
        println(qfAssertions)
    }

    @Test
    fun linTest0() {
        val formula =
            """
            (declare-fun y () (_ BitVec 4))
            (assert  (exists ((x (_ BitVec 4))) (bvult (bvmul x #b0111) y)))
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