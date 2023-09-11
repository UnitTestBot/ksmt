package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.solver.z3.KZ3SMTLibParser
import kotlin.test.Test

class BvQETest {
    class LinearTestsWithoutLinearCoefficients {
        private val ctx = KContext()

        @Test
        fun xLessOrEqualY() {
            val formula =
                """
            (declare-fun y () (_ BitVec 4))
            (assert (exists ((x (_ BitVec 4))) (bvule x y)))
            """
            val assertions = KZ3SMTLibParser(ctx).parse(formula)
            println(assertions)
            val qfAssertions = quantifierElimination(ctx, assertions)
            println(qfAssertions)
        }

        @Test
        fun xLessOrEqualZero() {
            val formula =
                """
            (assert (exists ((x (_ BitVec 4))) (bvult x #b0000)))
            """
            val assertions = KZ3SMTLibParser(ctx).parse(formula)
            println(assertions)
            val qfAssertions = quantifierElimination(ctx, assertions)
            println(qfAssertions)
        }

        @Test
        fun xLessOrEqualOne() {
            val formula =
                """
            (assert (exists ((x (_ BitVec 4))) (bvult x #b0001)))
            """
            val assertions = KZ3SMTLibParser(ctx).parse(formula)
            println(assertions)
            val qfAssertions = quantifierElimination(ctx, assertions)
            println(qfAssertions)
        }

        @Test
        fun regularTest0() {
            val formula =
                """
            (declare-fun y () (_ BitVec 4))
            (assert (exists ((x (_ BitVec 4))) (and (bvule x y) (bvuge x #b0010))))
            (assert (bvule y #b1000))
            """
            val assertions = KZ3SMTLibParser(ctx).parse(formula)
            println(assertions)
            val qfAssertions = quantifierElimination(ctx, assertions)
            println(qfAssertions)
        }

        @Test
        fun allInequalities() {
            val formula =
                """
            (declare-fun a () (_ BitVec 4))
            (declare-fun b () (_ BitVec 4))
            (declare-fun c () (_ BitVec 4))
            (declare-fun d () (_ BitVec 4))
            (declare-fun e () (_ BitVec 4))
            (declare-fun f () (_ BitVec 4))
            (declare-fun g () (_ BitVec 4))
            (declare-fun h () (_ BitVec 4))
            
            (assert (exists ((x (_ BitVec 4))) 
            (and (bvule x a) (bvuge b x) (bvuge x c) (bvule d x) 
            (bvult x e) (bvugt f x) (bvugt x g) (bvult h x))))
            """
            val assertions = KZ3SMTLibParser(ctx).parse(formula)
            println(assertions)
            val qfAssertions = quantifierElimination(ctx, assertions)
            println(qfAssertions)
        }

        @Test
        fun notExistsTest() {
            val formula =
                """
            (declare-fun y () (_ BitVec 4))
            (assert (exists ((x (_ BitVec 4))) (and (bvult x y) (bvule y #b0000))))
            """
            val assertions = KZ3SMTLibParser(ctx).parse(formula)
            println(assertions)
            val qfAssertions = quantifierElimination(ctx, assertions)
            println(qfAssertions)
        }
    }

    class LinearTests {
        private val ctx = KContext()

        @Test
        fun linTest0() {
            val formula =
                """
            (declare-fun y () (_ BitVec 4))
            (assert (exists ((x (_ BitVec 4))) (bvult (bvmul x #b0111) y)))
            (assert (bvult (bvnot y) #b1000))
            """
            val assertions = KZ3SMTLibParser(ctx).parse(formula)
            val qfAssertions = quantifierElimination(ctx, assertions)
            println(qfAssertions)
        }
    }

    class ExponentialTests {
        private val ctx = KContext()

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
}