package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.z3.KZ3SMTLibParser
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.sort.KBoolSort
import kotlin.test.Test


fun xorEquivalenceCheck(ctx: KContext,
                        quantifierAssertions: List<KExpr<KBoolSort>>,
                        quantifierFreeAssertions: List<KExpr<KBoolSort>>):
        Boolean = with(ctx) {
    for ((i, qAssertion) in quantifierAssertions.withIndex()) {
        val qfAssertion = quantifierFreeAssertions[i]
        val xorExpr = mkXor(qAssertion, qfAssertion)

        val solver = KZ3Solver(ctx)
        solver.assert(xorExpr)
        val status = solver.check()
        if (status == KSolverStatus.SAT) {
            print("\n${solver.model()}\n")
            return false
        }
    }
    return true
}

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
            assert(xorEquivalenceCheck(ctx, assertions, qfAssertions))
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
            assert(xorEquivalenceCheck(ctx, assertions, qfAssertions))
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
            assert(xorEquivalenceCheck(ctx, assertions, qfAssertions))
        }

        @Test
        fun test0() {
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
            assert(xorEquivalenceCheck(ctx, assertions, qfAssertions))
        }

        @Test
        fun test1() {
            val formula =
                """
            (declare-fun y () (_ BitVec 4))
            
            (assert (exists ((x (_ BitVec 4))) (and (bvule x y) (bvult x #b0010))))
            """
            val assertions = KZ3SMTLibParser(ctx).parse(formula)
            println(assertions)
            val qfAssertions = quantifierElimination(ctx, assertions)
            println(qfAssertions)
            assert(xorEquivalenceCheck(ctx, assertions, qfAssertions))
        }

        @Test
        fun test2() {
            val formula =
                """
            (declare-fun y () (_ BitVec 4))
            
            (assert (exists ((x (_ BitVec 4))) (and (bvule x y) (bvule (bvmul y #b1000) x))))
            """
            val assertions = KZ3SMTLibParser(ctx).parse(formula)
            println(assertions)
            val qfAssertions = quantifierElimination(ctx, assertions)
            println(qfAssertions)
            assert(xorEquivalenceCheck(ctx, assertions, qfAssertions))
        }

        @Test
        fun threeLess() {
            val formula =
                """
            (declare-fun a () (_ BitVec 4))
            (declare-fun b () (_ BitVec 4))
            (declare-fun c () (_ BitVec 4))
            
            (assert (exists ((x (_ BitVec 4))) (and (bvule x a) (bvuge b x) (bvult x c))))
            """
            val assertions = KZ3SMTLibParser(ctx).parse(formula)
            println(assertions)
            val qfAssertions = quantifierElimination(ctx, assertions)
            println(qfAssertions)
            assert(xorEquivalenceCheck(ctx, assertions, qfAssertions))
        }

        @Test
        fun twoLess() {
            val formula =
                """
            (declare-fun a () (_ BitVec 4))
            (declare-fun b () (_ BitVec 4))
            
            (assert (exists ((x (_ BitVec 4))) (and (bvult x a) (bvugt b x))))
            """
            val assertions = KZ3SMTLibParser(ctx).parse(formula)
            println(assertions)
            val qfAssertions = quantifierElimination(ctx, assertions)
            println(qfAssertions)
            assert(xorEquivalenceCheck(ctx, assertions, qfAssertions))
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
            assert(xorEquivalenceCheck(ctx, assertions, qfAssertions))
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
            assert(xorEquivalenceCheck(ctx, assertions, qfAssertions))
        }
    }

    class TestsWithSameLinearCoefficient {
        private val ctx = KContext()

        @Test
        fun test0() {
            val formula =
                """
            (declare-fun y () (_ BitVec 4))
            (assert (exists ((x (_ BitVec 4))) (bvule y (bvmul x #b0111))))
            (assert (bvule #b1111 y))
            """
            val assertions = KZ3SMTLibParser(ctx).parse(formula)
            println(assertions)
            val qfAssertions = quantifierElimination(ctx, assertions)
            println(qfAssertions)
            assert(xorEquivalenceCheck(ctx, assertions, qfAssertions))
        }

        @Test
        fun test1() {
            val formula =
                """
            (declare-fun y () (_ BitVec 4))
            (assert (exists ((x (_ BitVec 4))) (bvule y (bvmul x #b1000))))
            (assert (bvule #b1111 y))
            """
            val assertions = KZ3SMTLibParser(ctx).parse(formula)
            println(assertions)
            val qfAssertions = quantifierElimination(ctx, assertions)
            println(qfAssertions)
            assert(xorEquivalenceCheck(ctx, assertions, qfAssertions))
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
            (let ((x15 (bvmul x #b1111)))
            (and (bvule x15 a) (bvuge b x15) (bvuge x15 c) (bvule d x15) 
            (bvult x15 e) (bvugt f x15) (bvugt x15 g) (bvult h x15)))))
            """
            val assertions = KZ3SMTLibParser(ctx).parse(formula)
            println(assertions)
            val qfAssertions = quantifierElimination(ctx, assertions)
            println(qfAssertions)
            assert(xorEquivalenceCheck(ctx, assertions, qfAssertions))
        }
    }

    class TestsWithDifferentLinearCoefficient {
        private val ctx = KContext()

        @Test
        fun falseTest() {
            val formula =
                """
            (assert (exists ((x (_ BitVec 4))) 
            (and (bvugt x #b0001) (bvult (bvmul x #b0010) #b0011) (bvuge (bvmul x #b0100) #b0101))))
            """
            val assertions = KZ3SMTLibParser(ctx).parse(formula)
            println(assertions)
            val qfAssertions = quantifierElimination(ctx, assertions)
            println(qfAssertions)
            assert(xorEquivalenceCheck(ctx, assertions, qfAssertions))
        }

        @Test
        fun test() {
            val formula =
                """
            (declare-fun y () (_ BitVec 4))
            
            (assert (exists ((x (_ BitVec 4))) 
            (and (bvugt x #b0001) (bvult (bvmul x #b0010) #b0011) (bvuge (bvmul x #b0100) y))))
            """
            val assertions = KZ3SMTLibParser(ctx).parse(formula)
            println(assertions)
            val qfAssertions = quantifierElimination(ctx, assertions)
            println(qfAssertions)
            assert(xorEquivalenceCheck(ctx, assertions, qfAssertions))
        }
    }

    class ExponentialTests {
        private val ctx = KContext()

        @Test
        fun test() {
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