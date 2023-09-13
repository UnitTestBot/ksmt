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
            println(solver.model())
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