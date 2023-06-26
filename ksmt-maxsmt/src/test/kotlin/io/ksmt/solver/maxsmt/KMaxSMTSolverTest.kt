package io.ksmt.solver.maxsmt

import io.ksmt.KContext
import io.ksmt.expr.KNotExpr
import io.ksmt.expr.KOrBinaryExpr
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.utils.mkConst
import org.junit.jupiter.api.Test

class KMaxSMTSolverTest {
    @Test
    fun smokeTest() {
        with (KContext()) {
            val z3Solver = KZ3Solver(this)
            val maxSMTSolver = KMaxSMTSolver(this, z3Solver)
            val a = boolSort.mkConst("a")
            val b = boolSort.mkConst("b")
            val c = boolSort.mkConst("c")
            maxSMTSolver.assert(a)
            maxSMTSolver.assert(b)
            maxSMTSolver.assert(c)
            maxSMTSolver.assertSoft(mkAnd(a, mkNot(c)), 1)
            maxSMTSolver.assertSoft(mkNot(a), 1)
            val (model, iter) = maxSMTSolver.checkMaxSMT()
            println("Model:\n$model")
            println("Finished on $iter iteration")
        }
    }

    @Test
    fun smokeTest2() {
        with (KContext()) {
            val z3Solver = KZ3Solver(this)
            val maxSMTSolver = KMaxSMTSolver(this, z3Solver)
            val a = boolSort.mkConst("a")
            val b = boolSort.mkConst("b")
            maxSMTSolver.assert(KOrBinaryExpr(this, a, b))
            maxSMTSolver.assert(KOrBinaryExpr(this, KNotExpr(this, a), b))
            maxSMTSolver.assertSoft(KOrBinaryExpr(this, a, KNotExpr(this, b)), 1)
            maxSMTSolver.assertSoft(KOrBinaryExpr(this, KNotExpr(this, a), KNotExpr(this, b)), 1)
            val (model, iter) = maxSMTSolver.checkMaxSMT()
            println("Model:\n$model")
            println("Finished on $iter iteration")
        }
    }
}