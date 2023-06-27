package io.ksmt.solver.maxsat

import io.ksmt.KContext
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.utils.mkConst

fun main() {
    test()
}

fun test() = with(KContext()) {
    val z3Solver = KZ3Solver(this)
    val maxSATSolver = KMaxSATSolver(this, z3Solver)
    val a = boolSort.mkConst("a")
    val b = boolSort.mkConst("b")
    val c = boolSort.mkConst("c")
    maxSATSolver.assert(a)
    maxSATSolver.assert(b)
    maxSATSolver.assert(c)
    maxSATSolver.assertSoft(mkAnd(a, mkNot(c)), 1)
    maxSATSolver.assertSoft(mkNot(a), 1)
    val (model, iter) = maxSATSolver.checkMaxSMT()
    println("Model:\n$model")
    println("Finished on $iter iteration")

/*    val aIsTrue = mkEq(a, mkTrue())
    val bIsTrue = mkEq(b, mkTrue())
    val cIsTrue = mkEq(c, mkTrue())

    z3Solver.assert(aIsTrue)
    z3Solver.assert(bIsTrue)
    z3Solver.assert(cIsTrue)
    z3Solver.assert(mkAnd(aIsTrue, mkNot(cIsTrue)))
    z3Solver.assert(mkEq(a, mkFalse()))
    val status = z3Solver.check()
    if (status == KSolverStatus.UNSAT) {
        println(status)
        val unsatCore = z3Solver.unsatCore()
        println("Unsat core length: ${unsatCore.size}")
        unsatCore.forEach { x -> println(x) }
    }*/
}
