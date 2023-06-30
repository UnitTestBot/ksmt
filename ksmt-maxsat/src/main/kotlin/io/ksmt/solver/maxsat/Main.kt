package io.ksmt.solver.maxsat

import io.ksmt.KContext
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.utils.mkConst
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withTimeout

suspend fun main() {
    withTimeout(100) {
        while (isActive) println("Hi")
    }

/*    with (KContext()) {
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
        //val maxSATResult = maxSATSolver.checkMaxSMT()
    }*/
}
