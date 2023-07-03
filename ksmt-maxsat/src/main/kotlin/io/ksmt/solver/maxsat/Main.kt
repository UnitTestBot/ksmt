package io.ksmt.solver.maxsat

import io.ksmt.KContext
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.utils.getValue
import kotlin.time.Duration.Companion.microseconds
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.nanoseconds

suspend fun main() {
    with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)

        val x by intSort
        val y by intSort

        val a1 = x gt 0.expr
        val a2 = x lt y
        val a3 = x + y le 0.expr

        maxSATSolver.assert(a3 eq a1)
        maxSATSolver.assert(a3 or a2)

        maxSATSolver.assertSoft(a3, 3)
        maxSATSolver.assertSoft(!a3, 5)
        maxSATSolver.assertSoft(!a1, 10)
        maxSATSolver.assertSoft(!a2, 3)

        val result = maxSATSolver.checkMaxSAT(1L.milliseconds)
        println("Max SAT succeeded: ${result.maxSATSucceeded}")
        println("Hard constraints SAT status: ${result.hardConstraintsSATStatus}")
        println("Size SAT soft constraints: ${result.satSoftConstraints.size}")
        println("Soft constraints:\n${result.satSoftConstraints.forEach { println(it.expression) }}")
        println("Timeout exceeded:\n${result.timeoutExceeded}")
    }
}
