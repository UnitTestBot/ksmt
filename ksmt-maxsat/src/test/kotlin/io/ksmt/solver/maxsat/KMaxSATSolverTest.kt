package io.ksmt.solver.maxsat

import io.ksmt.KContext
import io.ksmt.expr.KNotExpr
import io.ksmt.expr.KOrBinaryExpr
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.utils.getValue
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class KMaxSATSolverTest {
    @Test
    fun noSoftConstraintsSATTest() = with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)
        val a by boolSort
        val b by boolSort
        val c by boolSort
        maxSATSolver.assert(a)
        maxSATSolver.assert(b)
        maxSATSolver.assert(c)
        maxSATSolver.assertSoft(mkAnd(a, mkNot(c)), 3)
        maxSATSolver.assertSoft(mkNot(a), 5)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == KSolverStatus.SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.isEmpty())
    }

    @Test
    fun oneOfTwoSoftConstraintsSATTest() = with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)
        val a by boolSort
        val b by boolSort

        val notA = KNotExpr(this, a)
        val notB =  KNotExpr(this, b)

        maxSATSolver.assert(KOrBinaryExpr(this, a, b))
        maxSATSolver.assert(KOrBinaryExpr(this, notA, b))
        maxSATSolver.assertSoft(KOrBinaryExpr(this, a, notB), 2)

        val notAOrNotBExpr = KOrBinaryExpr(this, notA, notB)
        val notAOrNotBWeight = 3
        maxSATSolver.assertSoft(notAOrNotBExpr, notAOrNotBWeight)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == KSolverStatus.SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.size == 1)
        assertSATSoftConstraints(listOf(SoftConstraint(notAOrNotBExpr, notAOrNotBWeight)), maxSATResult.satSoftConstraints)
    }

    @Test
    fun twoOfThreeSoftConstraintsSATTest() = with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)
        val a by boolSort
        val b by boolSort

        val notA = KNotExpr(this, a)
        val notB =  KNotExpr(this, b)

        maxSATSolver.assert(KOrBinaryExpr(this, a, b))

        val notAOrBExpr = KOrBinaryExpr(this, notA, b)
        val notAOrBWeight = 4
        maxSATSolver.assertSoft(notAOrBExpr, notAOrBWeight)

        val aOrNotBExpr = KOrBinaryExpr(this, a, notB)
        val aOrNotBWeight = 6
        maxSATSolver.assertSoft(aOrNotBExpr, aOrNotBWeight)

        maxSATSolver.assertSoft(KOrBinaryExpr(this, notA, notB), 2)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == KSolverStatus.SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.size == 2)
        val softConstraintsToAssertSAT =
                listOf(SoftConstraint(notAOrBExpr, notAOrBWeight), SoftConstraint(aOrNotBExpr, aOrNotBWeight))
        assertSATSoftConstraints(softConstraintsToAssertSAT, maxSATResult.satSoftConstraints)
    }

    @Test
    fun sameExpressionSoftConstraintsSATTest() = with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)

        val x by boolSort
        val y by boolSort

        val notX = KNotExpr(this, x)
        val notY =  KNotExpr(this, y)

        maxSATSolver.assert(KOrBinaryExpr(this, x , y))

        val notXOrYExpr = KOrBinaryExpr(this, notX, y)
        val notXOrYWeight = 7
        maxSATSolver.assertSoft(notXOrYExpr, notXOrYWeight)

        maxSATSolver.assertSoft(KOrBinaryExpr(this, x, notY), 6)

        val notXOrNotYExpr = KOrBinaryExpr(this, notX, notY)
        maxSATSolver.assertSoft(notXOrNotYExpr, 3)
        maxSATSolver.assertSoft(notXOrNotYExpr, 4)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == KSolverStatus.SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.size == 3)
        assertSATSoftConstraints(listOf(SoftConstraint(notXOrYExpr, notXOrYWeight), SoftConstraint(notXOrNotYExpr, 3),
                SoftConstraint(notXOrNotYExpr, 4)), maxSATResult.satSoftConstraints)
    }

    @Test
    fun sameExpressionSoftConstraintsUNSATTest() = with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)

        val x by boolSort
        val y by boolSort

        val notX = KNotExpr(this, x)
        val notY =  KNotExpr(this, y)

        maxSATSolver.assert(KOrBinaryExpr(this, x , y))

        val notXOrYExpr = KOrBinaryExpr(this, notX, y)
        val notXOrYExprWeight = 6
        maxSATSolver.assertSoft(notXOrYExpr, notXOrYExprWeight)

        val xOrNotYExpr = KOrBinaryExpr(this, x, notY)
        val xOrNotYWeight = 6
        maxSATSolver.assertSoft(xOrNotYExpr, xOrNotYWeight)

        maxSATSolver.assertSoft(KOrBinaryExpr(this, notX, notY), 3)
        maxSATSolver.assertSoft(KOrBinaryExpr(this, notX, notY), 2)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == KSolverStatus.SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.size == 2)
        assertSATSoftConstraints(listOf(SoftConstraint(notXOrYExpr, notXOrYExprWeight), SoftConstraint(xOrNotYExpr, xOrNotYWeight)),
                maxSATResult.satSoftConstraints)
    }

    private fun assertSATSoftConstraints(constraintsToAssert: List<SoftConstraint>,
                                         satConstraints: List<SoftConstraint>) {
        for (constraint in constraintsToAssert) {
            assertTrue(satConstraints.any { constraint.constraint.internEquals(it.constraint) && constraint.weight == it.weight })
        }
    }
}
