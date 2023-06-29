package io.ksmt.solver.maxsat

import io.ksmt.KContext
import io.ksmt.expr.KAndBinaryExpr
import io.ksmt.expr.KAndNaryExpr
import io.ksmt.expr.KNotExpr
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.utils.getValue
import io.ksmt.utils.mkConst
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
        maxSATSolver.assertSoft(a and !c, 3)
        maxSATSolver.assertSoft(!a, 5)

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

        maxSATSolver.assert(a or b)
        maxSATSolver.assert(!a or b)

        maxSATSolver.assertSoft(a or !b, 2)
        maxSATSolver.assertSoft(!a or !b, 3)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == KSolverStatus.SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.size == 1)
        assertSoftConstraintsSAT(listOf(SoftConstraint(!a or !b, 3)), maxSATResult.satSoftConstraints)
    }

    @Test
    fun twoOfThreeSoftConstraintsSATTest() = with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)
        val a by boolSort
        val b by boolSort

        maxSATSolver.assert(a or b)

        maxSATSolver.assertSoft(!a or b, 4)
        maxSATSolver.assertSoft(a or !b, 6)
        maxSATSolver.assertSoft(!a or !b, 2)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == KSolverStatus.SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.size == 2)
        val softConstraintsToAssertSAT =
                listOf(SoftConstraint(!a or b, 4), SoftConstraint(a or !b, 6))
        assertSoftConstraintsSAT(softConstraintsToAssertSAT, maxSATResult.satSoftConstraints)
    }

    @Test
    fun sameExpressionSoftConstraintsSATTest() = with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)

        val x by boolSort
        val y by boolSort

        maxSATSolver.assert(x or y)

        maxSATSolver.assertSoft(!x or y, 7)
        maxSATSolver.assertSoft(x or !y, 6)
        maxSATSolver.assertSoft(!x or !y, 3)
        maxSATSolver.assertSoft(!x or !y, 4)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == KSolverStatus.SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.size == 3)
        assertSoftConstraintsSAT(listOf(SoftConstraint(!x or y, 7), SoftConstraint(!x or !y, 3),
                SoftConstraint(!x or !y, 4)), maxSATResult.satSoftConstraints)
    }

    @Test
    fun sameExpressionSoftConstraintsUNSATTest() = with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)

        val x by boolSort
        val y by boolSort

        maxSATSolver.assert(x or y)

        maxSATSolver.assertSoft(!x or y, 6)
        maxSATSolver.assertSoft(x or !y, 6)
        maxSATSolver.assertSoft(!x or !y, 3)
        maxSATSolver.assertSoft(!x or !y, 2)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == KSolverStatus.SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.size == 2)
        assertSoftConstraintsSAT(listOf(SoftConstraint(!x or y, 6), SoftConstraint(x or !y, 6)),
                maxSATResult.satSoftConstraints)
    }

    @Test
    fun smokeTest5() = with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)
        val z = boolSort.mkConst("z")
        val a = boolSort.mkConst("a")
        val b = boolSort.mkConst("b")

        maxSATSolver.assert(z)
        maxSATSolver.assertSoft(KAndBinaryExpr(this, a, b), 1)
        val constr = KAndBinaryExpr(this, KNotExpr(this, a), KNotExpr(this, b))
        maxSATSolver.assertSoft(constr, 5)
        maxSATSolver.assertSoft(KAndNaryExpr(this, listOf(a, b, z)), 2)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(
            maxSATResult.satSoftConstraints.size == 1 && maxSATResult.satSoftConstraints[0].weight == 5 &&
                    maxSATResult.satSoftConstraints[0].constraint == constr
        )
    }

    @Test
    fun oneScopePushPopTest() = with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)
        val a by boolSort
        val b by boolSort

        maxSATSolver.assert(a or b)

        maxSATSolver.assertSoft(!a or b, 1)

        val maxSATResult = maxSATSolver.checkMaxSAT()
        assertSoftConstraintsSAT(listOf(SoftConstraint(!a or b, 1)), maxSATResult.satSoftConstraints)

        maxSATSolver.push()

        maxSATSolver.assertSoft(a or !b, 1)
        maxSATSolver.assertSoft(!a or !b, 1)
        val maxSATResultScoped = maxSATSolver.checkMaxSAT()
        assertTrue(maxSATResultScoped.satSoftConstraints.size == 2)

        maxSATSolver.pop()

        assertSoftConstraintsSAT(listOf(SoftConstraint(!a or b, 1)), maxSATResult.satSoftConstraints)
    }

    @Test
    fun threeScopesPushPopTests() = with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)

        val a by boolSort
        val b by boolSort
        val c by boolSort

        maxSATSolver.assert(a)

        maxSATSolver.push()
        maxSATSolver.assertSoft(a or b, 1)
        maxSATSolver.assertSoft(c or b, 1)
        val maxSATResult = maxSATSolver.checkMaxSAT()
        assertTrue(maxSATResult.satSoftConstraints.size == 2)

        maxSATSolver.push()
        maxSATSolver.assertSoft(!b and !c, 2)
        val maxSATResult2 = maxSATSolver.checkMaxSAT()
        assertTrue(maxSATResult2.satSoftConstraints.size == 1)

        maxSATSolver.push()
        maxSATSolver.assertSoft(a or c, 1)
        val maxSATResult3 = maxSATSolver.checkMaxSAT()
        //assertTrue(maxSATResult3.satSoftConstraints)

        maxSATSolver.pop(2u)

        maxSATSolver.pop()
    }

    private fun assertSoftConstraintsSAT(constraintsToAssert: List<SoftConstraint>,
                                         satConstraints: List<SoftConstraint>) {
        for (constraint in constraintsToAssert) {
            assertTrue(satConstraints.any { constraint.constraint.internEquals(it.constraint) &&
                        constraint.weight == it.weight })
        }
    }
}
