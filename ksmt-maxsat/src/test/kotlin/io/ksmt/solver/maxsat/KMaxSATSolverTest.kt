package io.ksmt.solver.maxsat

import io.ksmt.KContext
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.utils.getValue
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class KMaxSATSolverTest {
    @Test
    fun noSoftConstraintsSatTest() = with(KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)
        val a by boolSort
        val b by boolSort
        val c by boolSort
        maxSATSolver.assert(a)
        maxSATSolver.assert(b)
        maxSATSolver.assert(c)
        maxSATSolver.assertSoft(a and !c, 3u)
        maxSATSolver.assertSoft(!a, 5u)

        runBlocking {
            val maxSATResult = maxSATSolver.checkMaxSAT()

            assertTrue(maxSATResult.hardConstraintsSATStatus == KSolverStatus.SAT)
            assertTrue(maxSATResult.maxSATSucceeded)
            assertTrue(maxSATResult.satSoftConstraints.isEmpty())
        }
    }

    @Test
    fun oneOfTwoSoftConstraintsSatTest() = with(KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)
        val a by boolSort
        val b by boolSort

        maxSATSolver.assert(a or b)
        maxSATSolver.assert(!a or b)

        maxSATSolver.assertSoft(a or !b, 2u)
        maxSATSolver.assertSoft(!a or !b, 3u)

        runBlocking {
            val maxSATResult = maxSATSolver.checkMaxSAT()

            assertTrue(maxSATResult.hardConstraintsSATStatus == KSolverStatus.SAT)
            assertTrue(maxSATResult.maxSATSucceeded)
            assertTrue(maxSATResult.satSoftConstraints.size == 1)
            assertSoftConstraintsSat(listOf(SoftConstraint(!a or !b, 3u)), maxSATResult.satSoftConstraints)
        }
    }

    @Test
    fun twoOfThreeSoftConstraintsSatTest() = with(KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)
        val a by boolSort
        val b by boolSort

        maxSATSolver.assert(a or b)

        maxSATSolver.assertSoft(!a or b, 4u)
        maxSATSolver.assertSoft(a or !b, 6u)
        maxSATSolver.assertSoft(!a or !b, 2u)

        runBlocking {
            val maxSATResult = maxSATSolver.checkMaxSAT()

            assertTrue(maxSATResult.hardConstraintsSATStatus == KSolverStatus.SAT)
            assertTrue(maxSATResult.maxSATSucceeded)
            assertTrue(maxSATResult.satSoftConstraints.size == 2)
            val softConstraintsToAssertSAT =
                listOf(SoftConstraint(!a or b, 4u), SoftConstraint(a or !b, 6u))
            assertSoftConstraintsSat(softConstraintsToAssertSAT, maxSATResult.satSoftConstraints)
        }
    }

    @Test
    fun sameExpressionSoftConstraintsSatTest() = with(KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)

        val x by boolSort
        val y by boolSort

        maxSATSolver.assert(x or y)

        maxSATSolver.assertSoft(!x or y, 7u)
        maxSATSolver.assertSoft(x or !y, 6u)
        maxSATSolver.assertSoft(!x or !y, 3u)
        maxSATSolver.assertSoft(!x or !y, 4u)

        runBlocking {
            val maxSATResult = maxSATSolver.checkMaxSAT()

            assertTrue(maxSATResult.hardConstraintsSATStatus == KSolverStatus.SAT)
            assertTrue(maxSATResult.maxSATSucceeded)
            assertTrue(maxSATResult.satSoftConstraints.size == 3)
            assertSoftConstraintsSat(
                listOf(
                    SoftConstraint(!x or y, 7u),
                    SoftConstraint(!x or !y, 3u),
                    SoftConstraint(!x or !y, 4u),
                ),
                maxSATResult.satSoftConstraints,
            )
        }
    }

    @Test
    fun sameExpressionSoftConstraintsUnsatTest() = with(KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)

        val x by boolSort
        val y by boolSort

        maxSATSolver.assert(x or y)

        maxSATSolver.assertSoft(!x or y, 6u)
        maxSATSolver.assertSoft(x or !y, 6u)
        maxSATSolver.assertSoft(!x or !y, 3u)
        maxSATSolver.assertSoft(!x or !y, 2u)

        runBlocking {
            val maxSATResult = maxSATSolver.checkMaxSAT()

            assertTrue(maxSATResult.hardConstraintsSATStatus == KSolverStatus.SAT)
            assertTrue(maxSATResult.maxSATSucceeded)
            assertTrue(maxSATResult.satSoftConstraints.size == 2)
            assertSoftConstraintsSat(
                listOf(SoftConstraint(!x or y, 6u), SoftConstraint(x or !y, 6u)),
                maxSATResult.satSoftConstraints,
            )
        }
    }

    @Test
    fun chooseOneConstraintByWeightTest() = with(KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)
        val z by boolSort
        val a by boolSort
        val b by boolSort

        maxSATSolver.assert(z)
        maxSATSolver.assertSoft(a and b, 1u)
        maxSATSolver.assertSoft(!a and !b, 5u)
        maxSATSolver.assertSoft(a and b and z, 2u)

        runBlocking {
            val maxSATResult = maxSATSolver.checkMaxSAT()

            assertTrue(maxSATResult.satSoftConstraints.size == 1)
            assertSoftConstraintsSat(listOf(SoftConstraint(!a and !b, 5u)), maxSATResult.satSoftConstraints)
        }
    }

    @Test
    fun inequalitiesTest() = with(KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)

        val x by intSort
        val y by intSort

        val a1 = x gt 0.expr
        val a2 = x lt y
        val a3 = x + y le 0.expr

        maxSATSolver.assert(a3 eq a1)
        maxSATSolver.assert(a3 or a2)

        maxSATSolver.assertSoft(a3, 3u)
        maxSATSolver.assertSoft(!a3, 5u)
        maxSATSolver.assertSoft(!a1, 10u)
        maxSATSolver.assertSoft(!a2, 3u)

        runBlocking {
            val maxSATResult = maxSATSolver.checkMaxSAT()

            assertTrue(maxSATResult.satSoftConstraints.size == 2)
            assertSoftConstraintsSat(
                listOf(SoftConstraint(!a3, 5u), SoftConstraint(!a1, 10u)),
                maxSATResult.satSoftConstraints,
            )
        }
    }

    @Test
    fun oneScopePushPopTest() = with(KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)
        val a by boolSort
        val b by boolSort

        maxSATSolver.assert(a or b)

        maxSATSolver.assertSoft(!a or b, 1u)

        runBlocking {
            val maxSATResult = maxSATSolver.checkMaxSAT()
            assertSoftConstraintsSat(listOf(SoftConstraint(!a or b, 1u)), maxSATResult.satSoftConstraints)

            maxSATSolver.push()

            maxSATSolver.assertSoft(a or !b, 1u)
            maxSATSolver.assertSoft(!a or !b, 1u)
            val maxSATResultScoped = maxSATSolver.checkMaxSAT()
            assertTrue(maxSATResultScoped.satSoftConstraints.size == 2)

            maxSATSolver.pop()

            assertSoftConstraintsSat(listOf(SoftConstraint(!a or b, 1u)), maxSATResult.satSoftConstraints)
        }
    }

    @Test
    fun threeScopesPushPopTest() = with(KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)

        val a by boolSort
        val b by boolSort
        val c by boolSort

        maxSATSolver.assert(a)

        maxSATSolver.push()
        maxSATSolver.assertSoft(a or b, 1u)
        maxSATSolver.assertSoft(c or b, 1u)

        runBlocking {
            val maxSATResult = maxSATSolver.checkMaxSAT()
            assertTrue(maxSATResult.satSoftConstraints.size == 2)

            maxSATSolver.push()
            maxSATSolver.assertSoft(!b and !c, 2u)
            val maxSATResult2 = maxSATSolver.checkMaxSAT()
            assertTrue(maxSATResult2.satSoftConstraints.size == 2)

            maxSATSolver.push()
            maxSATSolver.assertSoft(a or c, 1u)
            val maxSATResult3 = maxSATSolver.checkMaxSAT()
            assertTrue(maxSATResult3.satSoftConstraints.size == 3)

            maxSATSolver.pop(2u)
            val maxSATResult4 = maxSATSolver.checkMaxSAT()
            assertTrue(maxSATResult4.satSoftConstraints.size == 2)

            maxSATSolver.pop()
            val maxSATResult5 = maxSATSolver.checkMaxSAT()
            assertTrue(maxSATResult5.satSoftConstraints.isEmpty())
        }
    }

    private fun assertSoftConstraintsSat(
        constraintsToAssert: List<SoftConstraint>,
        satConstraints: List<SoftConstraint>,
    ) {
        for (constraint in constraintsToAssert) {
            assertTrue(
                satConstraints.any {
                    constraint.expression.internEquals(it.expression) && constraint.weight == it.weight
                },
            )
        }
    }
}
