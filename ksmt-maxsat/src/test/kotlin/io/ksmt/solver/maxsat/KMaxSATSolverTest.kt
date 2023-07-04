package io.ksmt.solver.maxsat

import io.ksmt.KContext
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.utils.getValue
import io.ksmt.utils.mkConst
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
        maxSATSolver.assertSoft(a and !c, 3)
        maxSATSolver.assertSoft(!a, 5)

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

        maxSATSolver.assertSoft(a or !b, 2)
        maxSATSolver.assertSoft(!a or !b, 3)

        runBlocking {
            val maxSATResult = maxSATSolver.checkMaxSAT()

            assertTrue(maxSATResult.hardConstraintsSATStatus == KSolverStatus.SAT)
            assertTrue(maxSATResult.maxSATSucceeded)
            assertTrue(maxSATResult.satSoftConstraints.size == 1)
            assertSoftConstraintsSat(listOf(SoftConstraint(!a or !b, 3)), maxSATResult.satSoftConstraints)
        }
    }

    @Test
    fun twoOfThreeSoftConstraintsSatTest() = with(KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)
        val a by boolSort
        val b by boolSort

        maxSATSolver.assert(a or b)

        maxSATSolver.assertSoft(!a or b, 4)
        maxSATSolver.assertSoft(a or !b, 6)
        maxSATSolver.assertSoft(!a or !b, 2)

        runBlocking {
            val maxSATResult = maxSATSolver.checkMaxSAT()

            assertTrue(maxSATResult.hardConstraintsSATStatus == KSolverStatus.SAT)
            assertTrue(maxSATResult.maxSATSucceeded)
            assertTrue(maxSATResult.satSoftConstraints.size == 2)
            val softConstraintsToAssertSAT =
                listOf(SoftConstraint(!a or b, 4), SoftConstraint(a or !b, 6))
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

        maxSATSolver.assertSoft(!x or y, 7)
        maxSATSolver.assertSoft(x or !y, 6)
        maxSATSolver.assertSoft(!x or !y, 3)
        maxSATSolver.assertSoft(!x or !y, 4)

        runBlocking {
            val maxSATResult = maxSATSolver.checkMaxSAT()

            assertTrue(maxSATResult.hardConstraintsSATStatus == KSolverStatus.SAT)
            assertTrue(maxSATResult.maxSATSucceeded)
            assertTrue(maxSATResult.satSoftConstraints.size == 3)
            assertSoftConstraintsSat(
                listOf(
                    SoftConstraint(!x or y, 7),
                    SoftConstraint(!x or !y, 3),
                    SoftConstraint(!x or !y, 4),
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

        maxSATSolver.assertSoft(!x or y, 6)
        maxSATSolver.assertSoft(x or !y, 6)
        maxSATSolver.assertSoft(!x or !y, 3)
        maxSATSolver.assertSoft(!x or !y, 2)

        runBlocking {
            val maxSATResult = maxSATSolver.checkMaxSAT()

            assertTrue(maxSATResult.hardConstraintsSATStatus == KSolverStatus.SAT)
            assertTrue(maxSATResult.maxSATSucceeded)
            assertTrue(maxSATResult.satSoftConstraints.size == 2)
            assertSoftConstraintsSat(
                listOf(SoftConstraint(!x or y, 6), SoftConstraint(x or !y, 6)),
                maxSATResult.satSoftConstraints,
            )
        }
    }

    @Test
    fun chooseOneConstraintByWeightTest() = with(KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)
        val z = boolSort.mkConst("z")
        val a = boolSort.mkConst("a")
        val b = boolSort.mkConst("b")

        maxSATSolver.assert(z)
        maxSATSolver.assertSoft(a and b, 1)
        maxSATSolver.assertSoft(!a and !b, 5)
        maxSATSolver.assertSoft(a and b and z, 2)

        runBlocking {
            val maxSATResult = maxSATSolver.checkMaxSAT()

            assertTrue(maxSATResult.satSoftConstraints.size == 1)
            assertSoftConstraintsSat(listOf(SoftConstraint(!a and !b, 5)), maxSATResult.satSoftConstraints)
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

        maxSATSolver.assertSoft(a3, 3)
        maxSATSolver.assertSoft(!a3, 5)
        maxSATSolver.assertSoft(!a1, 10)
        maxSATSolver.assertSoft(!a2, 3)

        runBlocking {
            val maxSATResult = maxSATSolver.checkMaxSAT()

            assertTrue(maxSATResult.satSoftConstraints.size == 2)
            assertSoftConstraintsSat(
                listOf(SoftConstraint(!a3, 5), SoftConstraint(!a1, 10)),
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

        maxSATSolver.assertSoft(!a or b, 1)

        runBlocking {
            val maxSATResult = maxSATSolver.checkMaxSAT()
            assertSoftConstraintsSat(listOf(SoftConstraint(!a or b, 1)), maxSATResult.satSoftConstraints)

            maxSATSolver.push()

            maxSATSolver.assertSoft(a or !b, 1)
            maxSATSolver.assertSoft(!a or !b, 1)
            val maxSATResultScoped = maxSATSolver.checkMaxSAT()
            assertTrue(maxSATResultScoped.satSoftConstraints.size == 2)

            maxSATSolver.pop()

            assertSoftConstraintsSat(listOf(SoftConstraint(!a or b, 1)), maxSATResult.satSoftConstraints)
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
        maxSATSolver.assertSoft(a or b, 1)
        maxSATSolver.assertSoft(c or b, 1)

        runBlocking {
            val maxSATResult = maxSATSolver.checkMaxSAT()
            assertTrue(maxSATResult.satSoftConstraints.size == 2)

            maxSATSolver.push()
            maxSATSolver.assertSoft(!b and !c, 2)
            val maxSATResult2 = maxSATSolver.checkMaxSAT()
            assertTrue(maxSATResult2.satSoftConstraints.size == 2)

            maxSATSolver.push()
            maxSATSolver.assertSoft(a or c, 1)
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
                    constraint.expression.internEquals(it.expression) &&
                        constraint.weight == it.weight
                },
            )
        }
    }
}
